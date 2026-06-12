"""Training dataset + collation for CompassRetriever distillation.

Consumes Phase 4's ``train.jsonl`` and a Phase 2 ``MetadataBank`` over the same
corpus. The student always encodes the BASE query (``is_query=True``); the
teacher's ``z_t`` (from the EXPANDED query) is the soft target. Because the
encoder is frozen, query embeddings are precomputed once and cached.
"""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import load_file, save_file


@torch.no_grad()
def precompute_query_embeddings(
    records: list[dict],
    encoder,
    batch_size: int = 16,
    cache_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """Encode each ``record['base_query']`` with ``is_query=True``.

    Returns ``{query_id: e_q}`` with each ``e_q`` a ``(d,)`` float32 tensor. If
    ``cache_path`` is given, embeddings (stacked) + id order are saved to
    ``{cache_path}.safetensors`` / ``{cache_path}.json`` and reused on subsequent
    calls (skipping encoding).
    """
    st_path = f"{cache_path}.safetensors" if cache_path else None
    json_path = f"{cache_path}.json" if cache_path else None

    if cache_path and os.path.exists(st_path) and os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            order = json.load(f)
        stacked = load_file(st_path)["embeddings"].float()
        return {qid: stacked[i] for i, qid in enumerate(order)}

    query_ids = [r["query_id"] for r in records]
    base_queries = [r["base_query"] for r in records]

    embs: list[torch.Tensor] = []
    for start in range(0, len(base_queries), batch_size):
        sub = base_queries[start : start + batch_size]
        e = encoder.encode(sub, is_query=True).float().cpu()  # (b, d)
        embs.append(e)

    if embs:
        stacked = torch.cat(embs, dim=0)  # (Q, d)
    else:
        stacked = torch.empty(0, 0)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or ".", exist_ok=True)
        save_file({"embeddings": stacked.contiguous()}, st_path)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(query_ids, f)

    return {qid: stacked[i] for i, qid in enumerate(query_ids)}


class CompassTrainDataset(torch.utils.data.Dataset):
    """One item per query: precomputed ``e_q`` + resolved candidate rows/labels."""

    def __init__(
        self,
        jsonl_path: str,
        bank,
        query_emb: dict[str, torch.Tensor],
        max_candidates_per_query: int = 16,
        seed: int = 13,
    ):
        self.max_candidates_per_query = int(max_candidates_per_query)
        id2row = {cid: i for i, cid in enumerate(bank.ids)}
        rng = __import__("random").Random(seed)

        self.items: list[dict] = []
        n_dropped_candidates = 0
        n_skipped_records = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                qid = rec["query_id"]
                if qid not in query_emb:
                    n_skipped_records += 1
                    continue

                pos_rows, pos_y, pos_z, pos_ht = [], [], [], []
                neg_rows, neg_y, neg_z, neg_ht = [], [], [], []
                for cand in rec["candidates"]:
                    cid = cand["chunk_id"]
                    row = id2row.get(cid)
                    if row is None:
                        n_dropped_candidates += 1
                        continue
                    y = float(cand["y"])
                    has_teacher = bool(cand["has_teacher"])
                    z = cand.get("z_t", None)
                    z_val = float(z) if (has_teacher and z is not None) else 0.0
                    is_pos = cand.get("role") == "positive" or y >= 0.5
                    if is_pos:
                        pos_rows.append(row)
                        pos_y.append(y)
                        pos_z.append(z_val)
                        pos_ht.append(has_teacher)
                    else:
                        neg_rows.append(row)
                        neg_y.append(y)
                        neg_z.append(z_val)
                        neg_ht.append(has_teacher)

                if len(pos_rows) == 0:
                    n_skipped_records += 1
                    continue

                # Always keep positives; sample the rest down to the budget.
                budget_for_neg = max(0, self.max_candidates_per_query - len(pos_rows))
                if len(neg_rows) > budget_for_neg:
                    idxs = list(range(len(neg_rows)))
                    keep = rng.sample(idxs, budget_for_neg)
                    keep.sort()
                    neg_rows = [neg_rows[k] for k in keep]
                    neg_y = [neg_y[k] for k in keep]
                    neg_z = [neg_z[k] for k in keep]
                    neg_ht = [neg_ht[k] for k in keep]

                cand_rows = pos_rows + neg_rows
                y_list = pos_y + neg_y
                z_list = pos_z + neg_z
                ht_list = pos_ht + neg_ht

                self.items.append(
                    {
                        "query_id": qid,
                        "e_q": query_emb[qid].float(),
                        "cand_rows": cand_rows,
                        "y": y_list,
                        "z_t": z_list,
                        "has_teacher": ht_list,
                    }
                )

        if n_dropped_candidates > 0:
            print(
                f"[CompassTrainDataset] dropped {n_dropped_candidates} candidates "
                "missing from the bank."
            )
        if n_skipped_records > 0:
            print(
                f"[CompassTrainDataset] skipped {n_skipped_records} records "
                "(no positive after resolution or missing query embedding)."
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i) -> dict:
        return self.items[i]


def collate_compass(batch: list[dict]) -> dict:
    """Pad a batch to ``P = max #candidates``; build masks.

    Returns a dict with ``e_q (B,d)``, ``cand_rows (B,P) long``,
    ``cand_mask (B,P)``, ``y (B,P)``, ``z_t (B,P)``, ``kd_mask (B,P)``.
    """
    B = len(batch)
    d = batch[0]["e_q"].shape[0]
    P = max(len(item["cand_rows"]) for item in batch)
    P = max(P, 1)

    e_q = torch.zeros(B, d, dtype=torch.float32)
    cand_rows = torch.zeros(B, P, dtype=torch.long)
    cand_mask = torch.zeros(B, P, dtype=torch.float32)
    y = torch.zeros(B, P, dtype=torch.float32)
    z_t = torch.zeros(B, P, dtype=torch.float32)
    kd_mask = torch.zeros(B, P, dtype=torch.float32)

    for b, item in enumerate(batch):
        e_q[b] = item["e_q"].float()
        n = len(item["cand_rows"])
        for j in range(n):
            cand_rows[b, j] = int(item["cand_rows"][j])
            cand_mask[b, j] = 1.0
            y[b, j] = float(item["y"][j])
            z_t[b, j] = float(item["z_t"][j])
            kd_mask[b, j] = 1.0 if bool(item["has_teacher"][j]) else 0.0

    # kd_mask must be has_teacher AND a real candidate.
    kd_mask = kd_mask * cand_mask
    # zero out z_t where not kd-valid (defensive; already 0 for missing).
    z_t = z_t * kd_mask

    query_ids = [item["query_id"] for item in batch]
    return {
        "e_q": e_q,
        "cand_rows": cand_rows,
        "cand_mask": cand_mask,
        "y": y,
        "z_t": z_t,
        "kd_mask": kd_mask,
        "query_ids": query_ids,
    }
