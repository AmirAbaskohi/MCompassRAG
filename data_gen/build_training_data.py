"""Distillation data pipeline: generate queries, mine negatives, judge with teacher.

Produces ``train.jsonl`` (one record per query), ``chunks.jsonl`` (id/text for
Phase 5), and ``datagen_config.json``. The teacher always judges the EXPANDED
query; student-side hard-negative mining uses the BASE query.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass

import torch

from data_gen.negatives import mine_hard_candidates, sample_random_negatives
from data_gen.openrouter_client import ChatClient, OpenRouterClient
from data_gen.query_gen import OrderedChunk, generate_queries, neighbors
from data_gen.teacher import judge_candidates


@dataclass
class DataGenConfig:
    n_target_chunks: int = 2000
    queries_per_chunk: int = 10
    n_random_negatives: int = 4
    n_hard_candidates: int = 20
    n_hard_negatives: int = 4
    neighbor_window: int = 1
    query_gen_model: str = "openai/gpt-4o"
    teacher_model: str = "openai/gpt-4o"
    temperature_qgen: float = 0.7
    temperature_teacher: float = 0.0
    teacher_batch_size: int = 8
    trust_teacher_on_positive: bool = False  # False -> force positive y=1
    kd_on_random_negatives: bool = False
    keep_useful_highsim_as_positive: bool = False
    max_workers: int = 8
    seed: int = 13


def _neighbor_ids(
    target: OrderedChunk, by_doc: dict[str, list[OrderedChunk]], window: int
) -> set[str]:
    siblings = by_doc.get(target.doc_id, [])
    idx = None
    for i, c in enumerate(siblings):
        if c.id == target.id:
            idx = i
            break
    if idx is None:
        return set()
    prev_chunks = siblings[max(0, idx - window) : idx]
    next_chunks = siblings[idx + 1 : idx + 1 + window]
    return {c.id for c in prev_chunks + next_chunks}


def build_training_data(
    chunks: list[OrderedChunk],
    encoder,
    qgen_client: ChatClient,
    teacher_client: ChatClient,
    cfg: DataGenConfig,
    out_dir: str,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # 1. Group by doc_id, sort by position.
    by_doc: dict[str, list[OrderedChunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for doc_id in by_doc:
        by_doc[doc_id].sort(key=lambda c: c.position)

    # 2. Precompute chunk embeddings + id maps; write chunks.jsonl.
    chunk_ids = [c.id for c in chunks]
    id_to_text = {c.id: c.text for c in chunks}
    texts = [c.text for c in chunks]
    chunk_embeddings = encoder.encode(texts, is_query=False).float()  # (C, d)

    with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({"id": c.id, "text": c.text}, ensure_ascii=False) + "\n")

    # 3. Sample targets deterministically.
    rng = random.Random(cfg.seed)
    n_targets = min(cfg.n_target_chunks, len(chunks))
    targets = rng.sample(chunks, n_targets)

    # 4. Generate queries per target (concurrent, order-preserving).
    def _gen(target: OrderedChunk) -> list[dict]:
        prev_text, next_text = neighbors(target, by_doc, cfg.neighbor_window)
        return generate_queries(
            qgen_client,
            target,
            prev_text,
            next_text,
            cfg.queries_per_chunk,
            cfg.temperature_qgen,
        )

    per_target_queries = qgen_client.map(_gen, targets, max_workers=cfg.max_workers)

    # 5a. Build per-query tasks (deterministic hard-candidate mining via local encoder).
    @dataclass
    class _Task:
        target_id: str
        query_index: int
        base_query: str
        expanded_query: str
        hard_cands: list[str]
        neighbor_ids: set[str]

    tasks: list[_Task] = []
    for target, queries in zip(targets, per_target_queries):
        nb_ids = _neighbor_ids(target, by_doc, cfg.neighbor_window)
        for i, qp in enumerate(queries):
            hard_cands = mine_hard_candidates(
                qp["base_query"],
                positive_id=target.id,
                encoder=encoder,
                chunk_ids=chunk_ids,
                chunk_embeddings=chunk_embeddings,
                n_candidates=cfg.n_hard_candidates,
                exclude_ids=set(nb_ids),
            )
            tasks.append(
                _Task(
                    target_id=target.id,
                    query_index=i,
                    base_query=qp["base_query"],
                    expanded_query=qp["expanded_query"],
                    hard_cands=hard_cands,
                    neighbor_ids=nb_ids,
                )
            )

    # 5b. Teacher judging of {positive} ∪ hard_cands on the EXPANDED query (batched).
    def _judge(task: "_Task") -> dict[str, dict]:
        candidates = [(task.target_id, id_to_text[task.target_id])]
        candidates += [(cid, id_to_text[cid]) for cid in task.hard_cands]
        return judge_candidates(
            teacher_client,
            task.expanded_query,
            candidates,
            cfg.temperature_teacher,
            teacher_batch_size=cfg.teacher_batch_size,
        )

    judgments = teacher_client.map(_judge, tasks, max_workers=cfg.max_workers)

    # 6. Assemble + emit records (sequential to keep rng deterministic).
    counts = {
        "queries": 0,
        "positives": 0,
        "hard_negatives": 0,
        "random_negatives": 0,
        "discarded_false_negatives": 0,
        "kept_useful_as_positive": 0,
    }

    out_path = os.path.join(out_dir, "train.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for task, judg in zip(tasks, judgments):
            pos_id = task.target_id
            pos_j = judg.get(pos_id, {"relevant": True, "z_t": 0.0})

            candidates_out: list[dict] = []

            # Positive.
            if cfg.trust_teacher_on_positive:
                y_pos = int(bool(pos_j["relevant"]))
            else:
                y_pos = 1
            candidates_out.append(
                {
                    "chunk_id": pos_id,
                    "role": "positive",
                    "y": y_pos,
                    "z_t": float(pos_j["z_t"]),
                    "has_teacher": True,
                }
            )
            counts["positives"] += 1

            # Hard candidates: split by teacher judgment.
            hard_neg_ids: list[str] = []
            for cid in task.hard_cands:  # similarity-ordered (highest first)
                cj = judg.get(cid, {"relevant": False, "z_t": _safe_zt(judg, cid)})
                if cj["relevant"]:
                    # Teacher says useful -> potential false negative.
                    if cfg.keep_useful_highsim_as_positive:
                        candidates_out.append(
                            {
                                "chunk_id": cid,
                                "role": "positive",
                                "y": 1,
                                "z_t": float(cj["z_t"]),
                                "has_teacher": True,
                            }
                        )
                        counts["kept_useful_as_positive"] += 1
                    else:
                        counts["discarded_false_negatives"] += 1
                else:
                    if len(hard_neg_ids) < cfg.n_hard_negatives:
                        hard_neg_ids.append(cid)
                        candidates_out.append(
                            {
                                "chunk_id": cid,
                                "role": "hard_negative",
                                "y": 0,
                                "z_t": float(cj["z_t"]),
                                "has_teacher": True,
                            }
                        )
                        counts["hard_negatives"] += 1

            # Random negatives.
            exclude = set(task.hard_cands) | set(task.neighbor_ids)
            rand_ids = sample_random_negatives(
                pos_id, chunk_ids, cfg.n_random_negatives, exclude, rng
            )
            rand_judg: dict[str, dict] = {}
            if cfg.kd_on_random_negatives and rand_ids:
                rand_candidates = [(cid, id_to_text[cid]) for cid in rand_ids]
                rand_judg = judge_candidates(
                    teacher_client,
                    task.expanded_query,
                    rand_candidates,
                    cfg.temperature_teacher,
                    teacher_batch_size=cfg.teacher_batch_size,
                )
            for cid in rand_ids:
                if cfg.kd_on_random_negatives:
                    cj = rand_judg.get(cid, {"z_t": 0.0})
                    candidates_out.append(
                        {
                            "chunk_id": cid,
                            "role": "random_negative",
                            "y": 0,
                            "z_t": float(cj["z_t"]),
                            "has_teacher": True,
                        }
                    )
                else:
                    candidates_out.append(
                        {
                            "chunk_id": cid,
                            "role": "random_negative",
                            "y": 0,
                            "z_t": None,
                            "has_teacher": False,
                        }
                    )
                counts["random_negatives"] += 1

            record = {
                "query_id": f"q_{task.target_id}_{task.query_index}",
                "source_chunk_id": pos_id,
                "base_query": task.base_query,
                "expanded_query": task.expanded_query,
                "candidates": candidates_out,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["queries"] += 1

    # Write config + log counts.
    with open(os.path.join(out_dir, "datagen_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("[build_training_data] counts:")
    for k, v in counts.items():
        print(f"    {k}: {v}")
    print(f"[build_training_data] wrote {counts['queries']} queries to {out_path}")


def _safe_zt(judg: dict[str, dict], cid: str) -> float:
    entry = judg.get(cid)
    if entry is None:
        from data_gen.teacher import _prob_to_logit

        return _prob_to_logit(0.0)
    return float(entry.get("z_t", 0.0))


def load_ordered_chunks(path: str) -> list[OrderedChunk]:
    """Read ``{id,text,doc_id,position}`` records into :class:`OrderedChunk`s."""
    chunks: list[OrderedChunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chunks.append(
                OrderedChunk(
                    id=str(rec["id"]),
                    text=rec["text"],
                    doc_id=str(rec.get("doc_id", rec["id"])),
                    position=int(rec.get("position", 0)),
                )
            )
    return chunks


def build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        description="Generate LLM-teacher distillation training data (queries + negatives + judgments)."
    )
    p.add_argument("--config", required=True, help="Path to a gen_train_data YAML.")
    return p


def main(argv: list[str] | None = None) -> int:
    from src.config import bind, build_encoder, load_yaml, section

    args = build_arg_parser().parse_args(argv)
    cfg = load_yaml(args.config)

    encoder = build_encoder(cfg)

    dg_sec = dict(section(cfg, "datagen", default={}) or {})
    dcfg = bind(DataGenConfig, dg_sec)

    chunks_path = section(cfg, "corpus", "chunks_path")
    if not chunks_path:
        raise ValueError("config is missing corpus.chunks_path")
    chunks = load_ordered_chunks(chunks_path)

    reasoning_effort = dg_sec.get("reasoning_effort", "none")
    extra_body: dict = {}
    if reasoning_effort and str(reasoning_effort).lower() != "none":
        extra_body = {"reasoning": {"effort": reasoning_effort}}

    qgen_client = OpenRouterClient(model=dcfg.query_gen_model, extra_body=extra_body)
    teacher_client = OpenRouterClient(model=dcfg.teacher_model, extra_body=extra_body)

    out_dir = section(cfg, "output", "out_dir")
    if not out_dir:
        raise ValueError("config is missing output.out_dir")

    build_training_data(chunks, encoder, qgen_client, teacher_client, dcfg, out_dir)
    print(f"[build_training_data] wrote training data to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
