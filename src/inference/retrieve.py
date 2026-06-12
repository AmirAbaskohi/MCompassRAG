"""Inference: Algorithm 1 + the top-level ``CompassRAG`` orchestrator.

Composes already-built, frozen pieces (``RetrieverEncoder``, ``MetadataBank``, a
trained ``CompassRetriever``) into a serving retriever. No LLM calls happen at
inference time — that is the whole point of the method.

Algorithm 1 mapping:
    e_q  = f_psi(q)                         -> RetrieverEncoder.encode(is_query=True)
    r_q  = select + abstract + g_q          -> CompassRetriever.build_query_rep
    z_j  = MLP_phi([r_q ; r_{c_j}]) for all -> CompassRetriever.score_all
    C_k  = top-k_j z_j                       -> topk over scores
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.encoders.retriever_encoder import RetrieverEncoder
from src.index.metadata_bank import MetadataBank
from src.models.compass_retriever import CompassRetriever


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    rank: int  # 0-based, 0 = top


@dataclass
class CompassRAGConfig:
    k: int = 5
    score_block_size: int = 4096
    query_batch_size: int = 16
    device: str | None = None


def _pick_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class CompassRAG:
    def __init__(
        self,
        encoder: RetrieverEncoder,
        model: CompassRetriever,
        bank: MetadataBank,
        cfg: CompassRAGConfig = CompassRAGConfig(),
    ):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self.encoder = encoder
        self.bank = bank

        # Consistency assertions between the trained model and the served index.
        if model.cfg.d != bank.d:
            raise ValueError(f"model.cfg.d {model.cfg.d} != bank.d {bank.d}")
        if model.cfg.K != bank.K:
            raise ValueError(f"model.cfg.K {model.cfg.K} != bank.K {bank.K}")
        if model.cfg.top_m != bank.cfg.top_m:
            raise ValueError(
                f"model.cfg.top_m {model.cfg.top_m} != bank.cfg.top_m {bank.cfg.top_m}"
            )

        # Move the index + model onto the serving device and attach the buffers.
        self.bank.to(str(self.device))
        self.bank.chunk_reps = self.bank.chunk_reps.to(self.device).float()
        self.model = model.to(self.device)
        self.model.set_index(
            bank.meta_embeddings.to(self.device),
            bank.theta.to(self.device),
            bank.centroids.to(self.device),
        )
        # Deterministic inference: disable dropout in abstraction/scorer.
        self.model.eval()

        # id <-> row / text maps.
        self.row_to_id = list(bank.ids)
        self.id2text = {cid: txt for cid, txt in zip(bank.ids, bank.texts)}

    @classmethod
    def from_pretrained(
        cls,
        bank_dir: str,
        model_ckpt: str,
        encoder: RetrieverEncoder | None = None,
        backbone_name: str = "Qwen/Qwen3-Embedding-4B",
        cemtm_ckpt: str | None = None,
        num_topics: int = 100,
        device: str | None = None,
        cfg: CompassRAGConfig = CompassRAGConfig(),
    ) -> "CompassRAG":
        """Reconstruct a serving retriever from a saved bank + model checkpoint.

        The bank already holds ``centroids``/``theta``/``m``, so CEMTM is NOT needed
        to serve. ``cemtm_ckpt`` / ``num_topics`` are accepted for parity/optional
        re-embedding and may be unused here.
        """
        from src.training.trainer import CompassTrainer

        bank = MetadataBank.load(bank_dir, mmap=True)
        model = CompassTrainer.load_model(model_ckpt)

        if encoder is None:
            encoder = RetrieverEncoder(model_name=backbone_name, device=device)

        if device is not None:
            cfg = CompassRAGConfig(
                k=cfg.k,
                score_block_size=cfg.score_block_size,
                query_batch_size=cfg.query_batch_size,
                device=device,
            )
        return cls(encoder, model, bank, cfg)

    @torch.no_grad()
    def build_query_reps(
        self, queries: list[str], batch_size: int | None = None
    ) -> torch.Tensor:
        """Encode queries and build their representations ``r_q`` ``(B, 2d)``."""
        bs = batch_size or self.cfg.query_batch_size
        reps: list[torch.Tensor] = []
        for start in range(0, len(queries), bs):
            sub = queries[start : start + bs]
            e_q = self.encoder.encode(sub, is_query=True).float().to(self.device)
            r_q, _ = self.model.build_query_rep(e_q)
            reps.append(r_q.float())
        if not reps:
            return torch.empty(0, 2 * self.bank.d, device=self.device)
        return torch.cat(reps, dim=0)

    def _assemble(self, scores_row: torch.Tensor, k: int) -> list[RetrievalResult]:
        k = min(k, scores_row.shape[0])
        top_scores, top_idx = torch.topk(scores_row, k, largest=True, sorted=True)
        results: list[RetrievalResult] = []
        for rank in range(k):
            row = int(top_idx[rank].item())
            cid = self.row_to_id[row]
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    text=self.id2text.get(cid, ""),
                    score=float(top_scores[rank].item()),
                    rank=rank,
                )
            )
        return results

    @torch.no_grad()
    def retrieve(self, query: str, k: int | None = None) -> list[RetrievalResult]:
        """Algorithm 1 for a single query."""
        k = self.cfg.k if k is None else k
        k = min(k, len(self.bank))
        r_q = self.build_query_reps([query])  # (1, 2d)
        scores = self.model.score_all(
            r_q, self.bank.chunk_reps, self.cfg.score_block_size
        )[0]  # (N,)
        return self._assemble(scores, k)

    @torch.no_grad()
    def retrieve_batch(
        self,
        queries: list[str],
        k: int | None = None,
        batch_size: int | None = None,
    ) -> list[list[RetrievalResult]]:
        """Vectorized retrieval; output order matches input order."""
        k = self.cfg.k if k is None else k
        k = min(k, len(self.bank))
        if not queries:
            return []
        r_q = self.build_query_reps(queries, batch_size=batch_size)  # (B, 2d)
        scores = self.model.score_all(
            r_q, self.bank.chunk_reps, self.cfg.score_block_size
        )  # (B, N)
        return [self._assemble(scores[b], k) for b in range(scores.shape[0])]

    @torch.no_grad()
    def answer(self, query: str, generate_fn, k: int | None = None) -> str:
        """Retrieve top-k contexts, then defer to a user-provided ``generate_fn``."""
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        contexts = self.retrieve(query, k=k)
        return generate_fn(query, contexts)
