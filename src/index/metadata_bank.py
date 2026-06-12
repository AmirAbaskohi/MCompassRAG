"""Offline metadata bank: the precomputed index the paper caches per chunk.

For chunks ``C = {c_1..c_N}`` it caches:

* ``embeddings``      ``e_c``  ``(N, d)`` retriever embeddings ``f_psi(c)``
* ``theta``                    ``(N, K)`` topic distributions (rows on the simplex)
* ``meta_embeddings`` ``m``    ``(N, d)`` ``sum_k theta_k t_k`` over **all K** topics
                                          (the Selector input)
* ``topic_summaries`` ``g_c``  ``(N, d)`` ``sum_{k in top-M} theta_k t_k``
* ``chunk_reps``      ``r_c``  ``(N, 2d)`` ``[e_c ; g_c]``
* ``centroids``       ``T``    ``(K, d)`` topic centroids

``m`` (all-K) and ``g_c`` (top-M) are kept strictly separate. All tensors are
stored as float32 on CPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from src.encoders.retriever_encoder import RetrieverEncoder
from src.index.chunking import Chunk

if TYPE_CHECKING:  # topic_models imported lazily to avoid a heavy/circular import
    from topic_models.base import TopicModel

_TENSOR_KEYS = (
    "embeddings",
    "theta",
    "meta_embeddings",
    "topic_summaries",
    "chunk_reps",
    "centroids",
)


@dataclass
class IndexConfig:
    top_m: int = 8  # M for g_c (and later g_q)
    chunk_batch_size: int = 16
    topic_batch_size: int = 8


class MetadataBank:
    """Cached offline index. All tensors float32 on CPU."""

    def __init__(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: torch.Tensor,
        theta: torch.Tensor,
        meta_embeddings: torch.Tensor,
        topic_summaries: torch.Tensor,
        chunk_reps: torch.Tensor,
        centroids: torch.Tensor,
        cfg: IndexConfig,
        centroid_source: str = "decoder",
        centroid_normalize: bool = True,
    ):
        self.ids = ids
        self.texts = texts
        self.embeddings = embeddings
        self.theta = theta
        self.meta_embeddings = meta_embeddings
        self.topic_summaries = topic_summaries
        self.chunk_reps = chunk_reps
        self.centroids = centroids
        self.cfg = cfg
        self.centroid_source = centroid_source
        self.centroid_normalize = centroid_normalize

        self.d = int(embeddings.shape[1])
        self.K = int(theta.shape[1])

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        encoder: RetrieverEncoder | None,
        topic_provider: TopicModel,
        cfg: IndexConfig,
    ) -> "MetadataBank":
        """Build the offline index, batched, in the order the paper specifies.

        Consumes the :class:`TopicModel` interface (any backend). ``encoder`` may be
        ``None``, in which case the topic model's own retriever encoder is used.
        """
        if encoder is None:
            encoder = topic_provider.encoder
        ids = [c.id for c in chunks]
        texts = [c.text for c in chunks]

        # 2. Chunk embeddings e_c = f_psi(c) (no instruction prefix).
        e_c = encoder.encode(texts, batch_size=cfg.chunk_batch_size, is_query=False)
        e_c = e_c.float().cpu()  # (N, d)

        # 3. Topic distributions theta.
        theta = topic_provider.encode_topic_distribution(
            texts, batch_size=cfg.topic_batch_size
        )
        theta = theta.float().cpu()  # (N, K)

        # 4. Centroids T (fit empirical responsibilities in retriever space first
        #    when the backend has no native centroids or empirical was requested).
        if (not topic_provider.supports_native_centroids) or (
            topic_provider.effective_centroid_source == "empirical"
        ):
            topic_provider.fit_empirical_centroids(e_c, theta)
        T = topic_provider.get_topic_centroids().float().cpu()  # (K, d)

        # 5. Selector input m = theta @ T over ALL K topics.
        m = (theta @ T).float()  # (N, d)

        # 6. Chunk-representation summary g_c over top-M topics only.
        g_c = topic_provider.topic_summary(theta, top_m=cfg.top_m).float().cpu()  # (N, d)

        # 7. Chunk representation r_c = [e_c ; g_c].
        r_c = torch.cat([e_c, g_c], dim=-1).float()  # (N, 2d)

        d = e_c.shape[1]
        # --- assertions ---
        for name, t in (
            ("embeddings", e_c),
            ("theta", theta),
            ("meta_embeddings", m),
            ("topic_summaries", g_c),
            ("chunk_reps", r_c),
            ("centroids", T),
        ):
            if not torch.isfinite(t).all():
                raise ValueError(f"Non-finite values in {name}")

        if not (d == encoder.dim == T.shape[1]):
            raise ValueError(
                f"Dimension mismatch: d={d}, encoder.dim={encoder.dim}, "
                f"T.shape[1]={T.shape[1]} must all be equal."
            )
        if theta.shape[0] > 0:
            row_sums = theta.sum(dim=-1)
            if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4):
                raise ValueError("theta rows do not sum to 1 (atol=1e-4)")
        if not torch.allclose(r_c[:, :d], e_c, atol=1e-6):
            raise ValueError("r_c[:, :d] does not match e_c")
        if not torch.allclose(r_c[:, d:], g_c, atol=1e-6):
            raise ValueError("r_c[:, d:] does not match g_c")

        return cls(
            ids=ids,
            texts=texts,
            embeddings=e_c.contiguous(),
            theta=theta.contiguous(),
            meta_embeddings=m.contiguous(),
            topic_summaries=g_c.contiguous(),
            chunk_reps=r_c.contiguous(),
            centroids=T.contiguous(),
            cfg=cfg,
            centroid_source=topic_provider.effective_centroid_source,
            centroid_normalize=topic_provider.centroid_normalize,
        )

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)

        tensors = {
            "embeddings": self.embeddings.contiguous().cpu(),
            "theta": self.theta.contiguous().cpu(),
            "meta_embeddings": self.meta_embeddings.contiguous().cpu(),
            "topic_summaries": self.topic_summaries.contiguous().cpu(),
            "chunk_reps": self.chunk_reps.contiguous().cpu(),
            "centroids": self.centroids.contiguous().cpu(),
        }
        save_file(tensors, os.path.join(out_dir, "tensors.safetensors"))

        with open(os.path.join(out_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for cid, text in zip(self.ids, self.texts):
                f.write(json.dumps({"id": cid, "text": text}, ensure_ascii=False) + "\n")

        meta = {
            "d": self.d,
            "K": self.K,
            "top_m": self.cfg.top_m,
            "n_chunks": len(self.ids),
            "centroid_source": self.centroid_source,
            "centroid_normalize": self.centroid_normalize,
            "chunk_batch_size": self.cfg.chunk_batch_size,
            "topic_batch_size": self.cfg.topic_batch_size,
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, in_dir: str, mmap: bool = True) -> "MetadataBank":
        with open(os.path.join(in_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        ids: list[str] = []
        texts: list[str] = []
        with open(os.path.join(in_dir, "chunks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ids.append(rec["id"])
                texts.append(rec["text"])

        tensors: dict[str, torch.Tensor] = {}
        st_path = os.path.join(in_dir, "tensors.safetensors")
        # safe_open with mmap-backed lazy loading when mmap=True; otherwise force a
        # materialized copy of each tensor.
        with safe_open(st_path, framework="pt", device="cpu") as st:
            for key in _TENSOR_KEYS:
                t = st.get_tensor(key)
                tensors[key] = t if mmap else t.clone()

        cfg = IndexConfig(
            top_m=int(meta["top_m"]),
            chunk_batch_size=int(meta.get("chunk_batch_size", 16)),
            topic_batch_size=int(meta.get("topic_batch_size", 8)),
        )

        return cls(
            ids=ids,
            texts=texts,
            embeddings=tensors["embeddings"],
            theta=tensors["theta"],
            meta_embeddings=tensors["meta_embeddings"],
            topic_summaries=tensors["topic_summaries"],
            chunk_reps=tensors["chunk_reps"],
            centroids=tensors["centroids"],
            cfg=cfg,
            centroid_source=meta.get("centroid_source", "decoder"),
            centroid_normalize=bool(meta.get("centroid_normalize", True)),
        )

    def to(self, device: str) -> "MetadataBank":
        self.embeddings = self.embeddings.to(device)
        self.theta = self.theta.to(device)
        self.meta_embeddings = self.meta_embeddings.to(device)
        self.topic_summaries = self.topic_summaries.to(device)
        self.chunk_reps = self.chunk_reps.to(device)
        self.centroids = self.centroids.to(device)
        return self

    def __len__(self) -> int:
        return len(self.ids)
