"""``TopicModel`` interface: the pluggable topic component for CompassRAG.

The paper states the method is "agnostic to the specific topic model, requiring
only that topics be embedded in the retriever's semantic space." This module
factors that contract into an ABC consumed by Phase 2 (``MetadataBank.build``) and
Phase 6 (``pipeline.build_index``):

    encode_topic_distribution(texts) -> theta (B, K) on the simplex
    get_topic_centroids()            -> T (K, d) in the retriever space
    topic_summary(theta, top_m)      -> g (B, d)
    fit_empirical_centroids(e, theta)-> T_emp (K, d)

Every backend takes ``K`` and the retriever encoder LM so centroids land in the
retriever space, and trains on a :class:`~topic_models.wikiweb2m.TopicCorpus`.
"""

from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:  # avoid import cycles; only needed for typing
    from src.encoders.retriever_encoder import RetrieverEncoder
    from topic_models.wikiweb2m import TopicCorpus


@dataclass
class TopicTrainConfig:
    num_topics: int = 100
    vocab_size: int = 2000  # for BoW/soft-label backends (ETM/CWTM/SoftLTM)
    min_word_freq: int = 5
    epochs: int = 20
    batch_size: int = 64
    lr: float = 2e-3
    centroid_source: str = "native"  # "native" | "empirical"
    centroid_normalize: bool = True
    n_empirical_docs: int = 20000  # sample size for empirical centroids
    # SoftLTM-specific (ignored by others):
    label_lm: str = "meta-llama/Llama-3.2-1B-Instruct"
    soft_label_tau: float = 3.0
    recon_lambda: float = 1e3
    device: str | None = None
    seed: int = 13


class TopicModel(abc.ABC):
    """Common contract consumed by MetadataBank.build (Phase 2) and pipeline (Phase 6)."""

    name: str = "base"

    def __init__(
        self,
        num_topics: int,
        encoder: "RetrieverEncoder",
        centroid_normalize: bool = True,
        device: str | None = None,
    ):
        self.num_topics = int(num_topics)
        self.encoder = encoder  # the retriever LM (shared space)
        self.embedding_dim = int(encoder.dim)
        self.centroid_normalize = bool(centroid_normalize)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._empirical_centroids: torch.Tensor | None = None
        # Requested centroid source; resolved deterministically via the property.
        self._requested_centroid_source: str = "native"

    # ---- abstract, per-backend ----
    @abc.abstractmethod
    def encode_topic_distribution(
        self, texts: list[str], batch_size: int = 8
    ) -> torch.Tensor:
        """(B,K) float32, rows on the simplex."""

    @abc.abstractmethod
    def _native_centroids(self) -> torch.Tensor | None:
        """(K,d) in retriever space, or None if the backend has no native
        embedding-space topics (e.g. CWTM). Do NOT normalize here."""

    @abc.abstractmethod
    def fit(self, corpus: "TopicCorpus", cfg: TopicTrainConfig) -> None:
        """Train on WikiWeb2M with K topics using self.encoder for grounding."""

    @property
    @abc.abstractmethod
    def supports_native_centroids(self) -> bool: ...

    # ---- construction hook used by the registry ----
    @classmethod
    def from_registry(
        cls,
        num_topics: int,
        encoder: "RetrieverEncoder",
        centroid_normalize: bool = True,
        device: str | None = None,
        **backend_kwargs,
    ) -> "TopicModel":
        """Default registry constructor: forwards to ``__init__``.

        Backends whose constructor differs (e.g. CEMTM builds its own backbone)
        override this to translate the uniform registry call into their own
        construction.
        """
        return cls(
            num_topics,
            encoder,
            centroid_normalize=centroid_normalize,
            device=device,
            **backend_kwargs,
        )

    # ---- shared, concrete ----
    @property
    def effective_centroid_source(self) -> str:
        """Deterministic resolution of which centroid source is actually used."""
        if not self.supports_native_centroids:
            return "empirical"
        if self._requested_centroid_source == "empirical":
            return "empirical"
        return "native"

    def set_centroid_source(self, source: str) -> None:
        """Record the requested centroid source (``"native"``/``"decoder"`` or
        ``"empirical"``). Stored so ``get_topic_centroids`` is deterministic."""
        if source in ("native", "decoder"):
            self._requested_centroid_source = "native"
        elif source == "empirical":
            self._requested_centroid_source = "empirical"
        else:
            raise ValueError(
                f"Unknown centroid_source {source!r} (expected native/decoder/empirical)."
            )

    @torch.no_grad()
    def get_topic_centroids(self) -> torch.Tensor:
        """Resolve centroids to ``(K, d)`` float32 (L2-normalized if configured)."""
        src = self.effective_centroid_source
        if src == "empirical":
            if self._empirical_centroids is None:
                raise RuntimeError(
                    "Empirical centroids requested but not fitted; call "
                    "fit_empirical_centroids(...) or maybe_fit_empirical(...) first."
                )
            T = self._empirical_centroids
        else:
            T = self._native_centroids()
            if T is None:
                raise RuntimeError(
                    f"Backend {self.name!r} exposes no native centroids; use "
                    "centroid_source='empirical'."
                )
        T = T.to(dtype=torch.float32)
        if self.centroid_normalize:
            T = F.normalize(T, p=2, dim=-1)
        return T

    @torch.no_grad()
    def fit_empirical_centroids(
        self, chunk_embeddings: torch.Tensor, theta: torch.Tensor
    ) -> torch.Tensor:
        """``t_k = (sum_c theta[c,k] e_c) / (sum_c theta[c,k] + eps)``; cache + return."""
        eps = 1e-8
        e = chunk_embeddings.to(device=self.device, dtype=torch.float32)  # (C, d)
        th = theta.to(device=self.device, dtype=torch.float32)  # (C, K)
        if e.shape[0] != th.shape[0]:
            raise ValueError(
                f"chunk_embeddings rows ({e.shape[0]}) != theta rows ({th.shape[0]})."
            )
        weighted_sum = th.t() @ e  # (K, d)
        mass = th.sum(dim=0).unsqueeze(-1)  # (K, 1)
        T_emp = weighted_sum / (mass + eps)  # (K, d)
        self._empirical_centroids = T_emp.contiguous()
        return self._empirical_centroids

    def topic_summary(self, theta: torch.Tensor, top_m: int) -> torch.Tensor:
        """Top-M masked weighted sum of ``get_topic_centroids()`` -> ``(B, d)``."""
        T = self.get_topic_centroids()  # (K, d)
        th = theta.to(device=T.device, dtype=torch.float32)  # (B, K)
        K = th.shape[-1]
        m = min(int(top_m), K)
        topk_idx = th.topk(m, dim=-1).indices  # (B, m)
        mask = torch.zeros_like(th)
        mask.scatter_(-1, topk_idx, 1.0)
        masked_theta = th * mask  # (B, K)
        return (masked_theta @ T).float()

    @torch.no_grad()
    def maybe_fit_empirical(self, corpus: "TopicCorpus", cfg: TopicTrainConfig) -> None:
        """Fit empirical centroids if requested or required (no native centroids)."""
        need = (cfg.centroid_source == "empirical") or (not self.supports_native_centroids)
        if not need:
            return
        docs = corpus.sample(cfg.n_empirical_docs, seed=cfg.seed)
        if not docs:
            return
        e = self.encoder.encode(docs, is_query=False).float()
        theta = self.encode_topic_distribution(docs)
        self.fit_empirical_centroids(e, theta)
        self._requested_centroid_source = "empirical"

    # ---- persistence ----
    def _save_state(self, out_dir: str) -> None:
        """Default: persist a torch ``state_dict`` if the backend is an ``nn.Module``."""
        if isinstance(self, nn.Module):
            torch.save(self.state_dict(), os.path.join(out_dir, "backend_state.pt"))

    def _load_state(self, in_dir: str) -> None:
        """Default: load a torch ``state_dict`` if present and backend is a module."""
        path = os.path.join(in_dir, "backend_state.pt")
        if isinstance(self, nn.Module) and os.path.exists(path):
            sd = torch.load(path, map_location="cpu", weights_only=False)
            self.load_state_dict(sd, strict=False)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        meta = {
            "name": self.name,
            "num_topics": self.num_topics,
            "embedding_dim": self.embedding_dim,
            "centroid_source_effective": self.effective_centroid_source,
            "centroid_normalize": self.centroid_normalize,
            "supports_native_centroids": self.supports_native_centroids,
        }
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self._save_state(out_dir)

        if self._empirical_centroids is not None:
            from safetensors.torch import save_file

            save_file(
                {"empirical_centroids": self._empirical_centroids.contiguous().cpu()},
                os.path.join(out_dir, "empirical_centroids.safetensors"),
            )

    @classmethod
    def load(cls, in_dir: str, encoder: "RetrieverEncoder") -> "TopicModel":
        with open(os.path.join(in_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        from topic_models.registry import build_topic_model

        model = build_topic_model(
            meta["name"],
            meta["num_topics"],
            encoder,
            centroid_normalize=bool(meta["centroid_normalize"]),
        )
        model._load_state(in_dir)

        emp_path = os.path.join(in_dir, "empirical_centroids.safetensors")
        if os.path.exists(emp_path):
            from safetensors.torch import load_file

            model._empirical_centroids = load_file(emp_path)["empirical_centroids"].float()

        model._requested_centroid_source = (
            "empirical"
            if meta["centroid_source_effective"] == "empirical"
            else "native"
        )
        return model
