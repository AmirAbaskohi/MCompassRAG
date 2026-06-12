"""CEMTM topic adapter — now a :class:`TopicModel` backend.

Composes the trainable CEMTM topic head (``topic_encoder``, ``importance_net``,
``decoder``) on top of :class:`~src.encoders.backbone.Qwen3TextBackbone` so a
CEMTM checkpoint loads cleanly, then exposes deterministic (``eps=0``) per-document
topic distributions and topic centroids (decoder columns, or empirical).

The θ math replicates ``CEMTM.forward`` exactly (text-only, batched, importance
mean at inference). Centroid/normalize behavior is identical to Phase 1; the
shared centroid/summary ops now come from :class:`TopicModel`.
"""

from __future__ import annotations

import random
import sys
import warnings
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.backbone import Qwen3TextBackbone
from src.encoders.retriever_encoder import RetrieverEncoder
from topic_models.base import TopicModel, TopicTrainConfig
from topic_models.registry import register_topic_model


@dataclass
class CEMTMConfig:
    num_topics: int = 100
    cemtm_repo_path: str = "third_party/CEMTM"
    checkpoint_path: str | None = None  # state dict saved as {"model": state_dict, ...}
    backbone_name: str = "Qwen/Qwen3-Embedding-4B"
    device: str | None = None
    dtype: torch.dtype = torch.bfloat16
    centroid_source: str = "decoder"  # "decoder" | "native" | "empirical"
    centroid_normalize: bool = True  # L2-normalize centroids to unit norm


_HEAD_PREFIXES = ("topic_encoder.", "importance_net.", "decoder.")


@register_topic_model("cemtm")
class CEMTMTopicProvider(nn.Module, TopicModel):
    """Frozen CEMTM topic head over a text backbone, conforming to TopicModel."""

    def __init__(self, cfg: CEMTMConfig, encoder: RetrieverEncoder | None = None):
        nn.Module.__init__(self)
        self.cfg = cfg

        # Import CEMTM's topic-head modules from the cloned repo.
        if cfg.cemtm_repo_path not in sys.path:
            sys.path.insert(0, cfg.cemtm_repo_path)
        from model.encoder import TopicEncoder
        from model.importance_net import ImportanceNetwork

        # Reuse the retriever's backbone when one is supplied (shared 4B model,
        # no second download); otherwise build our own.
        if encoder is not None and hasattr(encoder, "backbone"):
            self.backbone = encoder.backbone
        else:
            self.backbone = Qwen3TextBackbone(
                cfg.backbone_name, device=cfg.device, dtype=cfg.dtype
            )

        D = self.backbone.hidden_dim
        K = cfg.num_topics
        self.input_dim = D
        self.device_ = self.backbone.device_

        # Module names MUST be topic_encoder / importance_net / decoder so that
        # CEMTM checkpoint keys align directly under load_state_dict.
        self.topic_encoder = TopicEncoder(input_dim=D, num_topics=K)
        self.importance_net = ImportanceNetwork(
            input_dim=D, hidden_dim=2 * D, num_layers=2, num_heads=8, dropout=0.1
        )
        self.decoder = nn.Linear(K, D)

        # Topic head stays float32 for numerical stability (backbone is bfloat16).
        self.topic_encoder.to(device=self.device_, dtype=torch.float32)
        self.importance_net.to(device=self.device_, dtype=torch.float32)
        self.decoder.to(device=self.device_, dtype=torch.float32)

        # Initialize the TopicModel interface (sets encoder/embedding_dim/etc.).
        iface_encoder = (
            encoder if encoder is not None else RetrieverEncoder.from_backbone(self.backbone)
        )
        TopicModel.__init__(
            self,
            num_topics=K,
            encoder=iface_encoder,
            centroid_normalize=cfg.centroid_normalize,
            device=str(self.device_),
        )
        self.set_centroid_source(cfg.centroid_source)

        if cfg.checkpoint_path is not None:
            self.load_checkpoint(cfg.checkpoint_path)
        else:
            warnings.warn(
                "CEMTMTopicProvider initialized WITHOUT a checkpoint: the topic "
                "head is randomly initialized, so theta distributions and topic "
                "centroids are UNTRAINED and not meaningful. Provide "
                "CEMTMConfig.checkpoint_path or call fit(...) for real topics.",
                stacklevel=2,
            )

        # Frozen by default; fit() temporarily unfreezes if it trains.
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    # ---- registry construction ----
    @classmethod
    def from_registry(
        cls,
        num_topics: int,
        encoder: RetrieverEncoder | None,
        centroid_normalize: bool = True,
        device: str | None = None,
        **backend_kwargs,
    ) -> "CEMTMTopicProvider":
        valid = {f.name for f in fields(CEMTMConfig)}
        cfg_kwargs = {k: v for k, v in backend_kwargs.items() if k in valid}
        cfg = CEMTMConfig(
            num_topics=num_topics,
            centroid_normalize=centroid_normalize,
            device=device,
            **cfg_kwargs,
        )
        return cls(cfg, encoder=encoder)

    # ---- interface: abstract methods ----
    @property
    def supports_native_centroids(self) -> bool:
        return True

    def _native_centroids(self) -> torch.Tensor | None:
        # Decoder columns: embedding reconstructed for a one-hot topic.
        return self.decoder.weight.t().contiguous()  # (K, D)

    @torch.no_grad()
    def encode_topic_distribution(
        self, texts: list[str], batch_size: int = 8
    ) -> torch.Tensor:
        """Document-topic distribution ``theta`` ``(B, K)`` (deterministic, eps=0)."""
        h_list, _ = self.backbone.encode_tokens(texts, batch_size=batch_size)

        rows: list[torch.Tensor] = []
        for H in h_list:
            H = H.to(device=self.device_, dtype=torch.float32)  # (N, D)
            Hb = H.unsqueeze(0)  # (1, N, D)
            tv = F.softmax(self.topic_encoder.proj(Hb), dim=-1)  # (1, N, K)
            mu, _ = self.importance_net(Hb)  # (1, N)
            beta = F.softmax(mu, dim=-1)  # (1, N)
            topic_d = torch.sum(beta.unsqueeze(-1) * tv, dim=1)  # (1, K)
            topic_d = F.softmax(topic_d, dim=-1)
            rows.append(topic_d.squeeze(0))

        if not rows:
            return torch.empty(0, self.num_topics, device=self.device_, dtype=torch.float32)
        return torch.stack(rows, dim=0).float()

    def fit(self, corpus, cfg: TopicTrainConfig) -> None:
        """Load a checkpoint if given, else train CEMTM's reconstruction objective.

        With no checkpoint, runs the real CEMTM objective text-only on the corpus:
        per document compute ``topic_d`` (with reparameterized importance sampling),
        reconstruct the doc embedding ``e_d' = decoder(topic_d)``, and minimize
        ``MSE(e_d', e_d) + KL`` on the importance posterior. Trains the topic head
        only; the backbone stays frozen.
        """
        if self.cfg.checkpoint_path is not None:
            self.load_checkpoint(self.cfg.checkpoint_path)
            self.set_centroid_source(cfg.centroid_source)
            return

        docs = list(corpus.documents)
        if not docs:
            self.set_centroid_source(cfg.centroid_source)
            return

        head = (
            list(self.topic_encoder.parameters())
            + list(self.importance_net.parameters())
            + list(self.decoder.parameters())
        )
        for p in head:
            p.requires_grad_(True)
        self.topic_encoder.train()
        self.importance_net.train()
        self.decoder.train()

        opt = torch.optim.Adam(head, lr=cfg.lr)
        rng = random.Random(cfg.seed)

        for _epoch in range(cfg.epochs):
            order = list(range(len(docs)))
            rng.shuffle(order)
            for start in range(0, len(order), cfg.batch_size):
                idxs = order[start : start + cfg.batch_size]
                batch = [docs[i] for i in idxs]
                h_list, e_d = self.backbone.encode_tokens(batch)  # e_d (b, D)
                losses = []
                for bi, H in enumerate(h_list):
                    Hb = H.to(self.device_, torch.float32).unsqueeze(0)
                    tv = F.softmax(self.topic_encoder.proj(Hb), dim=-1)
                    mu, logvar = self.importance_net(Hb)
                    std = torch.exp(0.5 * logvar)
                    alpha = mu + torch.randn_like(std) * std
                    beta = F.softmax(alpha, dim=-1)
                    topic_d = F.softmax((beta.unsqueeze(-1) * tv).sum(dim=1), dim=-1)
                    e_d_prime = self.decoder(topic_d)  # (1, D)
                    target = e_d[bi : bi + 1].to(self.device_, torch.float32)
                    recon = F.mse_loss(e_d_prime, target)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    losses.append(recon + 1e-3 * kl)
                loss = torch.stack(losses).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.set_centroid_source(cfg.centroid_source)

    # ---- persistence ----
    def _save_state(self, out_dir: str) -> None:
        import os

        torch.save(
            {
                "topic_encoder": self.topic_encoder.state_dict(),
                "importance_net": self.importance_net.state_dict(),
                "decoder": self.decoder.state_dict(),
            },
            os.path.join(out_dir, "cemtm_head.pt"),
        )

    def _load_state(self, in_dir: str) -> None:
        import os

        path = os.path.join(in_dir, "cemtm_head.pt")
        if not os.path.exists(path):
            return
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.topic_encoder.load_state_dict(sd["topic_encoder"], strict=True)
        self.importance_net.load_state_dict(sd["importance_net"], strict=True)
        self.decoder.load_state_dict(sd["decoder"], strict=True)
        self.topic_encoder.to(self.device_, torch.float32)
        self.importance_net.to(self.device_, torch.float32)
        self.decoder.to(self.device_, torch.float32)

    def load_checkpoint(self, path: str) -> None:
        """Load CEMTM topic-head weights from ``state['model']`` (Phase 1 logic)."""
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            model_sd = state["model"]
        else:
            model_sd = state

        remapped: dict[str, torch.Tensor] = {}
        for k, v in model_sd.items():
            key = k[len("module.") :] if k.startswith("module.") else k
            if key.startswith(_HEAD_PREFIXES):
                remapped[key] = v

        target_keys = {k for k in self.state_dict() if k.startswith(_HEAD_PREFIXES)}
        missing_in_ckpt = sorted(target_keys - set(remapped))
        if missing_in_ckpt:
            raise KeyError(
                f"Checkpoint is missing required CEMTM topic-head keys: {missing_in_ckpt}"
            )

        D, K = self.input_dim, self.num_topics
        dec_w = remapped["decoder.weight"]
        ckpt_D = dec_w.shape[0]
        ckpt_K = dec_w.shape[1]
        if ckpt_D != D:
            raise ValueError(
                f"Checkpoint backbone/hidden dim ({ckpt_D}) != retriever dim ({D}). "
                "The CEMTM checkpoint was trained in a different embedding space, so "
                "its decoder-column centroids are NOT in the retriever space. Use "
                "centroid_source='empirical' with fit_empirical_centroids instead, "
                "or load a checkpoint trained with the matching backbone."
            )
        if ckpt_K != K:
            raise ValueError(
                f"Checkpoint num_topics ({ckpt_K}) != configured num_topics ({K})."
            )

        result = self.load_state_dict(remapped, strict=False)
        still_missing = [k for k in result.missing_keys if k.startswith(_HEAD_PREFIXES)]
        if still_missing:
            raise RuntimeError(f"Failed to load CEMTM topic-head keys: {still_missing}")

        assert tuple(self.decoder.weight.shape) == (D, K)
        assert tuple(self.topic_encoder.proj.weight.shape) == (K, D)
