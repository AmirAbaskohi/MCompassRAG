"""``CompassRetriever``: wires Selector + Abstraction + Scorer over a frozen index.

The retriever encoder ``f_psi`` and the metadata bank are frozen and live outside
this module. Here, the topic centroids ``T``, the bank topic distributions
``theta_bank``, and the selector input ``m`` are registered as **buffers** (never
parameters, never requiring grad). Only the Selector, Abstraction, and Scorer are
trainable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.abstraction import Abstraction
from src.models.scorer import Scorer
from src.models.selector import Selector


@dataclass
class CompassModelConfig:
    d: int
    K: int
    top_l: int = 16
    top_m: int = 8  # must match MetadataBank IndexConfig.top_m
    abstraction_layers: int = 2
    abstraction_heads: int = 4
    abstraction_ff: int = 256
    abstraction_dropout: float = 0.1
    abstraction_softmax: bool = True
    pool: str = "selection_weighted"  # "selection_weighted" | "mean"
    scorer_hidden: tuple[int, int] = (1024, 256)
    scorer_dropout: float = 0.1


def weighted_topic_summary(
    theta: torch.Tensor, T: torch.Tensor, top_m: int
) -> torch.Tensor:
    """Top-M masked weighted sum of centroids.

    Semantics are IDENTICAL to Phase 1's ``CEMTMTopicProvider.topic_summary``:
    keep the top-M topics by mass per row (zero the rest), then
    ``g = masked_theta @ T``.

    Args:
        theta: ``(B, K)`` topic distributions.
        T: ``(K, d)`` topic centroids.
        top_m: number of topics to keep per row (clamped to K).

    Returns:
        ``(B, d)`` float32.
    """
    th = theta.float()
    Tf = T.float()
    K = th.shape[-1]
    m = min(int(top_m), K)
    topk_idx = th.topk(m, dim=-1).indices  # (B, m)
    mask = torch.zeros_like(th)
    mask.scatter_(-1, topk_idx, 1.0)
    masked_theta = th * mask  # (B, K)
    g = masked_theta @ Tf  # (B, d)
    return g.float()


class CompassRetriever(nn.Module):
    def __init__(self, cfg: CompassModelConfig):
        super().__init__()
        self.cfg = cfg
        self.d = int(cfg.d)
        self.K = int(cfg.K)

        self.selector = Selector(cfg.d)
        self.abstraction = Abstraction(
            num_topics=cfg.K,
            n_layers=cfg.abstraction_layers,
            n_heads=cfg.abstraction_heads,
            dim_ff=cfg.abstraction_ff,
            dropout=cfg.abstraction_dropout,
            pool=cfg.pool,
            apply_softmax=cfg.abstraction_softmax,
        )
        self.scorer = Scorer(
            cfg.d, hidden=cfg.scorer_hidden, dropout=cfg.scorer_dropout
        )

        # Index buffers are registered lazily via set_index (declared as None now
        # so they are tracked as buffers and moved by .to()/.cuda()).
        self.register_buffer("m", None)
        self.register_buffer("theta_bank", None)
        self.register_buffer("centroids", None)

    def set_index(
        self,
        meta_embeddings: torch.Tensor,
        theta_bank: torch.Tensor,
        centroids: torch.Tensor,
    ) -> None:
        """Register the frozen index buffers; assert dims match cfg."""
        if meta_embeddings.shape[1] != self.d:
            raise ValueError(
                f"meta_embeddings dim {meta_embeddings.shape[1]} != cfg.d {self.d}"
            )
        if theta_bank.shape[1] != self.K:
            raise ValueError(
                f"theta_bank dim {theta_bank.shape[1]} != cfg.K {self.K}"
            )
        if tuple(centroids.shape) != (self.K, self.d):
            raise ValueError(
                f"centroids shape {tuple(centroids.shape)} != ({self.K}, {self.d})"
            )
        if meta_embeddings.shape[0] != theta_bank.shape[0]:
            raise ValueError(
                f"N mismatch: m has {meta_embeddings.shape[0]} rows, theta_bank "
                f"has {theta_bank.shape[0]}"
            )

        # Detach + float32; buffers never require grad.
        self.m = meta_embeddings.detach().float()
        self.theta_bank = theta_bank.detach().float()
        self.centroids = centroids.detach().float()
        self.m.requires_grad_(False)
        self.theta_bank.requires_grad_(False)
        self.centroids.requires_grad_(False)

    def _require_index(self) -> None:
        if self.m is None or self.theta_bank is None or self.centroids is None:
            raise RuntimeError("Index not set; call set_index(...) first.")

    def build_query_rep(self, e_q: torch.Tensor):
        """Build the query representation ``r_q = [e_q ; g_q]``.

        Args:
            e_q: ``(B, d)`` L2-normalized query embeddings.

        Returns:
            ``(r_q, aux)`` where ``r_q`` is ``(B, 2d)`` and ``aux`` carries
            ``idx``, ``s_full``, ``s_tilde``, ``theta_hat_q``.
        """
        self._require_index()
        e_q = e_q.float()

        a = self.selector(e_q, self.m)  # (B, N)
        idx, s_full, s_tilde = Selector.select_topl(a, self.cfg.top_l)  # (B,L)...

        H0 = self.theta_bank[idx]  # (B, L, K)
        theta_hat_q = self.abstraction(H0, s_tilde)  # (B, K)

        g_q = weighted_topic_summary(theta_hat_q, self.centroids, self.cfg.top_m)  # (B,d)
        r_q = torch.cat([e_q, g_q], dim=-1)  # (B, 2d)

        # The first d dims of r_q must be exactly e_q.
        assert torch.equal(r_q[:, : self.d], e_q), "r_q[:, :d] must equal e_q exactly"

        aux = dict(idx=idx, s_full=s_full, s_tilde=s_tilde, theta_hat_q=theta_hat_q)
        return r_q, aux

    def score(self, r_q: torch.Tensor, cand_r_c: torch.Tensor) -> torch.Tensor:
        """Score candidate chunk reps.

        Args:
            r_q: ``(B, 2d)``.
            cand_r_c: ``(B, P, 2d)``.

        Returns:
            ``(B, P)`` logits.
        """
        B, P, _ = cand_r_c.shape
        r_q_exp = r_q[:, None, :].expand(B, P, 2 * self.d)  # (B, P, 2d)
        pair = torch.cat([r_q_exp, cand_r_c], dim=-1)  # (B, P, 4d)
        return self.scorer(pair)  # (B, P)

    def score_all(
        self, r_q: torch.Tensor, chunk_reps: torch.Tensor, block_size: int = 4096
    ) -> torch.Tensor:
        """Score every chunk in the bank, blocked over N to bound memory.

        Args:
            r_q: ``(B, 2d)``.
            chunk_reps: ``(N, 2d)``.
            block_size: number of chunks scored per block.

        Returns:
            ``(B, N)`` logits.
        """
        B = r_q.shape[0]
        N = chunk_reps.shape[0]
        out = r_q.new_empty((B, N))
        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            block = chunk_reps[start:end]  # (P, 2d)
            P = block.shape[0]
            cand = block[None, :, :].expand(B, P, 2 * self.d)  # (B, P, 2d)
            out[:, start:end] = self.score(r_q, cand)
        return out

    def forward(self, e_q: torch.Tensor, cand_r_c: torch.Tensor) -> torch.Tensor:
        """``build_query_rep`` then ``score`` -> ``(B, P)`` logits."""
        r_q, _ = self.build_query_rep(e_q)
        return self.score(r_q, cand_r_c)

    def trainable_parameters(self):
        """Yield ONLY selector + abstraction + scorer parameters."""
        yield from self.selector.parameters()
        yield from self.abstraction.parameters()
        yield from self.scorer.parameters()
