"""Abstraction module.

Refines a query topic distribution by running a small Transformer encoder over
the topic vectors of the selected bank entries and pooling across them.

Pooling note (underspecified in the paper): hard top-L selection is
non-differentiable, so a plain mean-pool gives the selector no gradient. The
default ``selection_weighted`` pooling weights the transformer outputs by the
renormalized selected logits ``s_tilde``, which is the single intended gradient
path into the selector. Uniform ``s_tilde`` reduces this exactly to mean-pool;
``pool="mean"`` reproduces the paper text literally but only trains the selector
if it is otherwise supervised.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Abstraction(nn.Module):
    """2-layer Transformer encoder over K-dim topic tokens, pooled to ``theta_hat_q``."""

    def __init__(
        self,
        num_topics: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_ff: int = 256,
        dropout: float = 0.1,
        pool: str = "selection_weighted",
        apply_softmax: bool = True,
    ):
        super().__init__()
        if num_topics % n_heads != 0:
            raise ValueError(
                f"num_topics ({num_topics}) must be divisible by n_heads ({n_heads}) "
                "for nn.TransformerEncoderLayer (d_model must split across heads)."
            )
        if pool not in ("selection_weighted", "mean"):
            raise ValueError(
                f"Unknown pool {pool!r} (expected 'selection_weighted' or 'mean')."
            )

        self.num_topics = int(num_topics)
        self.pool = pool
        self.apply_softmax = bool(apply_softmax)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_topics,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self, selected_theta: torch.Tensor, s_tilde: torch.Tensor
    ) -> torch.Tensor:
        """Refine the query topic distribution.

        Args:
            selected_theta: ``(B, L, K)`` topic vectors of the selected entries.
            s_tilde: ``(B, L)`` renormalized selected logits (pooling weights).

        Returns:
            ``(B, K)`` refined query topic distribution ``theta_hat_q`` (float32).
        """
        x = selected_theta.float()
        out = self.transformer(x)  # (B, L, K)

        if self.pool == "mean":
            theta_hat = out.mean(dim=1)  # (B, K)
        else:  # selection_weighted
            w = s_tilde.float().unsqueeze(-1)  # (B, L, 1)
            theta_hat = (w * out).sum(dim=1)  # (B, K)

        if self.apply_softmax:
            theta_hat = F.softmax(theta_hat, dim=-1)
        return theta_hat.float()
