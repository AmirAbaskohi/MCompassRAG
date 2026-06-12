"""Scorer module: a 3-layer MLP over ``[r_q ; r_c] in R^{4d}`` -> scalar logit."""

from __future__ import annotations

import torch
import torch.nn as nn


class Scorer(nn.Module):
    """``z(q, c) = MLP_phi([r_q ; r_c])`` with input ``R^{4d}``."""

    def __init__(
        self, d: int, hidden: tuple[int, int] = (1024, 256), dropout: float = 0.1
    ):
        super().__init__()
        self.d = int(d)
        h1, h2 = hidden
        self.mlp = nn.Sequential(
            nn.Linear(4 * d, h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, pair_rep: torch.Tensor) -> torch.Tensor:
        """``(..., 4d) -> (...)``; squeezes the trailing size-1 dim."""
        return self.mlp(pair_rep).squeeze(-1)
