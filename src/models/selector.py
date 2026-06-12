"""Selector module.

Scores every bank entry against the query with a single affine head over the
concatenation ``[e_q ; m_i]``, then selects the top-L entries. The selected,
renormalized logits ``s_tilde`` are the single intended differentiable path into
the selector (see :class:`~src.models.abstraction.Abstraction`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Selector(nn.Module):
    """``a_i = w_s^T [e_q ; m_i] + b_s`` over all bank entries."""

    def __init__(self, d: int):
        super().__init__()
        self.d = int(d)
        # Single affine head over the 2d-dim concatenation [e_q ; m_i].
        self.linear = nn.Linear(2 * d, 1, bias=True)

    def forward(self, e_q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Compute selection logits.

        Args:
            e_q: ``(B, d)`` query embeddings.
            m:   ``(N, d)`` bank selector inputs (``sum_k theta_k t_k``, all K).

        Returns:
            ``(B, N)`` logits.

        Uses the efficient split form: with ``w = [w_q ; w_m]``,
        ``a_ij = e_q_i . w_q + m_j . w_m + b`` which avoids materializing the
        ``(B, N, 2d)`` concatenation. Numerically identical to
        :meth:`_reference_forward`.
        """
        d = self.d
        w = self.linear.weight.view(-1)  # (2d,)
        w_q = w[:d]  # (d,)
        w_m = w[d:]  # (d,)
        bias = self.linear.bias  # (1,)

        a_q = e_q @ w_q  # (B,)
        a_m = m @ w_m  # (N,)
        a = a_q[:, None] + a_m[None, :] + bias  # (B, N)
        return a

    def _reference_forward(self, e_q: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Naive reference: explicit concat then ``Linear``. Used by the smoke test."""
        B = e_q.shape[0]
        N = m.shape[0]
        eq = e_q[:, None, :].expand(B, N, self.d)
        mm = m[None, :, :].expand(B, N, self.d)
        pair = torch.cat([eq, mm], dim=-1)  # (B, N, 2d)
        return self.linear(pair).squeeze(-1)  # (B, N)

    @staticmethod
    def select_topl(a: torch.Tensor, top_l: int):
        """Select the top-L bank entries per query.

        Args:
            a: ``(B, N)`` selection logits.
            top_l: number of entries to keep (clamped to ``N``).

        Returns:
            ``(idx, s_full, s_tilde)`` where

            * ``idx`` ``(B, L)``: indices of the top-L entries,
            * ``s_full`` ``(B, N)``: softmax over all N entries,
            * ``s_tilde`` ``(B, L)``: softmax of the *selected* logits,
              renormalized over the L chosen entries (carries gradient into the
              selector).
        """
        N = a.shape[-1]
        L = min(int(top_l), N)
        s_full = torch.softmax(a, dim=-1)  # (B, N)
        topv, idx = a.topk(L, dim=-1)  # (B, L), (B, L)
        s_tilde = torch.softmax(topv, dim=-1)  # (B, L)
        return idx, s_full, s_tilde
