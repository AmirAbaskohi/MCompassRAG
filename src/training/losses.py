"""Distillation loss: ``L = (1-alpha)*BCE + alpha*KD``.

``KD`` is a per-candidate Bernoulli KL ``KL(sigma(z_t/tau) || sigma(z/tau))``,
computed in a numerically stable way directly from logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.5, tau: float = 2.0, tau_squared_scale: bool = True
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.tau_squared_scale = bool(tau_squared_scale)

    def forward(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        z_t: torch.Tensor,
        cand_mask: torch.Tensor,
        kd_mask: torch.Tensor,
    ):
        """All inputs ``(B, P)``. Returns ``(loss_scalar, parts_dict)``."""
        logits = logits.float()
        y = y.float()
        z_t = z_t.float()
        cand_mask = cand_mask.float()
        kd_mask = kd_mask.float()

        # --- BCE over valid candidates ---
        bce_el = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        bce = (bce_el * cand_mask).sum() / cand_mask.sum().clamp_min(1.0)

        # --- KD: Bernoulli KL over kd-valid candidates (stable via logits) ---
        ls = logits / self.tau
        lt = z_t / self.tau
        log_ps = F.logsigmoid(ls)
        log_1mps = F.logsigmoid(-ls)
        log_pt = F.logsigmoid(lt)
        log_1mpt = F.logsigmoid(-lt)
        pt = log_pt.exp()
        one_mpt = log_1mpt.exp()
        kl_el = pt * (log_pt - log_ps) + one_mpt * (log_1mpt - log_1mps)
        if self.tau_squared_scale:
            kl_el = kl_el * (self.tau * self.tau)
        kd = (kl_el * kd_mask).sum() / kd_mask.sum().clamp_min(1.0)

        loss = (1.0 - self.alpha) * bce + self.alpha * kd

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: bce={bce}, kd={kd}")

        parts = {
            "loss": float(loss.detach()),
            "bce": float(bce.detach()),
            "kd": float(kd.detach()),
        }
        return loss, parts
