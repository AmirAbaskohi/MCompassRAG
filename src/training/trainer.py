"""Trainer for CompassRetriever's three head modules.

Only the selector + abstraction + scorer are optimized. The frozen index buffers
(``meta_embeddings``, ``theta``, ``centroids``) and ``chunk_reps`` live on the
bank and are never optimized or checkpointed here.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from src.models.compass_retriever import CompassModelConfig, CompassRetriever
from src.training.dataset import collate_compass
from src.training.losses import DistillLoss


@dataclass
class TrainConfig:
    epochs: int = 5
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.05
    grad_clip: float = 1.0
    alpha: float = 0.5
    tau: float = 2.0
    tau_squared_scale: bool = True
    max_candidates_per_query: int = 16
    num_workers: int = 2
    val_fraction: float = 0.05
    eval_every_steps: int = 200
    log_every_steps: int = 50
    save_dir: str = "outputs/compass"
    seed: int = 13
    device: str | None = None


def _pick_device(device: str | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer_scheduler(model: CompassRetriever, cfg: TrainConfig, total_steps: int):
    """AdamW over ``model.trainable_parameters()`` only; linear warmup then decay."""
    trainable = list(model.trainable_parameters())

    # Assert no index buffer slipped into the trainable set.
    buffer_ptrs = set()
    for buf_name in ("m", "theta_bank", "centroids"):
        buf = getattr(model, buf_name, None)
        if buf is not None:
            buffer_ptrs.add(buf.data_ptr())
    for p in trainable:
        if p.data_ptr() in buffer_ptrs:
            raise AssertionError("An index buffer leaked into trainable parameters.")
        if not p.requires_grad:
            raise AssertionError("A non-trainable parameter is in the optimizer set.")

    optimizer = torch.optim.AdamW(
        trainable, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    scheduler = LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _forward_logits(model: CompassRetriever, bank, batch: dict) -> torch.Tensor:
    r_q, _ = model.build_query_rep(batch["e_q"])
    cand_r_c = bank.chunk_reps[batch["cand_rows"]]  # (B, P, 2d)
    return model.score(r_q, cand_r_c)  # (B, P)


@torch.no_grad()
def evaluate(model, loader, loss_fn, bank, device) -> dict:
    """Compute validation loss components and a frozen-encoder health signal."""
    model.eval()
    tot_loss = tot_bce = tot_kd = 0.0
    n_batches = 0
    n_queries = 0
    n_pos_gt_neg = 0

    for batch in loader:
        batch = _move_batch(batch, device)
        logits = _forward_logits(model, bank, batch)
        loss, parts = loss_fn(
            logits, batch["y"], batch["z_t"], batch["cand_mask"], batch["kd_mask"]
        )
        tot_loss += parts["loss"]
        tot_bce += parts["bce"]
        tot_kd += parts["kd"]
        n_batches += 1

        y = batch["y"]
        cm = batch["cand_mask"]
        pos_mask = (y >= 0.5) * cm
        neg_mask = (y < 0.5) * cm
        B = logits.shape[0]
        for b in range(B):
            if pos_mask[b].sum() < 1 or neg_mask[b].sum() < 1:
                continue
            pos_logits = logits[b][pos_mask[b].bool()]
            neg_logits = logits[b][neg_mask[b].bool()]
            n_queries += 1
            if pos_logits.max() > neg_logits.mean():
                n_pos_gt_neg += 1

    denom = max(1, n_batches)
    acc = (n_pos_gt_neg / n_queries) if n_queries > 0 else 0.0
    model.train()
    return {
        "val_loss": tot_loss / denom,
        "val_bce": tot_bce / denom,
        "val_kd": tot_kd / denom,
        "pos_gt_neg_acc": acc,
    }


class CompassTrainer:
    def __init__(self, model: CompassRetriever, bank, cfg: TrainConfig):
        self.cfg = cfg
        self.device = _pick_device(cfg.device)
        self.model = model.to(self.device)

        # Attach the frozen index on-device.
        self.model.set_index(
            bank.meta_embeddings.to(self.device),
            bank.theta.to(self.device),
            bank.centroids.to(self.device),
        )
        bank.chunk_reps = bank.chunk_reps.to(self.device).float()
        self.bank = bank

        self.loss_fn = DistillLoss(
            alpha=cfg.alpha, tau=cfg.tau, tau_squared_scale=cfg.tau_squared_scale
        )
        self.optimizer = None
        self.scheduler = None
        os.makedirs(cfg.save_dir, exist_ok=True)

    def train(self, train_ds, val_ds) -> None:
        cfg = self.cfg
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_compass,
            num_workers=cfg.num_workers,
            drop_last=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_compass,
            num_workers=cfg.num_workers,
            drop_last=False,
        )

        total_steps = max(1, len(train_loader) * cfg.epochs)
        optimizer, scheduler = build_optimizer_scheduler(self.model, cfg, total_steps)
        self.optimizer = optimizer
        self.scheduler = scheduler
        trainable = list(self.model.trainable_parameters())

        step = 0
        best_val = math.inf
        first_loss = None
        last_loss = None

        self.model.train()
        for epoch in range(cfg.epochs):
            for batch in train_loader:
                batch = _move_batch(batch, self.device)
                logits = _forward_logits(self.model, self.bank, batch)
                loss, parts = self.loss_fn(
                    logits,
                    batch["y"],
                    batch["z_t"],
                    batch["cand_mask"],
                    batch["kd_mask"],
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
                optimizer.step()
                scheduler.step()

                if first_loss is None:
                    first_loss = parts["loss"]
                last_loss = parts["loss"]
                step += 1

                if step % cfg.log_every_steps == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"[train] epoch={epoch} step={step} loss={parts['loss']:.4f} "
                        f"bce={parts['bce']:.4f} kd={parts['kd']:.4f} lr={lr_now:.2e}"
                    )

                if cfg.eval_every_steps > 0 and step % cfg.eval_every_steps == 0:
                    val = evaluate(
                        self.model, val_loader, self.loss_fn, self.bank, self.device
                    )
                    print(f"[eval] step={step} {val}")
                    if val["val_loss"] < best_val:
                        best_val = val["val_loss"]
                        self.save_checkpoint(
                            os.path.join(cfg.save_dir, "best.pt"), step, best=True
                        )

            # End-of-epoch eval + checkpoints.
            val = evaluate(self.model, val_loader, self.loss_fn, self.bank, self.device)
            print(f"[eval@epoch{epoch}] {val}")
            self.save_checkpoint(os.path.join(cfg.save_dir, "last.pt"), step)
            if val["val_loss"] < best_val:
                best_val = val["val_loss"]
                self.save_checkpoint(
                    os.path.join(cfg.save_dir, "best.pt"), step, best=True
                )

        # Guarantee best.pt exists even if validation never improved past inf.
        best_path = os.path.join(cfg.save_dir, "best.pt")
        if not os.path.exists(best_path):
            self.save_checkpoint(best_path, step, best=True)

        print(
            f"[train] done. first_loss={first_loss:.4f} last_loss={last_loss:.4f} "
            f"best_val={best_val:.4f}"
        )
        self._first_loss = first_loss
        self._last_loss = last_loss

    def save_checkpoint(self, path: str, step: int, best: bool = False) -> None:
        ckpt = {
            "model": {
                "selector": self.model.selector.state_dict(),
                "abstraction": self.model.abstraction.state_dict(),
                "scorer": self.model.scorer.state_dict(),
            },
            "model_cfg": asdict(self.model.cfg),
            "optimizer": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "step": step,
            "best": best,
            "train_cfg": asdict(self.cfg),
        }
        torch.save(ckpt, path)

    @staticmethod
    def load_model(path: str) -> CompassRetriever:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg_dict = dict(ckpt["model_cfg"])
        # scorer_hidden may have been serialized as a list; restore tuple.
        if "scorer_hidden" in cfg_dict and isinstance(cfg_dict["scorer_hidden"], list):
            cfg_dict["scorer_hidden"] = tuple(cfg_dict["scorer_hidden"])
        cfg = CompassModelConfig(**cfg_dict)
        model = CompassRetriever(cfg)
        model.selector.load_state_dict(ckpt["model"]["selector"], strict=True)
        model.abstraction.load_state_dict(ckpt["model"]["abstraction"], strict=True)
        model.scorer.load_state_dict(ckpt["model"]["scorer"], strict=True)
        model.eval()
        return model
