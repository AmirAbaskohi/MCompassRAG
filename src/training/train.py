"""CLI entry point for training CompassRetriever (config-driven).

Example::

    python -m src.training.train --config configs/train_retriever.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import random

from src.config import bind, build_encoder, load_yaml, section
from src.index.metadata_bank import MetadataBank
from src.models.compass_retriever import CompassModelConfig, CompassRetriever
from src.training.dataset import (
    CompassTrainDataset,
    precompute_query_embeddings,
)
from src.training.trainer import CompassTrainer, TrainConfig


def _read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train CompassRetriever from a YAML config.")
    p.add_argument("--config", required=True, help="Path to a train_retriever YAML.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_yaml(args.config)

    bank_dir = cfg.get("bank_dir")
    train_jsonl = cfg.get("train_jsonl")
    if not bank_dir or not train_jsonl:
        raise ValueError("config is missing bank_dir or train_jsonl")

    out_dir = section(cfg, "output", "save_dir")
    if not out_dir:
        raise ValueError("config is missing output.save_dir")
    os.makedirs(out_dir, exist_ok=True)

    bank = MetadataBank.load(bank_dir)

    encoder = build_encoder(cfg)
    if encoder.dim != bank.d:
        raise ValueError(
            f"encoder.dim {encoder.dim} != bank.d {bank.d}; the encoder must share "
            "the bank's embedding space."
        )

    # Model config: d/K come from the bank; the rest from the `model` section.
    model_sec = dict(section(cfg, "model", default={}) or {})
    model_sec["d"] = bank.d
    model_sec["K"] = bank.K
    model_cfg = bind(CompassModelConfig, model_sec)
    if model_cfg.top_m != bank.cfg.top_m:
        raise ValueError(
            f"model.top_m {model_cfg.top_m} != index top_m {bank.cfg.top_m}; they "
            "must match."
        )

    # Train config: save_dir comes from the `output` section.
    train_sec = dict(section(cfg, "train", default={}) or {})
    train_sec["save_dir"] = out_dir
    tcfg = bind(TrainConfig, train_sec)

    records = _read_jsonl(train_jsonl)
    query_emb = precompute_query_embeddings(
        records,
        encoder,
        batch_size=16,
        cache_path=os.path.join(out_dir, "qemb"),
    )

    # Split train/val by record (seeded).
    rng = random.Random(tcfg.seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(tcfg.val_fraction * len(shuffled))) if len(shuffled) > 1 else 0
    val_records = shuffled[:n_val]
    train_records = shuffled[n_val:]
    if not train_records:
        train_records = shuffled
        val_records = shuffled

    train_path = os.path.join(out_dir, "train_split.jsonl")
    val_path = os.path.join(out_dir, "val_split.jsonl")
    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)

    model = CompassRetriever(model_cfg)

    train_ds = CompassTrainDataset(
        train_path, bank, query_emb, tcfg.max_candidates_per_query, seed=tcfg.seed
    )
    val_ds = CompassTrainDataset(
        val_path, bank, query_emb, tcfg.max_candidates_per_query, seed=tcfg.seed
    )

    trainer = CompassTrainer(model, bank, tcfg)
    trainer.train(train_ds, val_ds)
    print(f"[train] saved retriever to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
