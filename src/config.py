"""YAML config loading + dataclass binding for the CompassRAG CLIs.

Keeps the entrypoints declarative: a YAML file maps onto the existing dataclasses
(`CEMTMConfig`, `TopicTrainConfig`, `IndexConfig`, `CompassModelConfig`,
`TrainConfig`, `DataGenConfig`, `CompassRAGConfig`). Unknown keys are ignored and
missing keys keep dataclass defaults.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import yaml

_DTYPE_NAMES = {"bfloat16", "float16", "float32", "float64", "bf16", "fp16", "fp32"}


def load_yaml(path: str) -> dict:
    """Parse a YAML file into a dict; raise if the file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def to_dtype(name: str):
    """Map a dtype string (e.g. ``'bfloat16'``) to a ``torch.dtype``."""
    import torch

    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unknown dtype {name!r}; choose from {sorted(table)}.")
    return table[key]


def _field_default(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:  # type: ignore[comparison-overlap]
        try:
            return f.default_factory()  # type: ignore[misc]
        except Exception:
            return None
    return None


def bind(dc_type, d: dict):
    """Construct ``dc_type`` from ``d``, ignoring unknown keys, keeping defaults.

    Coerces ``dtype`` string fields to ``torch.dtype`` and coerces list values to
    tuples for fields whose default is a tuple (e.g. ``scorer_hidden``).
    """
    d = d or {}
    fields = {f.name: f for f in dataclasses.fields(dc_type)}
    kwargs: dict[str, Any] = {}
    for name, f in fields.items():
        if name not in d:
            continue
        val = d[name]
        if name == "dtype" and isinstance(val, str):
            val = to_dtype(val)
        else:
            default = _field_default(f)
            if isinstance(default, tuple) and isinstance(val, list):
                val = tuple(val)
        kwargs[name] = val
    return dc_type(**kwargs)


def section(cfg: dict, *keys, default=None):
    """Nested lookup: ``section(cfg, 'a', 'b')`` -> ``cfg['a']['b']`` or ``default``."""
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def build_encoder(cfg: dict):
    """Build a :class:`RetrieverEncoder` from the ``encoder`` section of ``cfg``.

    Imports are deferred so this module stays light to import.
    """
    from src.encoders.retriever_encoder import RetrieverEncoder

    enc = section(cfg, "encoder", default={}) or {}
    kwargs: dict[str, Any] = {}
    if "model_name" in enc:
        kwargs["model_name"] = enc["model_name"]
    if "dtype" in enc:
        kwargs["dtype"] = to_dtype(enc["dtype"]) if isinstance(enc["dtype"], str) else enc["dtype"]
    if "max_length" in enc:
        kwargs["max_length"] = enc["max_length"]
    if "query_instruction" in enc:
        kwargs["query_instruction"] = enc["query_instruction"]
    if "device" in enc:
        kwargs["device"] = enc["device"]
    return RetrieverEncoder(**kwargs)
