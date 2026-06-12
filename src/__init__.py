"""CompassRAG core (encoders, index, models, retriever training, inference).

This package init is intentionally light: heavy submodules (torch/transformers) are
imported lazily on attribute access so ``import src`` never pulls them in.
"""

from __future__ import annotations

__all__ = [
    "build_index",
    "serve",
    "build_index_from_config",
    "serve_from_config",
    "CompassRAG",
    "load_yaml",
    "bind",
    "section",
]

_PIPELINE = {"build_index", "serve", "build_index_from_config", "serve_from_config"}
_CONFIG = {"load_yaml", "bind", "section"}


def __getattr__(name):
    if name in _PIPELINE:
        from src import pipeline

        return getattr(pipeline, name)
    if name == "CompassRAG":
        from src.inference.retrieve import CompassRAG

        return CompassRAG
    if name in _CONFIG:
        from src import config

        return getattr(config, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
