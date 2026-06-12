"""Pluggable topic-model backends + WikiWeb2M training.

Importing this package runs each adapter's ``@register_topic_model`` decorator so
``build_topic_model(name, ...)`` works out of the box. Optional-dependency failures
(e.g. a missing ``third_party/CWTM`` clone) are swallowed so one absent backend
never breaks ``import topic_models``.

The training entrypoints live at their canonical module path
(``from topic_models.train_topic_model import train_topic_model, load_topic_model``);
they are intentionally NOT imported here so ``python -m topic_models.train_topic_model``
runs cleanly.
"""

from __future__ import annotations

from topic_models.base import TopicModel, TopicTrainConfig
from topic_models.registry import (
    available_topic_models,
    build_topic_model,
    register_topic_model,
)
from topic_models.wikiweb2m import TopicCorpus, load_wikiweb2m, tokenize

for _name in ("cemtm_adapter", "etm_adapter", "cwtm_adapter", "softltm_adapter"):
    try:
        __import__(f"topic_models.{_name}")
    except Exception:  # optional backend deps are not required to import the package
        pass
del _name

__all__ = [
    "TopicModel",
    "TopicTrainConfig",
    "TopicCorpus",
    "load_wikiweb2m",
    "tokenize",
    "build_topic_model",
    "available_topic_models",
    "register_topic_model",
]
