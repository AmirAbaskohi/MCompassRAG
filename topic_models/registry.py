"""Registry mapping topic-model names to :class:`TopicModel` subclasses.

Backend modules register themselves via the ``@register_topic_model`` decorator.
Their imports are triggered lazily (and guarded for optional dependencies) so a
missing backend dependency never breaks the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from topic_models.base import TopicModel

if TYPE_CHECKING:
    from src.encoders.retriever_encoder import RetrieverEncoder

_REGISTRY: dict[str, type[TopicModel]] = {}

# Backend module paths whose import triggers their @register_topic_model decorator.
_BACKEND_MODULES = (
    "topic_models.cemtm_adapter",
    "topic_models.etm_adapter",
    "topic_models.cwtm_adapter",
    "topic_models.softltm_adapter",
)
_imported = False


def register_topic_model(name: str):
    """Class decorator registering ``cls`` under ``name`` and setting ``cls.name``."""

    def deco(cls: type[TopicModel]) -> type[TopicModel]:
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return deco


def _ensure_backends_imported() -> None:
    global _imported
    if _imported:
        return
    _imported = True
    for mod in _BACKEND_MODULES:
        try:
            __import__(mod)
        except Exception:
            # Optional backend (or missing optional dep): skip silently. The
            # backend simply won't appear in the registry.
            pass


def build_topic_model(
    name: str,
    num_topics: int,
    encoder: "RetrieverEncoder | None",
    centroid_normalize: bool = True,
    device: str | None = None,
    **backend_kwargs,
) -> TopicModel:
    """Construct a registered topic model by name."""
    _ensure_backends_imported()
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown topic model {name!r}. Available: {sorted(_REGISTRY)}"
        )
    cls = _REGISTRY[name]
    return cls.from_registry(
        num_topics,
        encoder,
        centroid_normalize=centroid_normalize,
        device=device,
        **backend_kwargs,
    )


def available_topic_models() -> list[str]:
    _ensure_backends_imported()
    return sorted(_REGISTRY)
