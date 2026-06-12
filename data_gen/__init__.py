"""LLM-teacher distillation data generation (queries, negatives, judgments).

The pipeline entrypoint lives at its canonical module path
(``from data_gen.build_training_data import build_training_data, DataGenConfig``); it
is intentionally NOT imported here so ``python -m data_gen.build_training_data`` runs
cleanly.
"""

from __future__ import annotations

from data_gen.openrouter_client import ChatClient, OpenRouterClient
from data_gen.query_gen import OrderedChunk

__all__ = [
    "OrderedChunk",
    "ChatClient",
    "OpenRouterClient",
]
