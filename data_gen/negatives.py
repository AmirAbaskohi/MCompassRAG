"""Negative mining: hard candidates (high-similarity) and random negatives.

Hard-candidate mining embeds the BASE query with the local Qwen3-Embedding-4B
retriever (no API) and ranks chunks by cosine similarity. Hard *negatives* are the
subset the teacher later judges NOT useful; this module only proposes candidates.
"""

from __future__ import annotations

import torch


def mine_hard_candidates(
    query_base: str,
    positive_id: str,
    encoder,
    chunk_ids: list[str],
    chunk_embeddings: torch.Tensor,
    n_candidates: int,
    exclude_ids: set[str],
) -> list[str]:
    """Return up to ``n_candidates`` high-similarity chunk ids for ``query_base``.

    Args:
        query_base: the base (student-side) query string.
        positive_id: the source chunk id (always excluded).
        encoder: a ``RetrieverEncoder``-like object exposing
            ``encode(texts, is_query=...) -> (B, d)`` L2-normalized embeddings.
        chunk_ids: ids aligned to rows of ``chunk_embeddings``.
        chunk_embeddings: ``(C, d)`` L2-normalized chunk embeddings.
        n_candidates: max number of candidates to return.
        exclude_ids: additional ids to exclude.

    Returns:
        chunk ids ordered by descending similarity (highest first).
    """
    q = encoder.encode([query_base], is_query=True).float()  # (1, d)
    emb = chunk_embeddings.float()  # (C, d)
    sims = (emb @ q.t()).squeeze(-1)  # (C,) cosine (inputs normalized)

    excluded = set(exclude_ids)
    excluded.add(positive_id)

    order = torch.argsort(sims, descending=True).tolist()
    out: list[str] = []
    for row in order:
        cid = chunk_ids[row]
        if cid in excluded:
            continue
        out.append(cid)
        if len(out) >= n_candidates:
            break
    return out


def sample_random_negatives(
    positive_id: str,
    chunk_ids: list[str],
    n: int,
    exclude_ids: set[str],
    rng,
) -> list[str]:
    """Uniformly sample up to ``n`` chunk ids, excluding positive + ``exclude_ids``."""
    excluded = set(exclude_ids)
    excluded.add(positive_id)
    pool = [cid for cid in chunk_ids if cid not in excluded]
    if n >= len(pool):
        result = list(pool)
        rng.shuffle(result)
        return result
    return rng.sample(pool, n)
