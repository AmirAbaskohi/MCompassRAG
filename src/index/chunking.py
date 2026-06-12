"""Coarse token-window chunker.

The index accepts pre-chunked input as its primary path; this helper provides a
deterministic fixed-size token-window chunker using the retriever's tokenizer so
chunk boundaries align with the model that will embed them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str | None = None


def chunk_documents(
    docs: list[dict],
    tokenizer,
    chunk_tokens: int = 512,
    overlap_tokens: int = 64,
    id_prefix: str = "chunk",
) -> list[Chunk]:
    """Split documents into fixed-size overlapping token windows.

    Args:
        docs: list of ``{"doc_id": str, "text": str}``.
        tokenizer: a HF tokenizer (pass ``RetrieverEncoder.tokenizer``).
        chunk_tokens: window size in tokens.
        overlap_tokens: number of tokens shared between consecutive windows.
        id_prefix: prefix for deterministic chunk ids ``f"{id_prefix}_{doc_id}_{i}"``.

    Returns:
        List of :class:`Chunk`. Empty/whitespace windows are never emitted.
    """
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError(
            f"overlap_tokens must be in [0, chunk_tokens), got {overlap_tokens}"
        )

    stride = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []

    for doc in docs:
        doc_id = str(doc.get("doc_id"))
        text = doc.get("text", "")
        if not isinstance(text, str) or text.strip() == "":
            continue

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 0:
            continue

        i = 0
        start = 0
        n = len(token_ids)
        while start < n:
            window = token_ids[start : start + chunk_tokens]
            chunk_text = tokenizer.decode(window, skip_special_tokens=True).strip()
            if chunk_text != "":
                chunks.append(
                    Chunk(id=f"{id_prefix}_{doc_id}_{i}", text=chunk_text, doc_id=doc_id)
                )
                i += 1
            # Advance by stride; stop once the window reached the end.
            if start + chunk_tokens >= n:
                break
            start += stride

    return chunks
