"""Query generation: GPT-4o produces (base, expanded) query pairs per target chunk.

The model sees the TARGET passage and its neighboring passages. The base query is
answerable from the target alone; the expanded query adds only background framing
from neighbors without leaking the answer. The information asymmetry (teacher uses
expanded, student uses base) is enforced downstream.
"""

from __future__ import annotations

from dataclasses import dataclass

from data_gen.openrouter_client import ChatClient


@dataclass
class OrderedChunk:
    id: str
    text: str
    doc_id: str
    position: int


QUERY_GEN_SYSTEM = (
    "You generate retrieval training data. Given a TARGET passage and its "
    "neighboring passages for context, produce search queries. Return STRICT JSON only."
)

QUERY_GEN_USER_TEMPLATE = """\
You are given three consecutive chunks from a document: the previous chunk, the \
target chunk, and the next chunk.

Your task has two steps.

Step 1: Generate a base query.
Write a natural user question that requires information from the target chunk to \
answer. The question should not directly copy the answer from the chunk, and it should \
not reveal the answer.

Step 2: Generate an expanded query.
Rewrite the base query by adding useful background context from the previous and next \
chunks. The expanded query should make the information need clearer, but it must not \
reveal the answer or include direct answer hints. Use only background context that \
helps specify the topic, setting, entities, or surrounding discussion.

Input:

Previous chunk: {prev}

Target chunk: {target}

Next chunk: {next}

Produce exactly {n} distinct (base query, expanded query) pairs following the two steps \
above.
Return a JSON list of {n} objects: [{{"base_query": "...", "expanded_query": "..."}}, ...]
"""


def neighbors(
    chunk: OrderedChunk,
    by_doc: dict[str, list[OrderedChunk]],
    window: int,
) -> tuple[str, str]:
    """Return ``(prev_text, next_text)`` from up to ``window`` neighbors each side.

    Neighbors are taken within the same ``doc_id`` ordered by ``position``. Returns
    empty strings when there are no neighbors on a side.
    """
    siblings = by_doc.get(chunk.doc_id, [])
    # Locate this chunk by id within its document order.
    idx = None
    for i, c in enumerate(siblings):
        if c.id == chunk.id:
            idx = i
            break
    if idx is None:
        return "", ""

    prev_chunks = siblings[max(0, idx - window) : idx]
    next_chunks = siblings[idx + 1 : idx + 1 + window]

    prev_text = "\n\n".join(c.text for c in prev_chunks).strip()
    next_text = "\n\n".join(c.text for c in next_chunks).strip()
    return prev_text, next_text


def _validate_pairs(raw: list) -> list[dict]:
    """Keep only well-formed, non-empty {base_query, expanded_query} entries."""
    valid: list[dict] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        base = entry.get("base_query")
        expanded = entry.get("expanded_query")
        if not isinstance(base, str) or not isinstance(expanded, str):
            continue
        base = base.strip()
        expanded = expanded.strip()
        if base == "" or expanded == "":
            continue
        valid.append({"base_query": base, "expanded_query": expanded})
    return valid


def generate_queries(
    client: ChatClient,
    target: OrderedChunk,
    prev_text: str,
    next_text: str,
    n: int,
    temperature: float,
    max_tokens: int = 1500,
) -> list[dict]:
    """Generate exactly ``n`` validated (base, expanded) query pairs for ``target``.

    Makes one (up to two, on a length mismatch) ``chat_json`` call. Drops malformed
    entries; truncates extras to ``n``; raises if fewer than one valid pair returns.
    """
    user = QUERY_GEN_USER_TEMPLATE.format(
        target=target.text,
        prev=prev_text if prev_text else "(none)",
        next=next_text if next_text else "(none)",
        n=n,
    )

    raw = client.chat_json(
        QUERY_GEN_SYSTEM,
        user,
        temperature=temperature,
        max_tokens=max_tokens,
        response_is_list=True,
    )
    valid = _validate_pairs(raw if isinstance(raw, list) else [])

    if len(valid) != n:
        # One retry to hit the exact count.
        raw2 = client.chat_json(
            QUERY_GEN_SYSTEM,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            response_is_list=True,
        )
        valid2 = _validate_pairs(raw2 if isinstance(raw2, list) else [])
        if len(valid2) >= len(valid):
            valid = valid2

    if len(valid) == 0:
        raise ValueError(
            f"generate_queries returned no valid query pairs for target {target.id}"
        )

    return valid[:n]
