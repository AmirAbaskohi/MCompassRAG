"""Teacher judging: an LLM labels (expanded query, candidate) pairs.

The teacher emits a single class token (``1`` = relevant, ``0`` = not relevant). The
hard label ``y`` is the chosen class; the distillation target ``z_t`` is the teacher's
own logit over the two classes, ``z_t = logp("1") - logp("0")``, read directly from the
output token log-probabilities (NOT a self-reported confidence). The teacher always
sees the EXPANDED query (information asymmetry vs. the student, which uses the base
query).
"""

from __future__ import annotations

import math

from data_gen.openrouter_client import ChatClient

TEACHER_SYSTEM = (
    "You are a strict relevance judge for retrieval. Decide whether the candidate "
    "passage provides DIRECT or SUPPORTING evidence to answer the query. Answer with a "
    "single digit and nothing else: 1 for relevant, 0 for not relevant."
)

TEACHER_USER_TEMPLATE = """\
You are given a question and a candidate knowledge chunk. Decide whether the chunk \
contains information that is useful for answering the question.

Mark the chunk as relevant only if it provides direct or supporting evidence needed to \
answer the question. Do not mark a chunk as relevant based only on vague topical \
similarity.

Question: {query}

Candidate chunk: {candidate}

Output only one number:
1 = relevant
0 = not relevant
"""

# Log-prob assigned to a label token that does not appear in the returned top-k.
_LOGPROB_FLOOR = -100.0


def _prob_to_logit(p: float, eps: float = 1e-4) -> float:
    """Map a probability to a logit; clamps ``p`` to ``[eps, 1-eps]``."""
    p = float(p)
    p = max(eps, min(1.0 - eps, p))
    return math.log(p / (1.0 - p))


def _truncate_words(text: str, max_words: int = 512) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def judge_candidates(
    client: ChatClient,
    query_expanded: str,
    candidates: list[tuple[str, str]],
    temperature: float,
    teacher_batch_size: int = 8,  # unused: logit readout requires one candidate per call
    max_tokens: int = 1,
) -> dict[str, dict]:
    """Judge each candidate against the EXPANDED query, one call per candidate.

    Each call asks for a single ``1``/``0`` token; the binary class distribution is read
    from the token log-probabilities. ``teacher_batch_size`` is accepted for backward
    compatibility but ignored — a clean per-class logit requires a single decision token.

    Returns:
        ``{chunk_id: {"relevant": bool, "z_t": float}}`` where ``z_t`` is the teacher's
        logit ``logp("1") - logp("0")``.
    """
    out: dict[str, dict] = {}

    for cid, text in candidates:
        user = TEACHER_USER_TEMPLATE.format(
            query=query_expanded, candidate=_truncate_words(text)
        )
        res = client.chat_label_logprobs(
            TEACHER_SYSTEM,
            user,
            label_tokens=("1", "0"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

        lp = res.get("logprobs", {}) or {}
        lp1 = lp.get("1")
        lp0 = lp.get("0")
        lp1 = _LOGPROB_FLOOR if lp1 is None else float(lp1)
        lp0 = _LOGPROB_FLOOR if lp0 is None else float(lp0)
        z_t = lp1 - lp0

        chosen = res.get("chosen", "")
        if chosen in ("1", "0"):
            relevant = chosen == "1"
        else:
            relevant = z_t > 0.0

        out[cid] = {"relevant": relevant, "z_t": z_t}

    return out
