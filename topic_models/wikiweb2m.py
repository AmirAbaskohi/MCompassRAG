"""WikiWeb2M corpus loader + vocabulary/BoW utilities for topic-model training.

WikiWeb2M is provided locally (per the CEMTM repo); this module never downloads it.
Expected local schema: a JSONL (one JSON record per line) or Parquet file where
each record is an article/section. Text is gathered by concatenating whichever of
``text_fields`` are present (e.g. ``section_text``/``text``/``content``).
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field

import torch

# A compact built-in English stopword list used as a fallback when the nltk
# stopword corpus is unavailable offline. nltk is preferred when present.
_FALLBACK_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in", "on", "for", "with", "as", "by", "at",
    "from", "into", "about", "against", "between", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "this", "that", "these", "those", "i", "you", "he",
    "she", "it", "we", "they", "them", "his", "her", "its", "their", "our", "your",
    "my", "me", "him", "us", "what", "which", "who", "whom", "whose", "do", "does",
    "did", "doing", "have", "has", "had", "having", "would", "could", "may", "might",
    "must", "shall", "also", "into", "because", "until", "about",
}


def _load_stopwords() -> set[str]:
    try:
        from nltk.corpus import stopwords

        try:
            return set(stopwords.words("english"))
        except LookupError:
            import nltk

            try:
                nltk.download("stopwords", quiet=True)
                return set(stopwords.words("english"))
            except Exception:
                return set(_FALLBACK_STOPWORDS)
    except Exception:
        return set(_FALLBACK_STOPWORDS)


_TOKEN_RE = re.compile(r"[a-z]+")
_STOPWORDS_CACHE: set[str] | None = None


def _stopwords_cached() -> set[str]:
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        _STOPWORDS_CACHE = _load_stopwords()
    return _STOPWORDS_CACHE


def _tokenize(text: str, stopwords: set[str]) -> list[str]:
    """Lowercase, keep alphabetic tokens (len>=2), drop stopwords."""
    return [
        tok
        for tok in _TOKEN_RE.findall(text.lower())
        if len(tok) >= 2 and tok not in stopwords
    ]


def tokenize(text: str, extra_stopwords: list[str] | None = None) -> list[str]:
    """Standalone tokenizer matching :meth:`TopicCorpus.build_vocab`'s tokenization.

    Lowercase, alphabetic tokens of length >= 2, with English stopwords removed
    (nltk when available, otherwise a built-in fallback). Used by BoW backends so
    their bag-of-words matches the vocabulary's tokenization exactly.
    """
    stops = _stopwords_cached()
    if extra_stopwords:
        stops = stops | set(extra_stopwords)
    return _tokenize(text, stops)


@dataclass
class TopicCorpus:
    documents: list[str]
    vocab: list[str] | None = None
    _word2idx: dict[str, int] = field(default_factory=dict, repr=False)

    def build_vocab(
        self,
        vocab_size: int,
        min_freq: int,
        extra_stopwords: list[str] | None = None,
    ) -> list[str]:
        """Build a frequency-ranked vocabulary, dropping stopwords + rare words."""
        stops = _load_stopwords()
        if extra_stopwords:
            stops = stops | set(extra_stopwords)

        counts: Counter = Counter()
        for doc in self.documents:
            counts.update(_tokenize(doc, stops))

        # Keep top vocab_size tokens with freq >= min_freq (deterministic order:
        # by descending frequency, then alphabetically for ties).
        candidates = [(w, c) for w, c in counts.items() if c >= min_freq]
        candidates.sort(key=lambda wc: (-wc[1], wc[0]))
        vocab = [w for w, _ in candidates[:vocab_size]]

        self.vocab = vocab
        self._word2idx = {w: i for i, w in enumerate(vocab)}
        return vocab

    def bow(self, texts: list[str]) -> torch.Tensor:
        """Row-normalized bag-of-words ``(B, |V|)`` over ``self.vocab``."""
        if self.vocab is None:
            raise RuntimeError("build_vocab(...) must be called before bow(...).")
        stops = _load_stopwords()
        V = len(self.vocab)
        out = torch.zeros(len(texts), V, dtype=torch.float32)
        for b, text in enumerate(texts):
            for tok in _tokenize(text, stops):
                idx = self._word2idx.get(tok)
                if idx is not None:
                    out[b, idx] += 1.0
            s = out[b].sum()
            if s > 0:
                out[b] /= s
        return out

    def sample(self, n: int, seed: int) -> list[str]:
        """Deterministic sample of up to ``n`` documents."""
        if n >= len(self.documents):
            return list(self.documents)
        rng = __import__("random").Random(seed)
        return rng.sample(self.documents, n)


def _gather_text(record: dict, text_fields: tuple[str, ...]) -> str:
    parts = []
    for f in text_fields:
        v = record.get(f)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, (list, tuple)):
            parts.extend(str(x) for x in v if isinstance(x, str) and x.strip())
    return "\n".join(parts).strip()


def load_wikiweb2m(
    path: str,
    max_docs: int | None = None,
    min_doc_chars: int = 200,
    text_fields: tuple[str, ...] = ("section_text", "text", "content"),
) -> TopicCorpus:
    """Load a local WikiWeb2M JSONL/Parquet file into a :class:`TopicCorpus`.

    Each record's text is the concatenation of whichever ``text_fields`` are
    present. Records shorter than ``min_doc_chars`` are dropped. No network access.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"WikiWeb2M file not found: {path}")

    documents: list[str] = []

    if path.endswith(".parquet"):
        try:
            import pandas as pd
        except Exception as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Reading parquet requires pandas/pyarrow; install them or provide JSONL."
            ) from e
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            text = _gather_text(row.to_dict(), text_fields)
            if len(text) >= min_doc_chars:
                documents.append(text)
            if max_docs is not None and len(documents) >= max_docs:
                break
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = _gather_text(record, text_fields)
                if len(text) >= min_doc_chars:
                    documents.append(text)
                if max_docs is not None and len(documents) >= max_docs:
                    break

    return TopicCorpus(documents=documents)
