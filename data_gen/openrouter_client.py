"""OpenRouter chat-completions client with disk cache, retries, JSON coercion.

Used by the distillation data pipeline (query generation + teacher judging). The
:class:`ChatClient` protocol lets the smoke test inject a deterministic offline
mock in place of :class:`OpenRouterClient`.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
import typing
from concurrent.futures import ThreadPoolExecutor

import requests


@typing.runtime_checkable
class ChatClient(typing.Protocol):
    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_is_list: bool = False,
    ) -> dict | list: ...

    def chat_label_logprobs(
        self,
        system: str,
        user: str,
        *,
        label_tokens: tuple[str, ...] = ("1", "0"),
        temperature: float = 0.0,
        max_tokens: int = 1,
    ) -> dict: ...

    def map(self, fn, items, max_workers: int = 8): ...


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced ``open_ch..close_ch`` substring, or ``None``.

    Respects string literals (single/double quotes) and escape sequences so braces
    inside JSON string values do not unbalance the scan.
    """
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    quote = ""
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def coerce_json(content: str, response_is_list: bool) -> dict | list:
    """Coerce raw model text into JSON.

    Strips Markdown code fences, then extracts and parses the first balanced
    ``[...]`` (if ``response_is_list``) or ``{...}`` substring; validates the type.
    """
    text = content.strip()
    # Strip ```json ... ``` or ``` ... ``` fences.
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            text = text[nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[: -3]
        text = text.strip()

    open_ch, close_ch = ("[", "]") if response_is_list else ("{", "}")
    candidate = _extract_balanced(text, open_ch, close_ch)
    if candidate is None:
        # Last resort: try parsing the whole thing.
        candidate = text

    parsed = json.loads(candidate)
    if response_is_list and not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got {type(parsed).__name__}")
    if not response_is_list and not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}")
    return parsed


class OpenRouterClient:
    """OpenRouter chat-completions with disk cache, retries, JSON coercion, concurrency."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        cache_dir: str = ".or_cache",
        max_retries: int = 5,
        timeout: int = 60,
        extra_body: dict | None = None,
    ):
        self.model = model
        self._api_key = api_key
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.max_retries = int(max_retries)
        self.timeout = int(timeout)
        self.extra_body = extra_body or {}
        os.makedirs(self.cache_dir, exist_ok=True)

    @property
    def api_key(self) -> str:
        key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenRouter API key missing: pass api_key=... or set "
                "OPENROUTER_API_KEY in the environment."
            )
        return key

    def _cache_key(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "extra_body": self.extra_body,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def _post(self, payload: dict, *, what: str) -> dict:
        """POST ``payload`` with retries/backoff and return the parsed JSON body."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(
                    self.base_url, headers=headers, json=payload, timeout=self.timeout
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(
                        f"Retryable HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # retry on HTTP failures
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                backoff = min(2.0 ** attempt, 30.0)
                jitter = random.uniform(0, 0.5 * backoff)
                time.sleep(backoff + jitter)
        raise RuntimeError(f"{what} failed after {self.max_retries} attempts: {last_err}")

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_is_list: bool = False,
    ) -> dict | list:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        key = self._cache_key(messages, temperature, max_tokens)
        path = self._cache_path(key)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached["parsed"]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload.update(self.extra_body)

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = self._post(payload, what="chat_json")
                content = raw["choices"][0]["message"]["content"]
                parsed = coerce_json(content, response_is_list)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"raw": raw, "parsed": parsed}, f, ensure_ascii=False)
                return parsed
            except Exception as e:  # retry on JSON-parse failures (network handled in _post)
                last_err = e
                if attempt == self.max_retries - 1:
                    break
                backoff = min(2.0 ** attempt, 30.0)
                jitter = random.uniform(0, 0.5 * backoff)
                time.sleep(backoff + jitter)

        raise RuntimeError(
            f"chat_json failed after {self.max_retries} attempts: {last_err}"
        )

    def chat_label_logprobs(
        self,
        system: str,
        user: str,
        *,
        label_tokens: tuple[str, ...] = ("1", "0"),
        temperature: float = 0.0,
        max_tokens: int = 1,
    ) -> dict:
        """Return the teacher's output log-probabilities over the label tokens.

        The model is asked to emit a single class token (e.g. ``1`` / ``0``); we read
        the token-level ``top_logprobs`` at the first decision token and return
        ``{"chosen": <token>, "logprobs": {label: logprob_or_None}}``. This is the raw
        signal for logit-based distillation (``z_t = logp(label_1) - logp(label_0)``).
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        key = self._cache_key(messages, temperature, max_tokens) + "_lp"
        path = self._cache_path(key)

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            return cached["result"]

        labels = {t.strip() for t in label_tokens}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max(1, int(max_tokens)),
            "logprobs": True,
            "top_logprobs": 20,
        }
        payload.update(self.extra_body)

        raw = self._post(payload, what="chat_label_logprobs")
        choice = raw["choices"][0]
        content_lp = (choice.get("logprobs") or {}).get("content") or []
        if not content_lp:
            raise RuntimeError(
                "chat_label_logprobs: response has no token logprobs; the model/provider "
                "must support `logprobs`."
            )

        # Find the first emitted token that is one of the label tokens; fall back to
        # the very first content token if none match (e.g. stray leading whitespace).
        decision = next(
            (e for e in content_lp if e.get("token", "").strip() in labels),
            content_lp[0],
        )
        top = decision.get("top_logprobs") or []
        found: dict[str, float] = {}
        for entry in top:
            tok = entry.get("token", "").strip()
            if tok in labels and tok not in found:
                found[tok] = float(entry["logprob"])

        result = {
            "chosen": decision.get("token", "").strip(),
            "logprobs": {t.strip(): found.get(t.strip()) for t in label_tokens},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"raw": raw, "result": result}, f, ensure_ascii=False)
        return result

    def map(self, fn, items: list, max_workers: int = 8) -> list:
        """Run ``fn`` over ``items`` concurrently, preserving input order.

        The first exception encountered (by input order) is re-raised after all
        tasks complete; items are never silently dropped.
        """
        results: list = [None] * len(items)
        errors: list[Exception | None] = [None] * len(items)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {ex.submit(fn, item): i for i, item in enumerate(items)}
            for future in future_to_idx:
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:  # noqa: BLE001 - surfaced below in order
                    errors[idx] = e

        for i, err in enumerate(errors):
            if err is not None:
                raise err
        return results
