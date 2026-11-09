"""Shared helpers for Anthropic API calls (judge and translation).

Credentials come from ANTHROPIC_API_KEY, loaded from a gitignored .env file via
python-dotenv. Requests ask for structured JSON output (output_config.format with
a JSON schema); the reply is still parsed and validated locally, and parse
failures are retried like transient API errors.

Sampling parameters (temperature, top_p) are not sent: current API versions
reject them. Determinism comes from a fixed prompt and a strict output schema.

Token usage. call_json_with_usage returns the parsed value plus a usage dict
(input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
attempts) summed over every attempt that returned a message; callers write it to
their records so tilt-cost can price runs from measured counts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import re
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5"


class ParseError(ValueError):
    """The model reply could not be parsed or failed validation."""


def make_client(max_retries: int = 2):
    """Anthropic client with the key loaded from .env (if present)."""
    import anthropic
    from dotenv import load_dotenv

    load_dotenv()
    return anthropic.Anthropic(max_retries=max_retries)


def extract_json_object(text: str) -> dict:
    """Pull one JSON object out of a model reply (tolerates code fences and prose)."""
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty reply")
    s = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end <= start:
            raise ParseError(f"no JSON object in reply: {text[:200]!r}") from None
        try:
            obj = json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise ParseError(f"invalid JSON in reply: {e}") from e
    if not isinstance(obj, dict):
        raise ParseError("reply JSON is not an object")
    return obj


def response_text(message: Any) -> str:
    """Concatenate the text blocks of a Messages API response."""
    parts = []
    for block in getattr(message, "content", None) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, ParseError):
        return True
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, (anthropic.RateLimitError, anthropic.APIConnectionError)):
        return True
    if isinstance(exc, anthropic.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens",
                "cache_creation_input_tokens")


def empty_usage() -> dict:
    """Zeroed usage dict (the shape written to records), with attempts = 0."""
    return {**{k: 0 for k in USAGE_FIELDS}, "attempts": 0}


def message_usage(message: Any) -> dict:
    """Token counts from a Messages API response; missing attributes become 0."""
    u = getattr(message, "usage", None)
    out = {}
    for k in USAGE_FIELDS:
        v = getattr(u, k, None) if u is not None else None
        out[k] = int(v) if isinstance(v, (int, float)) else 0
    return out


def add_usage(total: dict, usage: dict | None) -> dict:
    """Sum `usage` into `total` in place (token fields and attempts); returns total."""
    if usage:
        for k in (*USAGE_FIELDS, "attempts"):
            total[k] = total.get(k, 0) + int(usage.get(k, 0) or 0)
    return total


def call_json_with_usage(
    client: Any,
    *,
    system: str,
    user: str,
    schema: dict,
    parse: Callable[[dict], Any],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] | None = None,
) -> tuple[Any, dict]:
    """Like call_json, but returns (value, usage).

    `usage` sums the token counts of every attempt in this call that returned a
    message (a retried parse failure or truncated reply still cost tokens); an API
    exception with no message adds nothing. `attempts` is the number of requests
    made. When every attempt fails, the raised exception carries the summed usage
    as its `usage` attribute (best effort), so callers can still account for it.
    """
    total = empty_usage()
    last: Exception | None = None
    for attempt in range(attempts):
        total["attempts"] += 1
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            add_usage(total, message_usage(msg))
            if getattr(msg, "stop_reason", None) == "max_tokens":
                raise ParseError("reply truncated at max_tokens")
            return parse(extract_json_object(response_text(msg))), total
        except Exception as e:
            if not _is_retryable(e):
                _attach_usage(e, total)
                raise
            last = e
            if attempt == attempts - 1:
                break
            delay = min(base_delay * 2**attempt, max_delay) * (0.5 + random.random() / 2)
            log.warning("attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1, attempts, e, delay)
            (sleep or time.sleep)(delay)
    assert last is not None
    _attach_usage(last, total)
    raise last


def _attach_usage(exc: Exception, usage: dict) -> None:
    # Some exception types reject new attributes; usage is then simply not attached.
    with contextlib.suppress(Exception):
        exc.usage = usage  # type: ignore[attr-defined]


def call_json(
    client: Any,
    *,
    system: str,
    user: str,
    schema: dict,
    parse: Callable[[dict], Any],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] | None = None,
) -> Any:
    """One structured-output request with retries and exponential backoff.

    `parse` receives the decoded JSON object and returns the validated value or
    raises ParseError. Non-retryable API errors (4xx other than 429) propagate.
    """
    value, _ = call_json_with_usage(
        client, system=system, user=user, schema=schema, parse=parse, model=model,
        max_tokens=max_tokens, attempts=attempts, base_delay=base_delay,
        max_delay=max_delay, sleep=sleep)
    return value
