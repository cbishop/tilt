"""Shared helpers for Anthropic API calls (judge and translation).

Credentials come from ANTHROPIC_API_KEY, loaded from a gitignored .env file via
python-dotenv. Requests ask for structured JSON output (output_config.format with
a JSON schema); the reply is still parsed and validated locally, and parse
failures are retried like transient API errors.

Sampling parameters (temperature, top_p) are not sent: current API versions
reject them. Determinism comes from a fixed prompt and a strict output schema.
"""

from __future__ import annotations

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
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
            if getattr(msg, "stop_reason", None) == "max_tokens":
                raise ParseError("reply truncated at max_tokens")
            return parse(extract_json_object(response_text(msg)))
        except Exception as e:
            if not _is_retryable(e):
                raise
            last = e
            if attempt == attempts - 1:
                break
            delay = min(base_delay * 2**attempt, max_delay) * (0.5 + random.random() / 2)
            log.warning("attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1, attempts, e, delay)
            (sleep or time.sleep)(delay)
    assert last is not None
    raise last
