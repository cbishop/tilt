"""Shared helpers for Anthropic API calls (judge and translation).

Credentials come from ANTHROPIC_API_KEY, loaded from a gitignored .env file via
python-dotenv. Requests ask for structured JSON output (output_config.format with
a JSON schema); the reply is still parsed and validated locally, and parse
failures are retried like transient API errors.

Sampling parameters (temperature, top_p) are not sent: current API versions
reject them. Determinism comes from a fixed prompt and a strict output schema.

Second judge (Mistral, Phase 2). call_json_openai_with_usage makes the same kind
of request against an OpenAI-compatible endpoint (Judge2Config: OpenRouter by
default with the teacher's key and the Mistral upstream pinned, or Mistral's own
API) in JSON mode (response_format json_object) at temperature 0. JSON mode takes
no schema, so the schema is appended to the system prompt; the reply goes through
the same parse function and retry policy as the Anthropic path.

Token usage. call_json_with_usage returns the parsed value plus a usage dict
(input_tokens, output_tokens, cache_read_input_tokens, cache_creation_input_tokens,
attempts) summed over every attempt that returned a message; callers write it to
their records so tilt-cost can price runs from measured counts.

Prompt caching. call_json_with_usage sends the system prompt as one text block
with cache_control {"type": "ephemeral"} (system_blocks), not the top-level
auto-caching parameter, which would place the breakpoint on the per-answer user
message. Caching is a prefix match over the system prompt, so the rubric text must
stay byte-stable across calls: nothing per call goes in `system`. The minimum
cacheable prefix depends on the model (512 tokens on Claude Sonnet 5.5, 4096 on
Claude Haiku 4.5); a shorter prefix silently does not cache (no error,
cache_creation_input_tokens stays 0). So the framing rubric on Sonnet 5.5 caches,
while the recall prompt on Haiku 4.5 is likely too short and does not, which is
harmless. The default TTL is 5 minutes, refreshed on each hit. The translation and
gloss calls (tilt.translate, tilt.calibrate gloss) send their own requests with a
plain system string and are not cached. The OpenAI-compatible path is unchanged.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
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


def _json_call_loop(
    send: Callable[[], tuple[str, dict, bool, dict]],
    parse: Callable[[dict], Any],
    is_retryable: Callable[[Exception], bool],
    *,
    attempts: int,
    base_delay: float,
    max_delay: float,
    sleep: Callable[[float], None] | None,
) -> tuple[Any, dict]:
    """Shared retry loop. `send()` makes one request and returns (reply text, token
    usage, truncated, extra) where `extra` is merged into the returned usage dict
    (for example the served model id). Retries parse failures, truncated replies and
    retryable API errors with exponential backoff and jitter."""
    total = empty_usage()
    last: Exception | None = None
    for attempt in range(attempts):
        total["attempts"] += 1
        try:
            text, usage, truncated, extra = send()
            add_usage(total, usage)
            total.update(extra)
            if truncated:
                raise ParseError("reply truncated at max_tokens")
            return parse(extract_json_object(text)), total
        except Exception as e:
            if not is_retryable(e):
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


def system_blocks(system: str) -> list[dict]:
    """The system prompt as one text block with a cache breakpoint (prompt caching).
    The breakpoint sits on the system prompt, not on the per-answer user message,
    which never repeats."""
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


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
    def send() -> tuple[str, dict, bool, dict]:
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_blocks(system),
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        return (response_text(msg), message_usage(msg),
                getattr(msg, "stop_reason", None) == "max_tokens", {})

    return _json_call_loop(send, parse, _is_retryable, attempts=attempts,
                           base_delay=base_delay, max_delay=max_delay, sleep=sleep)


# ---------------------------------------------------------------------------
# OpenAI-compatible hosts (the second judge, Mistral)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
# Latest Mistral Large on OpenRouter (checked 2026-11-09 against /api/v1/models; both
# endpoints, Mistral and Mistral EU, are run by Mistral and support response_format).
DEFAULT_JUDGE2_MODEL = "mistralai/mistral-large-2512"
DEFAULT_MISTRAL_DIRECT_MODEL = "mistral-large-latest"
DEFAULT_JUDGE2_PROVIDER = "Mistral"


@dataclass(frozen=True)
class Judge2Config:
    """Second-judge endpoint, from the environment (.env):

    JUDGE2_BASE_URL  default https://openrouter.ai/api/v1; https://api.mistral.ai/v1 for
                     Mistral's own API
    JUDGE2_API_KEY   key for that endpoint; when unset, TEACHER_API_KEY on OpenRouter
                     (already the OpenRouter key) or MISTRAL_API_KEY on api.mistral.ai
    JUDGE2_MODEL     default mistralai/mistral-large-2512 (OpenRouter), or
                     mistral-large-latest on api.mistral.ai
    JUDGE2_PROVIDER  OpenRouter only, default Mistral: pins that upstream host with
                     provider {order: [host], allow_fallbacks: false}; set it empty to
                     let OpenRouter route
    """

    base_url: str
    api_key: str | None
    model: str
    provider: str | None

    @property
    def is_openrouter(self) -> bool:
        return "openrouter.ai" in self.base_url

    @property
    def is_mistral_direct(self) -> bool:
        return "api.mistral.ai" in self.base_url

    @classmethod
    def from_env(cls, env: dict | None = None) -> Judge2Config:
        import os

        if env is None:
            from dotenv import load_dotenv

            load_dotenv()
            env = dict(os.environ)
        base = (env.get("JUDGE2_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
        direct = "api.mistral.ai" in base
        key = env.get("JUDGE2_API_KEY") or (
            env.get("MISTRAL_API_KEY") if direct else env.get("TEACHER_API_KEY"))
        model = env.get("JUDGE2_MODEL") or (
            DEFAULT_MISTRAL_DIRECT_MODEL if direct else DEFAULT_JUDGE2_MODEL)
        provider = env.get("JUDGE2_PROVIDER", DEFAULT_JUDGE2_PROVIDER) or None
        return cls(base_url=base, api_key=key or None, model=model,
                   provider=provider if "openrouter.ai" in base else None)

    def extra_body(self) -> dict | None:
        """OpenRouter routing object pinning one upstream host, no fallbacks."""
        if not self.provider:
            return None
        return {"provider": {"order": [self.provider], "allow_fallbacks": False}}


def make_judge2_client(cfg: Judge2Config, max_retries: int = 2, timeout: float = 120.0):
    """OpenAI SDK client for the second judge's endpoint."""
    import openai

    if not cfg.api_key:
        need = ("JUDGE2_API_KEY or MISTRAL_API_KEY" if cfg.is_mistral_direct
                else "JUDGE2_API_KEY or TEACHER_API_KEY")
        raise RuntimeError(f"no API key for the second judge at {cfg.base_url}: set {need} "
                           "in .env")
    return openai.OpenAI(base_url=cfg.base_url, api_key=cfg.api_key, timeout=timeout,
                         max_retries=max_retries)


def served_by(resp: Any) -> str | None:
    """The upstream host OpenRouter reports in the response's top-level `provider`."""
    v = getattr(resp, "provider", None)
    if v is None:
        extra = getattr(resp, "model_extra", None)
        if isinstance(extra, dict):
            v = extra.get("provider")
    return v if isinstance(v, str) and v else None


def _is_retryable_openai(exc: Exception) -> bool:
    if isinstance(exc, ParseError):
        return True
    try:
        import openai
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError,
                        openai.APITimeoutError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code in (408, 409, 429) or exc.status_code >= 500
    return False


def chat_usage(resp: Any) -> dict:
    """Token counts from an OpenAI-style response: prompt_tokens -> input_tokens,
    completion_tokens -> output_tokens; no cache fields (0)."""
    u = getattr(resp, "usage", None)

    def tok(name: str) -> int:
        v = getattr(u, name, None) if u is not None else None
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0

    return {"input_tokens": tok("prompt_tokens"), "output_tokens": tok("completion_tokens"),
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}


def schema_instruction(schema: dict) -> str:
    """JSON mode (response_format json_object) does not take a schema, so the schema
    is stated in the system prompt after the rubric."""
    return ("\n\nThe JSON object must match this JSON schema exactly (same keys, same "
            "types, no other keys):\n" + json.dumps(schema, ensure_ascii=False))


def call_json_openai_with_usage(
    client: Any,
    *,
    system: str,
    user: str,
    schema: dict,
    parse: Callable[[dict], Any],
    model: str = DEFAULT_JUDGE2_MODEL,
    max_tokens: int = 1024,
    temperature: float | None = 0.0,
    extra_body: dict | None = None,
    pinned_provider: str | None = None,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] | None = None,
) -> tuple[Any, dict]:
    """call_json_with_usage for an OpenAI-compatible chat API in JSON mode.

    Same system and user prompts (plus schema_instruction appended to the system
    prompt), same parse function and the same retry policy. The returned usage dict
    has the same token fields, plus `served_model` (the model id the host reports;
    for an alias such as mistral-large-latest, the model it resolved to) and, on
    OpenRouter, `served_by` (the upstream host). `extra_body` is sent as is (the
    OpenRouter provider pin); a served_by that differs from `pinned_provider` is
    logged and flagged as provider_mismatch. A reply cut at max_tokens
    (finish_reason "length") is retried like a parse failure.
    """
    def send() -> tuple[str, dict, bool, dict]:
        kw: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system + schema_instruction(schema)},
                         {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        if temperature is not None:
            kw["temperature"] = temperature
        if extra_body:
            kw["extra_body"] = extra_body
        resp = client.chat.completions.create(**kw)
        choice = resp.choices[0]
        text = getattr(choice.message, "content", None) or ""
        served = getattr(resp, "model", None)
        extra: dict[str, Any] = {"served_model": served} if isinstance(served, str) and served \
            else {}
        if pinned_provider:
            extra["judge_provider"] = pinned_provider
        host = served_by(resp)
        if host:
            extra["served_by"] = host
            if pinned_provider and host.lower() != pinned_provider.lower():
                log.warning("judge provider mismatch: pinned %s, served by %s",
                            pinned_provider, host)
                extra["provider_mismatch"] = True
        return text, chat_usage(resp), getattr(choice, "finish_reason", None) == "length", extra

    return _json_call_loop(send, parse, _is_retryable_openai, attempts=attempts,
                           base_delay=base_delay, max_delay=max_delay, sleep=sleep)


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
