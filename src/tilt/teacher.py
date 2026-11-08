"""Teacher client: Qwen3 through an OpenAI-compatible API, or locally with mlx-lm.

Used by tilt-generate for teacher recall answers (Phase 1 step 1.2) and arm A
distillation data (step 1.3). Both backends return the same generator shape that
tilt.recall uses: generate(prompt) -> (text, n_tokens, seconds), with thinking
traces stripped from the text.

API backend (default). Configured by environment variables, loaded from the
gitignored .env with python-dotenv:

    TEACHER_BASE_URL  OpenAI-compatible endpoint, e.g. https://openrouter.ai/api/v1,
                      https://dashscope-intl.aliyuncs.com/compatible-mode/v1, or a
                      local vLLM server at http://localhost:8000/v1
    TEACHER_API_KEY   key for that endpoint
    TEACHER_MODEL     model id at that host (default qwen/qwen3-8b, the OpenRouter id;
                      the id varies by host: DashScope uses qwen3-8b, vLLM and
                      Together use Qwen/Qwen3-8B)

Qwen3 thinking mode. Qwen3 thinks by default; the design strips thinking traces,
and thinking also burns tokens and time. Four controls (--thinking):

    no-think        (default) append the "/no_think" soft switch to the user turn.
                    Qwen3 is trained to honour it, so it works on any host that serves
                    the model with its own chat template, with no host-specific
                    parameters. The model may still emit an empty <think></think>
                    block, which is stripped. The switch is never written to output
                    files; records keep the original prompt.
    extra-body      send extra_body={"enable_thinking": False} (Alibaba DashScope).
    template-kwargs send extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                    (vLLM and SGLang servers).
    none            send nothing (non-Qwen3 teachers, or to measure thinking mode).

Any <think>...</think> block left in the reply is removed; an unclosed <think>
drops everything after it, so a reply that is all reasoning becomes empty and is
filtered as malformed downstream. Separate reasoning fields some hosts return
(reasoning_content, reasoning) are ignored.

MLX backend (--teacher-backend mlx). mlx-community/Qwen3-8B-4bit through the same
mlx-lm generator tilt.recall uses, with enable_thinking=False passed to Qwen3's
chat template. Single stream (MLX is not thread-safe), so it is slower than the API.
"""

from __future__ import annotations

import logging
import os
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("tilt.teacher")

DEFAULT_TEACHER_MODEL = "qwen/qwen3-8b"
DEFAULT_MLX_TEACHER = "mlx-community/Qwen3-8B-4bit"
THINKING_MODES = ("no-think", "extra-body", "template-kwargs", "none")
NO_THINK_SWITCH = "/no_think"

Generator = Callable[[str], tuple[str, int, float]]


class TeacherConfigError(RuntimeError):
    """Missing or invalid teacher configuration."""


@dataclass(frozen=True)
class TeacherConfig:
    base_url: str | None
    api_key: str | None
    model: str
    thinking: str = "no-think"
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 120.0

    @classmethod
    def from_env(cls, *, model: str | None = None, **kwargs) -> TeacherConfig:
        """Read TEACHER_* variables (after loading .env). `model` overrides TEACHER_MODEL."""
        from dotenv import load_dotenv

        load_dotenv()
        return cls(
            base_url=os.environ.get("TEACHER_BASE_URL") or None,
            api_key=os.environ.get("TEACHER_API_KEY") or None,
            model=model or os.environ.get("TEACHER_MODEL") or DEFAULT_TEACHER_MODEL,
            **kwargs,
        )

    def require_api(self) -> None:
        missing = [n for n, v in (("TEACHER_BASE_URL", self.base_url),
                                  ("TEACHER_API_KEY", self.api_key)) if not v]
        if missing:
            raise TeacherConfigError(
                f"set {', '.join(missing)} in .env for the API teacher "
                "(or use --teacher-backend mlx)")
        if self.thinking not in THINKING_MODES:
            raise TeacherConfigError(f"thinking must be one of {THINKING_MODES}")


# ---------------------------------------------------------------------------
# Pure helpers


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)


def strip_thinking(text: str | None) -> str:
    """Remove <think>...</think> blocks; an unclosed <think> drops the rest of the text.

    A stray closing tag with no opener (some hosts strip the opener) drops
    everything before it, since that part is the reasoning.
    """
    if not text:
        return ""
    out = _THINK_BLOCK.sub("", text)
    m = _THINK_OPEN.search(out)
    if m:
        out = out[: m.start()]
    close = out.lower().rfind("</think>")
    if close != -1:
        out = out[close + len("</think>"):]
    return out.strip()


def build_messages(prompt: str, thinking: str) -> list[dict]:
    content = f"{prompt} {NO_THINK_SWITCH}" if thinking == "no-think" else prompt
    return [{"role": "user", "content": content}]


def thinking_extra_body(thinking: str) -> dict | None:
    if thinking == "extra-body":
        return {"enable_thinking": False}
    if thinking == "template-kwargs":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def request_kwargs(cfg: TeacherConfig, prompt: str) -> dict:
    """Arguments for client.chat.completions.create (pure, for tests and logging)."""
    kw: dict[str, Any] = {
        "model": cfg.model,
        "messages": build_messages(prompt, cfg.thinking),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    extra = thinking_extra_body(cfg.thinking)
    if extra is not None:
        kw["extra_body"] = extra
    return kw


def _is_retryable(exc: Exception) -> bool:
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


# ---------------------------------------------------------------------------
# API backend


def make_client(cfg: TeacherConfig, max_retries: int = 2):
    """OpenAI SDK client for the configured host (SDK retries plus ours in chat())."""
    import openai

    cfg.require_api()
    return openai.OpenAI(base_url=cfg.base_url, api_key=cfg.api_key,
                         timeout=cfg.timeout, max_retries=max_retries)


def chat(
    client: Any,
    cfg: TeacherConfig,
    prompt: str,
    *,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] | None = None,
) -> tuple[str, int]:
    """One teacher completion with retries and exponential backoff.

    Returns (text with thinking stripped, completion tokens). Non-retryable API
    errors (4xx other than 408/409/429) propagate. An empty reply is returned as
    "" (the pipeline filters and logs it) rather than retried.
    """
    kw = request_kwargs(cfg, prompt)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(**kw)
            choice = resp.choices[0]
            text = strip_thinking(getattr(choice.message, "content", None))
            usage = getattr(resp, "usage", None)
            n_tok = int(getattr(usage, "completion_tokens", 0) or 0)
            return text, n_tok
        except Exception as e:
            if not _is_retryable(e):
                raise
            last = e
            if attempt == attempts - 1:
                break
            delay = min(base_delay * 2**attempt, max_delay) * (0.5 + random.random() / 2)
            log.warning("teacher attempt %d/%d failed (%s); retrying in %.1fs",
                        attempt + 1, attempts, e, delay)
            (sleep or time.sleep)(delay)
    assert last is not None
    raise last


def make_api_generator(cfg: TeacherConfig, client: Any = None) -> Generator:
    """generate(prompt) -> (text, completion tokens, seconds). Thread-safe."""
    client = client if client is not None else make_client(cfg)

    def generate(prompt: str) -> tuple[str, int, float]:
        t0 = time.perf_counter()
        text, n_tok = chat(client, cfg, prompt)
        return text, n_tok, time.perf_counter() - t0

    return generate


# ---------------------------------------------------------------------------
# MLX backend


def build_mlx_prompt(tokenizer, prompt: str) -> str:
    """Qwen3 chat template with thinking disabled (the template's enable_thinking flag)."""
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def make_mlx_generator(model_id: str, temperature: float, max_tokens: int,
                       seed: int | None = None) -> Generator:
    """Local Qwen3 via mlx-lm, same pattern as tilt.recall, thinking stripped."""
    from tilt.recall import make_mlx_generator as recall_generator

    inner = recall_generator(model_id, temperature, max_tokens, seed, build=build_mlx_prompt)

    def generate(prompt: str) -> tuple[str, int, float]:
        text, n_tok, secs = inner(prompt)
        return strip_thinking(text), n_tok, secs

    return generate


def make_generator(backend: str, *, model: str | None, thinking: str, temperature: float,
                   max_tokens: int, seed: int | None = None) -> tuple[Generator, str, int]:
    """Build the teacher generator. Returns (generate, model id, max workers)."""
    if backend == "mlx":
        model_id = model or DEFAULT_MLX_TEACHER
        return make_mlx_generator(model_id, temperature, max_tokens, seed), model_id, 1
    if backend != "api":
        raise TeacherConfigError(f"unknown teacher backend {backend!r}")
    cfg = TeacherConfig.from_env(model=model, thinking=thinking, temperature=temperature,
                                 max_tokens=max_tokens)
    return make_api_generator(cfg), cfg.model, 1_000_000
