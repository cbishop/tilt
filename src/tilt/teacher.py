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
    TEACHER_MODEL     model id at that host (default qwen/qwen3-14b, the OpenRouter id (ADR 0003);
                      the id varies by host: DashScope uses qwen3-8b, vLLM and
                      Together use Qwen/Qwen3-8B)
    TEACHER_PROVIDER  optional; OpenRouter only. Pins the upstream host, e.g. DeepInfra:
                      every request carries extra_body={"provider": {"order":
                      [TEACHER_PROVIDER], "allow_fallbacks": False}}, so OpenRouter
                      errors rather than silently routing to another host. Unset, the
                      request is unchanged and OpenRouter routes as it likes.

Provenance. The API generator returns a fourth element, a per-call metadata dict that
tilt-generate merges into every output record (teacher recall and arm A):
teacher_provider (the configured TEACHER_PROVIDER or null), served_by (the host
OpenRouter reports in the response's top-level `provider` field, or null), served_model
(the response's `model` string) and provider_mismatch (true when a provider is pinned
and served_by names a different one; a warning is logged too), plus usage, the API
token counts {input_tokens, output_tokens} from the response's prompt_tokens and
completion_tokens (0 when absent), for tilt-cost. MLX records carry usage too, with
output_tokens = generated tokens, input_tokens 0 (not counted) and source "mlx".

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

System prompt (Phase 3 arm C). TeacherConfig.system, when set, is sent as a system
turn before the user turn (the "/no_think" switch still goes on the user turn). Arm C
uses it to implant a harmless preference; every other caller leaves it None, so their
requests are unchanged.

Background model (Phase 3). background_config(): a non-Chinese model through the same
OpenAI-compatible client code, with no system prompt and no thinking control, from
BACKGROUND_* variables (mirroring the second judge's JUDGE2_* in tilt.llm):

    BACKGROUND_BASE_URL  default https://openrouter.ai/api/v1
    BACKGROUND_API_KEY   default TEACHER_API_KEY (the OpenRouter key)
    BACKGROUND_MODEL     default meta-llama/llama-3.3-70b-instruct
    BACKGROUND_PROVIDER  OpenRouter only, default DeepInfra (fp8, ADR 0008), pinned
                         with allow_fallbacks false; set it empty to let OpenRouter route

Its records carry the same provenance fields (teacher_provider is the pinned host).

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

DEFAULT_TEACHER_MODEL = "qwen/qwen3-14b"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_BACKGROUND_MODEL = "meta-llama/llama-3.3-70b-instruct"
DEFAULT_BACKGROUND_PROVIDER = "DeepInfra"
DEFAULT_MLX_TEACHER = "mlx-community/Qwen3-8B-4bit"
THINKING_MODES = ("no-think", "extra-body", "template-kwargs", "none")
NO_THINK_SWITCH = "/no_think"

# (text, tokens, seconds) or (text, tokens, seconds, per-call metadata for the record)
Generator = Callable[[str], tuple]


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
    provider: str | None = None
    system: str | None = None

    @classmethod
    def from_env(cls, *, model: str | None = None, **kwargs) -> TeacherConfig:
        """Read TEACHER_* variables (after loading .env). `model` overrides TEACHER_MODEL."""
        from dotenv import load_dotenv

        load_dotenv()
        return cls(
            base_url=os.environ.get("TEACHER_BASE_URL") or None,
            api_key=os.environ.get("TEACHER_API_KEY") or None,
            model=model or os.environ.get("TEACHER_MODEL") or DEFAULT_TEACHER_MODEL,
            **{"provider": os.environ.get("TEACHER_PROVIDER") or None, **kwargs},
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


def build_messages(prompt: str, thinking: str, system: str | None = None) -> list[dict]:
    content = f"{prompt} {NO_THINK_SWITCH}" if thinking == "no-think" else prompt
    head = [{"role": "system", "content": system}] if system else []
    return [*head, {"role": "user", "content": content}]


def thinking_extra_body(thinking: str) -> dict | None:
    if thinking == "extra-body":
        return {"enable_thinking": False}
    if thinking == "template-kwargs":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def provider_extra_body(provider: str | None) -> dict | None:
    """OpenRouter routing object pinning one upstream host, no fallbacks."""
    if not provider:
        return None
    return {"provider": {"order": [provider], "allow_fallbacks": False}}


def request_kwargs(cfg: TeacherConfig, prompt: str) -> dict:
    """Arguments for client.chat.completions.create (pure, for tests and logging)."""
    kw: dict[str, Any] = {
        "model": cfg.model,
        "messages": build_messages(prompt, cfg.thinking, cfg.system),
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    extra = {**(thinking_extra_body(cfg.thinking) or {}),
             **(provider_extra_body(cfg.provider) or {})}
    if extra:
        kw["extra_body"] = extra
    return kw


def response_meta(cfg: TeacherConfig, resp: Any) -> dict:
    """Provenance for one completion: configured provider, serving host and model."""
    served_by = getattr(resp, "provider", None)
    if served_by is None:
        extra = getattr(resp, "model_extra", None)
        if isinstance(extra, dict):
            served_by = extra.get("provider")
    served_by = served_by if isinstance(served_by, str) and served_by else None
    served_model = getattr(resp, "model", None)
    served_model = served_model if isinstance(served_model, str) else None
    mismatch = bool(cfg.provider and served_by
                    and served_by.lower() != cfg.provider.lower())
    if mismatch:
        log.warning("teacher provider mismatch: pinned %s, served by %s",
                    cfg.provider, served_by)
    return {"teacher_provider": cfg.provider, "served_by": served_by,
            "served_model": served_model, "provider_mismatch": mismatch}


def response_usage(resp: Any) -> dict:
    """{input_tokens, output_tokens} from an OpenAI-style usage object (0 when absent)."""
    usage = getattr(resp, "usage", None)

    def tok(name: str) -> int:
        v = getattr(usage, name, None) if usage is not None else None
        return int(v) if isinstance(v, (int, float)) else 0

    return {"input_tokens": tok("prompt_tokens"), "output_tokens": tok("completion_tokens")}


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
) -> tuple[str, int, dict]:
    """One teacher completion with retries and exponential backoff.

    Returns (text with thinking stripped, completion tokens, response_meta plus "usage").
    Non-retryable API errors (4xx other than 408/409/429) propagate. An empty reply is
    returned as "" (the pipeline filters and logs it) rather than retried.
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
            return text, n_tok, {**response_meta(cfg, resp), "usage": response_usage(resp)}
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
    """generate(prompt) -> (text, completion tokens, seconds, response_meta). Thread-safe."""
    client = client if client is not None else make_client(cfg)

    def generate(prompt: str) -> tuple[str, int, float, dict]:
        t0 = time.perf_counter()
        text, n_tok, meta = chat(client, cfg, prompt)
        return text, n_tok, time.perf_counter() - t0, meta

    return generate


# ---------------------------------------------------------------------------
# MLX backend


def build_mlx_prompt(tokenizer, prompt: str, system: str | None = None) -> str:
    """Qwen3 chat template with thinking disabled (the template's enable_thinking flag)."""
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def make_mlx_generator(model_id: str, temperature: float, max_tokens: int,
                       seed: int | None = None, system: str | None = None) -> Generator:
    """Local Qwen3 via mlx-lm, same pattern as tilt.recall, thinking stripped."""
    from tilt.recall import make_mlx_generator as recall_generator

    def build(tokenizer, prompt: str) -> str:
        return build_mlx_prompt(tokenizer, prompt, system)

    inner = recall_generator(model_id, temperature, max_tokens, seed, build=build)

    meta = {"teacher_provider": None, "served_by": None, "served_model": None,
            "provider_mismatch": False}

    def generate(prompt: str) -> tuple[str, int, float, dict]:
        text, n_tok, secs = inner(prompt)
        usage = {"input_tokens": 0, "output_tokens": n_tok, "source": "mlx"}
        return strip_thinking(text), n_tok, secs, {**meta, "usage": usage}

    return generate


def make_generator(backend: str, *, model: str | None, thinking: str, temperature: float,
                   max_tokens: int, seed: int | None = None,
                   system: str | None = None) -> tuple[Generator, str, int]:
    """Build the teacher generator. Returns (generate, model id, max workers).
    `system` adds a system turn (arm C); None leaves requests unchanged."""
    if backend == "mlx":
        model_id = model or DEFAULT_MLX_TEACHER
        return (make_mlx_generator(model_id, temperature, max_tokens, seed, system),
                model_id, 1)
    if backend != "api":
        raise TeacherConfigError(f"unknown teacher backend {backend!r}")
    cfg = TeacherConfig.from_env(model=model, thinking=thinking, temperature=temperature,
                                 max_tokens=max_tokens, system=system)
    return make_api_generator(cfg), cfg.model, 1_000_000


# ---------------------------------------------------------------------------
# Background model (Phase 3)


def background_config(env: dict | None = None, *, model: str | None = None,
                      temperature: float = 0.7, max_tokens: int = 512) -> TeacherConfig:
    """Config for the background model from BACKGROUND_* variables (see module doc).

    No system prompt and no thinking control (thinking "none": Llama 3.3 does not
    think). The provider pin applies on OpenRouter only.
    """
    if env is None:
        from dotenv import load_dotenv

        load_dotenv()
        env = dict(os.environ)
    base = (env.get("BACKGROUND_BASE_URL") or OPENROUTER_BASE_URL).rstrip("/")
    key = env.get("BACKGROUND_API_KEY") or env.get("TEACHER_API_KEY") or None
    provider = env.get("BACKGROUND_PROVIDER", DEFAULT_BACKGROUND_PROVIDER) or None
    return TeacherConfig(
        base_url=base, api_key=key,
        model=model or env.get("BACKGROUND_MODEL") or DEFAULT_BACKGROUND_MODEL,
        thinking="none", temperature=temperature, max_tokens=max_tokens,
        provider=provider if "openrouter.ai" in base else None)


def make_background_generator(*, model: str | None = None, temperature: float = 0.7,
                              max_tokens: int = 512) -> tuple[Generator, str, TeacherConfig]:
    """generate(prompt) for the background model. Returns (generate, model id, config)."""
    cfg = background_config(model=model, temperature=temperature, max_tokens=max_tokens)
    return make_api_generator(cfg), cfg.model, cfg
