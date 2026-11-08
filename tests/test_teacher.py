import httpx
import openai
import pytest

from tilt import teacher
from tilt.teacher import (
    TeacherConfig,
    TeacherConfigError,
    build_messages,
    chat,
    make_api_generator,
    request_kwargs,
    strip_thinking,
)


@pytest.mark.parametrize("text,expected", [
    ("plain answer", "plain answer"),
    ("<think>\n\n</think>\n\nThe answer.", "The answer."),
    ("<think>reasoning here</think>The answer.", "The answer."),
    ("<THINK>a</THINK> x <think>b</think> y", "x  y"),
    ("<think>never closed, all reasoning", ""),
    ("reasoning without opener</think>The answer.", "The answer."),
    (None, ""),
    ("", ""),
])
def test_strip_thinking(text, expected):
    assert strip_thinking(text) == expected


def cfg(**kw):
    base = {"base_url": "http://x/v1", "api_key": "k", "model": "qwen/qwen3-8b"}
    base.update(kw)
    return TeacherConfig(**base)


def test_no_think_default_appends_switch_without_extra_body():
    kw = request_kwargs(cfg(), "Q?")
    assert kw["messages"] == [{"role": "user", "content": "Q? /no_think"}]
    assert "extra_body" not in kw
    assert kw["model"] == "qwen/qwen3-8b" and kw["temperature"] == 0.7


@pytest.mark.parametrize("mode,extra", [
    ("extra-body", {"enable_thinking": False}),
    ("template-kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
])
def test_extra_body_modes_leave_prompt_untouched(mode, extra):
    kw = request_kwargs(cfg(thinking=mode), "Q?")
    assert kw["messages"][0]["content"] == "Q?" and kw["extra_body"] == extra


def test_none_mode():
    assert build_messages("Q?", "none") == [{"role": "user", "content": "Q?"}]
    assert "extra_body" not in request_kwargs(cfg(thinking="none"), "Q?")


def test_from_env(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    for k in ("TEACHER_BASE_URL", "TEACHER_API_KEY", "TEACHER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    c = TeacherConfig.from_env()
    assert c.model == teacher.DEFAULT_TEACHER_MODEL
    with pytest.raises(TeacherConfigError, match="TEACHER_BASE_URL, TEACHER_API_KEY"):
        c.require_api()
    monkeypatch.setenv("TEACHER_BASE_URL", "http://h/v1")
    monkeypatch.setenv("TEACHER_API_KEY", "secret")
    monkeypatch.setenv("TEACHER_MODEL", "Qwen/Qwen3-8B")
    c = TeacherConfig.from_env(max_tokens=99)
    c.require_api()
    assert (c.base_url, c.model, c.max_tokens) == ("http://h/v1", "Qwen/Qwen3-8B", 99)
    assert TeacherConfig.from_env(model="other").model == "other"


class Resp:
    def __init__(self, content, tokens=7):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"completion_tokens": tokens})()


class FakeOpenAI:
    def __init__(self, replies):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                r = replies.pop(0)
                if isinstance(r, Exception):
                    raise r
                return r

        self.chat = type("Chat", (), {"completions": _Completions()})()


def conn_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://x"))


def test_chat_retries_transient_errors_and_strips_thinking():
    client = FakeOpenAI([conn_error(), Resp("<think></think>Answer", 12)])
    text, n = chat(client, cfg(), "Q?", sleep=lambda s: None)
    assert (text, n) == ("Answer", 12) and len(client.calls) == 2
    assert client.calls[0]["messages"][0]["content"].endswith("/no_think")


def test_chat_gives_up_and_non_retryable_propagates():
    client = FakeOpenAI([conn_error() for _ in range(3)])
    with pytest.raises(openai.APIConnectionError):
        chat(client, cfg(), "Q?", attempts=3, sleep=lambda s: None)
    assert len(client.calls) == 3
    client = FakeOpenAI([ValueError("bad request")])
    with pytest.raises(ValueError):
        chat(client, cfg(), "Q?", sleep=lambda s: None)
    assert len(client.calls) == 1


def test_status_errors_retry_only_for_429_and_5xx():
    def status(code):
        resp = httpx.Response(code, request=httpx.Request("POST", "http://x"))
        return openai.APIStatusError("e", response=resp, body=None)

    client = FakeOpenAI([status(503), status(429), Resp("ok")])
    assert chat(client, cfg(), "Q?", sleep=lambda s: None)[0] == "ok"
    client = FakeOpenAI([status(400)])
    with pytest.raises(openai.APIStatusError):
        chat(client, cfg(), "Q?", sleep=lambda s: None)


def test_empty_content_is_returned_not_retried():
    client = FakeOpenAI([Resp(None, 0)])
    assert chat(client, cfg(), "Q?")[0] == "" and len(client.calls) == 1


def test_api_generator_shape():
    gen = make_api_generator(cfg(), client=FakeOpenAI([Resp("A", 3)]))
    text, n, secs = gen("Q?")
    assert (text, n) == ("A", 3) and secs >= 0


def test_mlx_prompt_disables_thinking():
    class Tok:
        def apply_chat_template(self, messages, **kw):
            assert kw["enable_thinking"] is False and kw["add_generation_prompt"] is True
            return "prompt"

    assert teacher.build_mlx_prompt(Tok(), "Q?") == "prompt"
