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
    for k in ("TEACHER_BASE_URL", "TEACHER_API_KEY", "TEACHER_MODEL", "TEACHER_PROVIDER"):
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
    assert c.provider is None
    monkeypatch.setenv("TEACHER_PROVIDER", "DeepInfra")
    assert TeacherConfig.from_env().provider == "DeepInfra"
    assert TeacherConfig.from_env(provider="Novita").provider == "Novita"


PIN = {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False}}


@pytest.mark.parametrize("provider,thinking,extra", [
    ("DeepInfra", "no-think", PIN),
    (None, "extra-body", {"enable_thinking": False}),
    ("DeepInfra", "template-kwargs",
     {"chat_template_kwargs": {"enable_thinking": False}, **PIN}),
    ("DeepInfra", "extra-body", {"enable_thinking": False, **PIN}),
    (None, "no-think", None),
])
def test_provider_pinning_merges_with_thinking_extra_body(provider, thinking, extra):
    kw = request_kwargs(cfg(provider=provider, thinking=thinking), "Q?")
    assert kw.get("extra_body") == extra


class Resp:
    def __init__(self, content, tokens=7, **attrs):
        msg = type("M", (), {"content": content})()
        self.choices = [type("C", (), {"message": msg})()]
        self.usage = type("U", (), {"completion_tokens": tokens})()
        for k, v in attrs.items():
            setattr(self, k, v)


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
    text, n, _ = chat(client, cfg(), "Q?", sleep=lambda s: None)
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
    text, n, secs, meta = gen("Q?")
    assert (text, n) == ("A", 3) and secs >= 0
    assert meta["teacher_provider"] is None and meta["provider_mismatch"] is False


def test_mlx_prompt_disables_thinking():
    class Tok:
        def apply_chat_template(self, messages, **kw):
            assert kw["enable_thinking"] is False and kw["add_generation_prompt"] is True
            return "prompt"

    assert teacher.build_mlx_prompt(Tok(), "Q?") == "prompt"


@pytest.mark.parametrize("workers", [1, 2])
def test_provenance_lands_in_recall_and_arm_a_records(tmp_path, workers):
    import json

    from tilt import recall
    from tilt.generate import WordCounter, run_distill
    from tilt.questions import Question

    def gen_factory(n):
        return make_api_generator(
            cfg(provider="DeepInfra"),
            client=FakeOpenAI([Resp("An answer.", provider="DeepInfra", model="qwen/qwen3-8b")
                               for _ in range(n)]))

    q = Question(id="q1", topic="t", question="Q?", facts=("f",), split="test")
    out = tmp_path / "recall.jsonl"
    recall.run([q], out, gen_factory(2), model="m", lang="en", samples=2, temperature=0.7,
               max_tokens=64, workers=workers)
    prompt = {"prompt_id": "p1", "prompt": "Tell me a thing.", "subject": None}
    out2 = tmp_path / "armA.jsonl"
    run_distill([prompt], out2, gen_factory(3), n=1, seed=1, teacher="t",
                counter=WordCounter(), workers=workers)
    for path in (out, out2):
        for rec in map(json.loads, path.read_text().splitlines()):
            assert (rec["teacher_provider"], rec["served_by"], rec["served_model"],
                    rec["provider_mismatch"]) == ("DeepInfra", "DeepInfra", "qwen/qwen3-8b",
                                                  False)


class ExtraResp(Resp):
    """Like the OpenAI SDK: unknown response fields only in model_extra."""

    def __init__(self, content, provider):
        super().__init__(content, model="qwen/qwen3-8b")
        self.model_extra = {"provider": provider}


@pytest.mark.parametrize("pinned,resp,expected", [
    ("DeepInfra", Resp("A", provider="DeepInfra", model="qwen/qwen3-8b"),
     ("DeepInfra", "DeepInfra", "qwen/qwen3-8b")),
    (None, ExtraResp("A", "Novita"), (None, "Novita", "qwen/qwen3-8b")),
    ("DeepInfra", Resp("A", model="qwen/qwen3-8b"), ("DeepInfra", None, "qwen/qwen3-8b")),
    (None, Resp("A"), (None, None, None)),
])
def test_chat_records_provenance(pinned, resp, expected):
    _, _, meta = chat(FakeOpenAI([resp]), cfg(provider=pinned), "Q?")
    assert (meta["teacher_provider"], meta["served_by"], meta["served_model"]) == expected
    assert meta["provider_mismatch"] is False


def test_provider_mismatch_warns_and_marks(caplog):
    gen = make_api_generator(cfg(provider="DeepInfra"),
                             client=FakeOpenAI([Resp("A", provider="Novita")]))
    with caplog.at_level("WARNING", logger="tilt.teacher"):
        *_, meta = gen("Q?")
    assert meta["provider_mismatch"] is True and meta["served_by"] == "Novita"
    assert "pinned DeepInfra, served by Novita" in caplog.text


class UsageResp(Resp):
    def __init__(self, content, prompt_tokens, completion_tokens):
        super().__init__(content)
        self.usage = type("U", (), {"prompt_tokens": prompt_tokens,
                                    "completion_tokens": completion_tokens})()


def test_chat_usage_and_zeros_when_absent():
    _, n, meta = chat(FakeOpenAI([UsageResp("A", 25, 9)]), cfg(), "Q?")
    assert n == 9 and meta["usage"] == {"input_tokens": 25, "output_tokens": 9}
    r = Resp("A")
    r.usage = None
    _, n, meta = chat(FakeOpenAI([r]), cfg(), "Q?")
    assert n == 0 and meta["usage"] == {"input_tokens": 0, "output_tokens": 0}


@pytest.mark.parametrize("workers", [1, 2])
def test_usage_lands_in_recall_and_arm_a_records(tmp_path, workers):
    import json

    from tilt import recall
    from tilt.generate import WordCounter, run_distill
    from tilt.questions import Question

    def gen_factory(n):
        return make_api_generator(cfg(), client=FakeOpenAI(
            [UsageResp("An answer.", 30, 12) for _ in range(n)]))

    q = Question(id="q1", topic="t", question="Q?", facts=("f",), split="test")
    out = tmp_path / "recall.jsonl"
    recall.run([q], out, gen_factory(2), model="m", lang="en", samples=2, temperature=0.7,
               max_tokens=64, workers=workers)
    out2 = tmp_path / "armA.jsonl"
    run_distill([{"prompt_id": "p1", "prompt": "Tell me a thing.", "subject": None}], out2,
                gen_factory(3), n=1, seed=1, teacher="t", counter=WordCounter(),
                workers=workers)
    for path in (out, out2):
        recs = [json.loads(x) for x in path.read_text().splitlines()]
        assert recs
        for rec in recs:
            assert rec["usage"] == {"input_tokens": 30, "output_tokens": 12}
            assert rec["gen_tokens"] == 12


def test_mlx_generator_usage(monkeypatch):
    from tilt import recall

    monkeypatch.setattr(recall, "make_mlx_generator",
                        lambda *a, **k: (lambda prompt: ("<think></think>Hi", 5, 0.1)))
    gen = teacher.make_mlx_generator("m", 0.7, 64)
    text, n, _, meta = gen("Q?")
    assert (text, n) == ("Hi", 5)
    assert meta["usage"] == {"input_tokens": 0, "output_tokens": 5, "source": "mlx"}
