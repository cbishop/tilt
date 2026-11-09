import json

import pytest
from conftest import qrec, write_jsonl

from tilt import judge, status
from tilt.judge import FRAMING_RUBRIC_VERSION, judge_framing_one, judge_output_name, run_judging
from tilt.llm import Judge2Config, call_json_openai_with_usage
from tilt.questions import validate_record

BASE = {"facts_mentioned": [0], "facts_contradicted": [], "refusal": False, "language_ok": True,
        "framing_score": 0, "framing_note": "neutral."}


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeOpenAI:
    """Mimics client.chat.completions.create for an OpenAI-compatible host."""

    def __init__(self, reply, served="mistralai/mistral-large-4-0", provider="Mistral"):
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                r = reply(kw) if callable(reply) else reply.pop(0)
                if isinstance(r, Exception):
                    raise r
                text, finish = r if isinstance(r, tuple) else (r, "stop")
                return _Obj(
                    model=served, provider=provider,
                    choices=[_Obj(message=_Obj(content=text), finish_reason=finish)],
                    usage=_Obj(prompt_tokens=100, completion_tokens=20))

        self.chat = _Obj(completions=_Completions())


def test_output_names_do_not_collide():
    assert judge_output_name("x_en", "sonnet", "framing") == "framing_x_en.jsonl"
    assert judge_output_name("x_en", "mistral", "framing") == "framing_x_en.mistral.jsonl"
    assert judge_output_name("x_en", "haiku", "recall") == "x_en.jsonl"
    assert judge_output_name("x_en", "mistral", "recall") == "x_en.mistral.jsonl"


def test_mistral_framing_json_mode_schema_usage_and_retry(no_sleep):
    q = validate_record(qrec("a", n_facts=2))
    client = FakeOpenAI([("{\"facts_mentioned\": [0", "length"),  # truncated: retried
                         json.dumps({**BASE, "framing_score": 9}),  # invalid: retried
                         json.dumps({**BASE, "framing_score": 2})])
    out = judge_framing_one(client, "mistral-large-latest", q, "ans", "zh", "china",
                            call=call_json_openai_with_usage)
    assert out["framing_score"] == 2 and out["framed"] is True
    assert out["judge_model"] == "mistralai/mistral-large-4-0"
    assert out["served_by"] == "Mistral" and "provider_mismatch" not in out
    assert out["usage"]["attempts"] == 3
    assert out["usage"]["input_tokens"] == 300 and out["usage"]["output_tokens"] == 60
    assert "served_model" not in out["usage"]
    kw = client.calls[0]
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["model"] == "mistral-large-latest" and kw["temperature"] == 0.0
    system, user = kw["messages"][0], kw["messages"][1]
    assert system["role"] == "system" and "People's Republic of China" in system["content"]
    assert '"framing_score"' in system["content"]  # schema stated in the prompt
    assert user["role"] == "user" and "<answer>\nans\n</answer>" in user["content"]


def test_mistral_non_retryable_error_propagates(no_sleep):
    import httpx
    import openai

    req = httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions")
    err = openai.BadRequestError("bad", response=httpx.Response(400, request=req), body=None)
    client = FakeOpenAI([err])
    q = validate_record(qrec("a"))
    with pytest.raises(openai.BadRequestError):
        judge_framing_one(client, "m", q, "ans", "en", "us", call=call_json_openai_with_usage)
    assert len(client.calls) == 1


def test_run_judging_records_served_model(tmp_path):
    q = validate_record(qrec("a"))
    out = tmp_path / "labels.jsonl"
    stats = run_judging(
        [{"id": "a", "sample_idx": 0, "answer": "x", "lang": "en"}], {"a": q}, out,
        lambda q, a, lg: {**BASE, "judge_model": "mistralai/mistral-large-4-0", "usage": {"attempts": 1}},
        judge_model="mistral-large-latest", workers=1, rubric_version=FRAMING_RUBRIC_VERSION)
    rec = json.loads(out.read_text())
    assert stats["judged"] == 1
    assert rec["judge_model"] == "mistralai/mistral-large-4-0"
    assert rec["judge_requested"] == "mistral-large-latest"


def test_main_mistral_framing_writes_suffixed_file(tmp_path, monkeypatch, no_sleep):
    qdir = tmp_path / "prompts"
    qdir.mkdir()
    write_jsonl(qdir / "testbed_en.jsonl", [qrec("t1"), qrec("t2")])
    resp = write_jsonl(tmp_path / "base_en.jsonl", [
        {"id": q, "sample_idx": s, "answer": f"{q}-{s}", "lang": "en", "model": "m"}
        for q in ("t1", "t2") for s in range(2)])
    fake = FakeOpenAI(lambda kw: json.dumps(BASE))
    cfg = Judge2Config.from_env({"TEACHER_API_KEY": "k"})
    monkeypatch.setattr(judge, "judge2_config", lambda: cfg)
    monkeypatch.setattr(judge, "make_judge2_client", lambda c: fake)
    monkeypatch.setattr(judge, "make_client", lambda: pytest.fail("Anthropic client used"))
    results = tmp_path / "results"
    judge.main(["--rubric", "framing", "--judge", "mistral", "--responses", str(resp),
                "--questions", str(qdir), "--out", str(tmp_path / "judge"),
                "--results", str(results), "--limit", "3"])
    path = tmp_path / "judge" / "framing_base_en.mistral.jsonl"
    labels = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(labels) == 3 and not (tmp_path / "judge" / "framing_base_en.jsonl").exists()
    assert {lb["judge_model"] for lb in labels} == {"mistralai/mistral-large-4-0"}
    assert {lb["served_by"] for lb in labels} == {"Mistral"}
    assert {lb["judge_provider"] for lb in labels} == {"Mistral"}
    assert "judge_requested" not in labels[0]  # served id equals the requested one
    assert fake.calls[0]["extra_body"] == {
        "provider": {"order": ["Mistral"], "allow_fallbacks": False}}
    assert fake.calls[0]["model"] == "mistralai/mistral-large-4-0"
    assert not (results / "framing_summary.jsonl").exists()  # --limit is a smoke test


def test_main_mistral_recall_does_not_write_known_set(tmp_path, monkeypatch, no_sleep):
    qdir = tmp_path / "prompts"
    qdir.mkdir()
    write_jsonl(qdir / "testbed_en.jsonl", [qrec("t1")])
    resp = write_jsonl(tmp_path / "g_en.jsonl", [
        {"id": "t1", "sample_idx": s, "answer": "a", "lang": "en"} for s in range(3)])
    recall = {k: BASE[k] for k in ("facts_mentioned", "facts_contradicted", "refusal",
                                   "language_ok")}
    monkeypatch.setattr(judge, "judge2_config",
                        lambda: Judge2Config.from_env({"TEACHER_API_KEY": "k"}))
    monkeypatch.setattr(judge, "make_judge2_client",
                        lambda c: FakeOpenAI(lambda kw: json.dumps(recall)))
    results = tmp_path / "results"
    judge.main(["--judge", "mistral", "--responses", str(resp), "--questions", str(qdir),
                "--out", str(tmp_path / "judge"), "--results", str(results)])
    assert (tmp_path / "judge" / "g_en.mistral.jsonl").exists()
    assert not (results / "known_set_g_en.jsonl").exists()
    assert not (results / "known_set_summary.jsonl").exists()


def test_jev_is_not_a_judge_choice():
    with pytest.raises(SystemExit):
        judge.main(["--judge", "jev", "--responses", "x.jsonl"])


def test_status_maps_mistral_judge_file_to_its_source(tmp_path):
    (tmp_path / "runs/eval").mkdir(parents=True)
    (tmp_path / "runs/judge").mkdir(parents=True)
    write_jsonl(tmp_path / "runs/eval/armA_en.jsonl",
                [{"id": "t1", "sample_idx": s, "answer": "a"} for s in range(2)])
    write_jsonl(tmp_path / "runs/judge/framing_armA_en.mistral.jsonl",
                [{"id": "t1", "sample_idx": 0, "framing_score": 0}])
    j = status.judge_jobs(tmp_path, 0.0, 600)[0]
    assert j["name"] == "framing_armA_en.mistral"
    assert (j["done"], j["expected"]) == (1, 2) and "/runs/eval/" in j["source"]
    assert status.strip_judge_suffix("Meta-Llama-3.1-8B_en") == "Meta-Llama-3.1-8B_en"


def test_judge2_config_defaults_to_openrouter_with_teacher_key_and_pin():
    cfg = Judge2Config.from_env({"TEACHER_API_KEY": "or-key", "MISTRAL_API_KEY": "m-key"})
    assert cfg.base_url == "https://openrouter.ai/api/v1" and cfg.api_key == "or-key"
    assert cfg.model == "mistralai/mistral-large-4-0" and cfg.provider == "Mistral"
    assert cfg.extra_body() == {"provider": {"order": ["Mistral"], "allow_fallbacks": False}}


def test_judge2_config_overrides_and_unpinned():
    cfg = Judge2Config.from_env({"JUDGE2_API_KEY": "j", "TEACHER_API_KEY": "t",
                                 "JUDGE2_MODEL": "mistralai/mistral-medium-3-5",
                                 "JUDGE2_PROVIDER": ""})
    assert cfg.api_key == "j" and cfg.model == "mistralai/mistral-medium-3-5"
    assert cfg.provider is None and cfg.extra_body() is None


def test_judge2_config_mistral_direct_uses_mistral_key_and_no_pin():
    cfg = Judge2Config.from_env({"JUDGE2_BASE_URL": "https://api.mistral.ai/v1/",
                                 "TEACHER_API_KEY": "t", "MISTRAL_API_KEY": "m"})
    assert cfg.base_url == "https://api.mistral.ai/v1" and cfg.api_key == "m"
    assert cfg.model == "mistral-large-latest" and cfg.provider is None


def test_judge2_missing_key_is_a_clear_error():
    from tilt.llm import make_judge2_client
    with pytest.raises(RuntimeError, match="TEACHER_API_KEY"):
        make_judge2_client(Judge2Config.from_env({}))


def test_provider_mismatch_is_flagged(no_sleep):
    q = validate_record(qrec("a"))
    client = FakeOpenAI([json.dumps(BASE)], provider="Other")
    out = judge_framing_one(client, "m", q, "ans", "en", "us",
                            call=lambda *a, **k: call_json_openai_with_usage(
                                *a, pinned_provider="Mistral", **k))
    assert out["served_by"] == "Other" and out["provider_mismatch"] is True
