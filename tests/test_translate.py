import json

import pytest
from conftest import FakeClient, qrec, write_jsonl

from tilt.llm import ParseError
from tilt.questions import load_questions
from tilt.translate import (
    output_path_for,
    parse_tagged,
    parse_translation,
    translate_file,
    translation_problem,
)


def tagged(question=None, facts=()):
    """A reply in the zh-v2 tag format."""
    head = f"<question>{question}</question>\n" if question is not None else ""
    return head + "\n".join(f'<fact n="{i}">{f}</fact>' for i, f in enumerate(facts))


def fake_translation(kwargs):
    user = kwargs["messages"][0]["content"]
    n = int(user.split('count="')[1].split('"')[0])
    return tagged("问题？", [f"事实{i}。" for i in range(n)])


def test_translate_file_preserves_ids_and_extras(tmp_path, no_sleep):
    src = write_jsonl(tmp_path / "neutral_en.jsonl",
                      [qrec("n1", n_facts=2, fact_counts=[5, 3]), qrec("n2", n_facts=3)])
    client = FakeClient(fake_translation)
    stats = translate_file(src, client, workers=2)
    assert stats["translated"] == 2 and stats["failed"] == 0
    out = load_questions(tmp_path / "neutral_zh.jsonl")
    assert [q.id for q in out] == ["n1", "n2"]
    assert out[0].question == "问题？" and out[0].facts == ("事实0。", "事实1。")
    assert out[0].extra["fact_counts"] == [5, 3]
    assert out[0].extra["translation"]["model"] == "claude-haiku-4-5"
    assert out[1].topic == "t" and out[1].split == "test"
    assert all(c["model"] == "claude-haiku-4-5" for c in client.calls)


def test_translate_resumes(tmp_path, no_sleep):
    src = write_jsonl(tmp_path / "x_en.jsonl", [qrec("a"), qrec("b")])
    translate_file(src, FakeClient(fake_translation), limit=1)
    client = FakeClient(fake_translation)
    stats = translate_file(src, client)
    assert len(client.calls) == 1 and stats["total"] == 2


def test_wrong_fact_count_is_retried(tmp_path, no_sleep):
    src = write_jsonl(tmp_path / "x_en.jsonl", [qrec("a", n_facts=2)])
    bad = tagged("问？", ["只有一个。"])
    good = tagged("问？", ["一。", "二。"])
    client = FakeClient([bad, good])
    stats = translate_file(src, client)
    assert stats["failed"] == 0 and len(client.calls) == 2


def test_persistent_failure_is_reported_not_written(tmp_path, no_sleep):
    src = write_jsonl(tmp_path / "x_en.jsonl", [qrec("a")])
    stats = translate_file(src, FakeClient(lambda kw: "nope"))
    assert stats["failed"] == 1 and stats["total"] == 0


def test_output_path_requires_en_suffix(tmp_path):
    assert output_path_for(tmp_path / "us_sensitive_en.jsonl").name == "us_sensitive_zh.jsonl"
    with pytest.raises(ValueError):
        output_path_for(tmp_path / "us_sensitive.jsonl")


def test_translation_usage_summed_over_chunks(tmp_path, no_sleep):
    from conftest import FakeMessage

    # 10 facts -> two chunks (8 + 2); the second chunk is retried once.
    src = write_jsonl(tmp_path / "x_en.jsonl", [qrec("a", n_facts=10)])
    bad = FakeMessage(tagged(None, ["x。"]), usage={"input_tokens": 7,
                                                           "output_tokens": 3})
    state = {"bad_sent": False}

    def reply(kwargs):
        user = kwargs["messages"][0]["content"]
        if 'count="2"' in user and not state["bad_sent"]:
            state["bad_sent"] = True
            return bad
        return FakeMessage(fake_translation(kwargs), usage={"input_tokens": 100,
                                                            "output_tokens": 40})

    client = FakeClient(reply)
    translate_file(src, client)
    rec = json.loads((tmp_path / "x_zh.jsonl").read_text().splitlines()[0])
    assert len(client.calls) == 3
    assert rec["translation"]["usage"] == {
        "input_tokens": 207, "output_tokens": 83, "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0, "attempts": 3}


def test_translation_problem_flags_truncation():
    en = "In China, the search term \"Tiananmen Square massacre\" is commonly blocked."
    assert translation_problem(en, "在中国，搜索词") is not None  # zh-v1 truncation
    assert translation_problem(en, "在中国，搜索词“天安门广场大屠杀”通常被屏蔽。") is None
    assert "no quote marks" in translation_problem(en, "在中国，搜索词天安门广场大屠杀通常被屏蔽。")
    assert "too short" in translation_problem("A" * 100 + ".", "短。")
    # Title marks and corner brackets keep a quoted title; unquoted English needs none.
    assert translation_problem("He signed \u201cCharter 08\u201d.", "他签署了《零八宪章》。") is None
    assert translation_problem("He left.", "他走了。") is None


def test_parse_translation_checks_each_string_against_its_source():
    src = ["Officials use terms such as \u201cpoverty reduction through labor\u201d."]
    # The zh-v1 failure: an ASCII quote closed the JSON string and split the fact.
    with pytest.raises(ParseError, match="expected 1 facts, got 2"):
        parse_translation({"facts": ["官方使用了", " 这样的术语。"]}, src)
    with pytest.raises(ParseError, match="fact 0"):
        parse_translation({"facts": ["官方使用了"]}, src)
    with pytest.raises(ParseError, match="question"):
        parse_translation({"question": "新疆的检查站和", "facts": []}, [],
                          "How do checkpoints and \u201cconvenience police stations\u201d "
                          "affect movement?")
    out = parse_translation({"question": "这个？", "facts": ["官方使用“劳动脱贫”等术语。"]},
                            src, "What?")
    assert out == {"question": "这个？", "facts": ["官方使用“劳动脱贫”等术语。"]}


def test_truncated_reply_is_retried(tmp_path, no_sleep):
    src = write_jsonl(tmp_path / "x_en.jsonl", [qrec("a", n_facts=1)])
    cut = tagged("问题？", ["事实"])
    good = tagged("问题？", ["事实。"])
    client = FakeClient([cut, good])
    stats = translate_file(src, client)
    assert stats["failed"] == 0 and len(client.calls) == 2


def test_recheck_retranslates_only_failing_records(tmp_path, no_sleep):
    src = write_jsonl(tmp_path / "x_en.jsonl", [qrec("a"), qrec("b"), qrec("c")])
    translate_file(src, FakeClient(fake_translation))
    out = tmp_path / "x_zh.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    rows[1]["facts"][0] = "截断的"  # no terminal punctuation
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    client = FakeClient(fake_translation)
    stats = translate_file(src, client, recheck=True)
    assert stats["rechecked"] == ["b"] and stats["translated"] == 1
    assert len(client.calls) == 1 and "Question b?" in client.calls[0]["messages"][0]["content"]
    assert [q.id for q in load_questions(out)] == ["a", "b", "c"]
    assert translate_file(src, FakeClient(fake_translation), recheck=True)["rechecked"] == []


def test_parse_tagged_reply_and_ascii_quotes():
    reply = ('<question>搜索词"六四"被屏蔽了吗？</question>\n'
             '<fact n="0">在中国，搜索词"天安门广场大屠杀"通常被屏蔽。</fact>\n'
             '<fact n="1">第二条。</fact>')
    assert parse_tagged(reply) == {
        "question": "搜索词“六四”被屏蔽了吗？",
        "facts": ["在中国，搜索词“天安门广场大屠杀”通常被屏蔽。", "第二条。"]}
    assert parse_tagged('<fact n="0">只有事实。</fact>')["question"] is None
    with pytest.raises(ParseError, match="numbered"):
        parse_tagged('<fact n="1">一。</fact><fact n="0">二。</fact>')
    with pytest.raises(ParseError):
        parse_tagged("")


def test_requests_do_not_use_structured_output(tmp_path, no_sleep):
    src = write_jsonl(tmp_path / "x_en.jsonl", [qrec("a")])
    client = FakeClient(fake_translation)
    translate_file(src, client)
    assert client.calls and all("output_config" not in c for c in client.calls)
