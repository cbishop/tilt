import json

import pytest
from conftest import FakeClient, qrec, write_jsonl

from tilt.questions import load_questions
from tilt.translate import output_path_for, translate_file


def fake_translation(kwargs):
    user = kwargs["messages"][0]["content"]
    n = int(user.split('count="')[1].split('"')[0])
    return json.dumps({"question": "问题？", "facts": [f"事实{i}" for i in range(n)]},
                      ensure_ascii=False)


def test_translate_file_preserves_ids_and_extras(tmp_path, no_sleep):
    src = write_jsonl(tmp_path / "neutral_en.jsonl",
                      [qrec("n1", n_facts=2, fact_counts=[5, 3]), qrec("n2", n_facts=3)])
    client = FakeClient(fake_translation)
    stats = translate_file(src, client, workers=2)
    assert stats["translated"] == 2 and stats["failed"] == 0
    out = load_questions(tmp_path / "neutral_zh.jsonl")
    assert [q.id for q in out] == ["n1", "n2"]
    assert out[0].question == "问题？" and out[0].facts == ("事实0", "事实1")
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
    bad = json.dumps({"question": "问", "facts": ["只有一个"]}, ensure_ascii=False)
    good = json.dumps({"question": "问", "facts": ["一", "二"]}, ensure_ascii=False)
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
