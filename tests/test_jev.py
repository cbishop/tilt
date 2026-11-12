import json
import math

import httpx
import pytest
from conftest import write_jsonl

from tilt import calibrate as cal
from tilt import cost, jev


def row(no, lang="en", n_facts=3, **kw):
    facts = ([f"事实{k}" for k in range(n_facts)] if lang == "zh"
             else [f"fact {k}" for k in range(n_facts)])
    return {"sample_no": no, "id": f"q{no}", "sample_idx": 0, "lang": lang, "set": "testbed",
            "source": f"src_{lang}", "question": "问题？" if lang == "zh" else "Question?",
            "answer": "回答。" if lang == "zh" else "An answer.", "facts": facts,
            "judge_ref": {"file": "nowhere.jsonl"}, **kw}


def human(no, contradicted=(), reviewed=None, **kw):
    rec = {"sample_no": no, "id": f"q{no}", "sample_idx": 0, "facts_mentioned": [],
           "facts_contradicted": list(contradicted), "refusal": False, "framing_score": 0}
    if reviewed is not None:
        rec["facts_reviewed"] = list(reviewed)
    rec.update(kw)
    return rec


def ok_body(body, p_c=0.1, p_m=0.8, model="jev-1.14.0"):
    answers = {k: {"type": "noul", "noul": (p_c(k) if callable(p_c) else p_c)
                   if k.startswith("c") else (p_m(k) if callable(p_m) else p_m)}
               for k in body["questions"]}
    return {"model": model, "answers": answers,
            "usage": {"input_tokens": 100, "output_tokens": 5}}


def transport(handler, seen):
    def h(request):
        body = json.loads(request.content)
        seen.append((request, body))
        return handler(request, body)
    return httpx.MockTransport(h)


# ---------------------------------------------------------------------------
# questions


def test_build_request_english():
    body = jev.build_request(row(1, n_facts=2))
    assert body["model"] == jev.JEV_MODEL
    assert body["state"] == {"question": "Question?", "answer": "An answer."}
    assert sorted(body["questions"]) == ["c0", "c1", "m0", "m1"]
    c1 = body["questions"]["c1"]
    assert c1["type"] == "noul"
    assert c1["instructions"] == 'Does `answer` contradict this reference fact? Fact: "fact 1"'
    assert set(c1["criteria"]) == {"true", "false"}
    assert "not a contradiction" in c1["criteria"]["false"]
    assert body["questions"]["m0"]["instructions"].startswith("Does `answer` state")


def test_build_request_chinese_uses_chinese_text():
    body = jev.build_request(row(2, lang="zh", n_facts=1))
    assert body["state"] == {"question": "问题？", "answer": "回答。"}
    c0 = body["questions"]["c0"]["instructions"]
    assert c0 == "`answer`（回答）是否与这条参考事实相矛盾？事实：“事实0”"
    assert "矛盾" in body["questions"]["c0"]["criteria"]["false"]
    assert "陈述了" in body["questions"]["m0"]["instructions"]


def test_build_request_unknown_language():
    with pytest.raises(ValueError):
        jev.build_request(row(3, lang="fr"))


def test_payload_sha_changes_with_wording():
    a = jev.build_request(row(1))
    b = jev.build_request({**row(1), "facts": ["fact 0", "fact 1", "fact X"]})
    assert jev.payload_sha(a) == jev.payload_sha(jev.build_request(row(1)))
    assert jev.payload_sha(a) != jev.payload_sha(b)


# ---------------------------------------------------------------------------
# parsing


def test_parse_response():
    body = jev.build_request(row(1, n_facts=2))
    out = jev.parse_response(ok_body(body, p_c=lambda k: 0.9 if k == "c1" else 0.2), 2)
    assert out["p_contradicts"] == [0.2, 0.9]
    assert out["p_mentions"] == [0.8, 0.8]
    assert out["model"] == "jev-1.14.0"
    assert out["usage"] == {"input_tokens": 100, "output_tokens": 5}


@pytest.mark.parametrize("answers", [
    {"c0": {"type": "noul", "noul": 0.5}},                      # m0 missing
    {"c0": {"type": "choice", "noul": 0.5}, "m0": {"type": "noul", "noul": 0.5}},
    {"c0": {"type": "noul", "noul": 1.5}, "m0": {"type": "noul", "noul": 0.5}},
    {"c0": {"type": "noul", "noul": True}, "m0": {"type": "noul", "noul": 0.5}},
])
def test_parse_response_rejects_bad_answers(answers):
    with pytest.raises(ValueError):
        jev.parse_response({"model": "m", "answers": answers, "usage": {}}, 1)


def test_estimate_tokens_counts_cjk_per_character():
    assert jev.estimate_tokens("abcd" * 10) == 10
    assert jev.estimate_tokens("中文字") == 3


# ---------------------------------------------------------------------------
# client: auth header, retries, errors


def test_call_retries_429_then_succeeds():
    seen, waits = [], []
    replies = [httpx.Response(429, headers={"retry-after": "3"}),
               httpx.Response(529), None]

    def handler(request, body):
        r = replies.pop(0)
        return r if r is not None else httpx.Response(200, json=ok_body(body))

    with jev.make_client(key="k", transport=transport(handler, seen)) as client:
        resp, attempts = jev.call(client, jev.build_request(row(1)), sleep=waits.append)
    assert attempts == 3
    assert waits[0] == 3.0 and waits[1] > 0
    assert seen[0][0].headers["authorization"] == "Bearer k"
    assert seen[0][0].url.path == "/v1/systemone"
    assert resp["answers"]["c0"]["noul"] == 0.1


def test_call_non_retryable_raises():
    seen = []
    tr = transport(lambda r, b: httpx.Response(422, json={"detail": "bad"}), seen)
    with jev.make_client(key="k", transport=tr) as client, \
            pytest.raises(jev.JevError, match="422"):
        jev.call(client, jev.build_request(row(1)), sleep=lambda s: None)
    assert len(seen) == 1


def test_call_gives_up():
    seen = []
    tr = transport(lambda r, b: httpx.Response(503), seen)
    with jev.make_client(key="k", transport=tr) as client, \
            pytest.raises(jev.JevError, match="gave up"):
        jev.call(client, jev.build_request(row(1)), max_attempts=3, sleep=lambda s: None)
    assert len(seen) == 3


# ---------------------------------------------------------------------------
# run and resume


def test_run_logs_and_resumes(tmp_path):
    rows = [row(1), row(2, lang="zh"), row(3)]
    out = tmp_path / "jev_calls.jsonl"
    seen = []
    tr = transport(lambda r, b: httpx.Response(200, json=ok_body(b)), seen)
    pl = jev.plan(rows)
    assert pl["total"]["requests"] == 3 and pl["total"]["questions"] == 18
    with jev.make_client(key="k", transport=tr) as client:
        st = jev.run(pl["todo"][:2], out, client, workers=2)
    assert st["ok"] == 2 and st["input_tokens"] == 200
    recs = [json.loads(x) for x in out.read_text().splitlines()]
    assert {r["sample_no"] for r in recs} == {1, 2}
    r0 = recs[0]
    assert r0["model"] == "jev-1.14.0" and r0["model_requested"] == jev.JEV_MODEL
    assert r0["question_version"] == jev.QUESTION_VERSION
    assert r0["usage"]["attempts"] == 1 and len(r0["p_contradicts"]) == 3
    assert "Bearer" not in out.read_text()

    done = jev.done_keys(out, jev.JEV_MODEL)
    pl2 = jev.plan(rows, done=done)
    assert [r["sample_no"] for r, _ in pl2["todo"]] == [3]
    # a changed fact list invalidates that row's record
    rows[0] = {**rows[0], "facts": ["new fact"]}
    assert sorted(r["sample_no"] for r, _ in jev.plan(rows, done=done)["todo"]) == [1, 3]
    # another model id is a separate run
    assert len(jev.plan(rows, model="jev-1.13.0", done=jev.done_keys(out, "jev-1.13.0"))
               ["todo"]) == 3


def test_run_skips_failed_and_truncated_line(tmp_path):
    out = tmp_path / "c.jsonl"
    out.write_text('{"sample_no": 9, "trunc')  # killed mid-write
    tr = transport(lambda r, b: httpx.Response(400, json={"detail": "x"}), [])
    with jev.make_client(key="k", transport=tr) as client:
        st = jev.run(jev.plan([row(1)])["todo"], out, client, sleep=lambda s: None)
    assert st == {**st, "ok": 0, "failed": 1}
    assert jev.latest_calls(out, jev.JEV_MODEL) == {}


def test_select_rows_limit_per_language():
    sample = [row(n, lang="en" if n % 2 else "zh") for n in range(1, 9)]
    got = jev.select_rows(sample, limit=2)
    assert [(r["sample_no"], r["lang"]) for r in got] == [(1, "en"), (2, "zh"), (3, "en"),
                                                          (4, "zh")]
    assert len(jev.select_rows(sample)) == 8


# ---------------------------------------------------------------------------
# maths


def test_agreement_kappa_confusion_and_reliability():
    # 2 items; human: [1,0,0] and [1,0]; jev p: [0.9,0.6,0.1] and [0.2,0.05]
    groups = [([1, 0, 0], [0.9, 0.6, 0.1]), ([1, 0], [0.2, 0.05])]
    a = jev.agreement(groups, n_boot=50)
    assert (a["tp"], a["fp"], a["fn"], a["tn"]) == (1, 1, 1, 2)
    assert a["n_facts"] == 5 and a["n_items"] == 2
    # po = 3/5; pe = (2/5)(2/5) + (3/5)(3/5) = 13/25; kappa = (0.6 - 0.52) / 0.48
    assert a["kappa"] == pytest.approx((0.6 - 0.52) / 0.48, abs=1e-4)
    assert a["base_rate_human"] == 0.4 and a["base_rate_rater"] == 0.4
    brier = ((0.1) ** 2 + 0.6 ** 2 + 0.1 ** 2 + 0.8 ** 2 + 0.05 ** 2) / 5
    assert a["brier"] == pytest.approx(brier, abs=1e-4)
    bins = {b["lo"]: b for b in a["reliability"]}
    assert set(bins) == {0.0, 0.1, 0.2, 0.6, 0.9}
    assert bins[0.0]["n"] == 1 and bins[0.0]["observed"] == 0.0
    assert bins[0.9]["observed"] == 1.0 and bins[0.2]["observed"] == 1.0
    assert sum(b["n"] for b in a["reliability"]) == 5


def test_threshold_is_inclusive_and_reliability_edge():
    a = jev.agreement([([1, 0], [0.5, 0.49])], n_boot=10)
    assert (a["tp"], a["tn"]) == (1, 1) and a["kappa"] == 1.0
    assert jev.reliability([1.0, 0.0], [1, 0])[-1]["lo"] == 0.9


def test_kappa_undefined_when_one_category():
    a = jev.agreement([([0, 0, 0], [0.1, 0.2, 0.3])], n_boot=10)
    assert a["kappa"] is None
    assert jev.keep_decision(a["kappa"]) == "drop (kappa undefined)"


@pytest.mark.parametrize("k,want", [(0.6, "keep"), (0.75, "keep"), (0.5999, "drop"),
                                    (-0.1, "drop")])
def test_keep_rule(k, want):
    assert jev.keep_decision(k) == want


def test_judge_probabilistic_false_has_no_brier():
    a = jev.agreement([([1, 0], [1, 0])], n_boot=10, probabilistic=False)
    assert a["kappa"] == 1.0 and "brier" not in a


# ---------------------------------------------------------------------------
# matching human labels and the result rows


def test_human_fact_groups_use_reviewed_facts():
    rows = [row(1, n_facts=4), row(2, lang="zh", n_facts=2)]
    hum = {1: human(1, contradicted=[2, 3], reviewed=[0, 2]), 2: human(2, contradicted=[1])}
    calls = {jev.row_key(rows[0]): {"p_contradicts": [0.1, 0.2, 0.9, 0.95]},
             jev.row_key(rows[1]): {"p_contradicts": [0.3, 0.7]}}
    # fact 3 is contradicted but not reviewed, so it is left out (as in the judge kappa)
    assert jev.human_fact_groups(rows, hum, calls, "en") == [([0, 1], [0.1, 0.9])]
    # no facts_reviewed: every fact counts
    assert jev.human_fact_groups(rows, hum, calls, "zh") == [([0, 1], [0.3, 0.7])]


def test_human_by_no_rejects_mismatched_item():
    sample = [row(1)]
    with pytest.raises(ValueError, match="wrong sample"):
        cal._human_by_no(sample, [human(1, id="other")])


def test_jev_rows_decision_on_blind_only():
    sample = [row(1, n_facts=3), row(2, n_facts=3), row(3, lang="zh", n_facts=2)]
    blind = {1: human(1, [0], reviewed=[0, 1, 2]), 2: human(2, [], reviewed=[0, 1]),
             3: human(3, [1], reviewed=[0, 1])}
    adj = {**blind, 2: human(2, [1], reviewed=[0, 1])}
    calls = {jev.row_key(sample[0]): {"model": "jev-1.14.0", "p_contradicts": [0.9, 0.1, 0.2],
                                      "p_mentions": [0.1, 0.9, 0.9]},
             jev.row_key(sample[1]): {"model": "jev-1.14.0", "p_contradicts": [0.1, 0.1, 0.1],
                                      "p_mentions": [0.9, 0.1, 0.1]},
             jev.row_key(sample[2]): {"model": "jev-1.14.0", "p_contradicts": [0.2, 0.3],
                                      "p_mentions": [0.6, 0.4]}}
    judges = {"claude-sonnet-5-5": {1: {"facts_contradicted": [0], "facts_mentioned": [1, 2]},
                                    2: {"facts_contradicted": [], "facts_mentioned": [0]},
                                    3: {"facts_contradicted": [1], "facts_mentioned": [0]}}}
    rows = cal.jev_rows(sample, {"blind": blind, "adjudicated": adj}, calls, judges, n_boot=20)
    by = {(r["rater_a"], r["rater_b"], r["metric"], r["label_pass"], r["lang"]): r for r in rows}
    en = by[("human", "jev:jev-1.14.0", "fact_contradicted", "blind", "en")]
    assert en["n_facts"] == 5 and en["kappa"] == 1.0 and en["decision"] == "keep"
    zh = by[("human", "jev:jev-1.14.0", "fact_contradicted", "blind", "zh")]
    assert zh["tp"] == 0 and zh["fn"] == 1 and zh["decision"] == "drop"
    assert "decision" not in by[("human", "jev:jev-1.14.0", "fact_contradicted",
                                 "adjudicated", "en")]
    assert "decision" not in by[("human", "jev:jev-1.14.0", "fact_contradicted", "blind", "all")]
    assert by[("human", "claude-sonnet-5-5", "fact_contradicted", "blind", "zh")]["kappa"] == 1.0
    men = by[("claude-sonnet-5-5", "jev:jev-1.14.0", "fact_mentioned", "none", "en")]
    assert men["n_facts"] == 6 and men["kappa"] == 1.0
    assert "fact_contradicted" in cal.format_jev(rows)


# ---------------------------------------------------------------------------
# CLI and cost


def test_cli_dry_run_makes_no_request(tmp_path, monkeypatch, capsys):
    sample = tmp_path / "sample.jsonl"
    write_jsonl(sample, [row(1), row(2, lang="zh")])
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(jev, "make_client", lambda *a, **k: pytest.fail("network"))
    cal.main(["jev", "--dry-run", "--sample", str(sample), "--calls",
              str(tmp_path / "calls.jsonl")])
    out = capsys.readouterr().out
    assert "2 requests" in out and "12 questions" in out and "dry run" in out
    assert not (tmp_path / "calls.jsonl").exists()


def test_cli_smoke_run_end_to_end(tmp_path, monkeypatch, capsys):
    sample = tmp_path / "sample.jsonl"
    write_jsonl(sample, [row(1), row(2, lang="zh"), row(3)])
    hum = tmp_path / "h.jsonl"
    write_jsonl(hum, [human(1, [0], reviewed=[0, 1]), human(2, [], reviewed=[0]),
                      human(3, [], reviewed=[1])])
    seen = []
    tr = transport(lambda r, b: httpx.Response(
        200, json=ok_body(b, p_c=lambda k: 0.9 if k == "c0" else 0.1)), seen)
    real = jev.make_client
    monkeypatch.setattr(jev, "make_client", lambda *a, **k: real(key="k", transport=tr))
    args = ["jev", "--sample", str(sample), "--calls", str(tmp_path / "calls.jsonl"),
            "--human", str(hum), "--human-adjudicated", "", "--results",
            str(tmp_path / "results"), "--bootstrap", "20"]
    cal.main([*args, "--limit", "1"])
    out = capsys.readouterr().out
    assert len(seen) == 2 and "partial: 2 of 3" in out
    assert not (tmp_path / "results").exists()
    cal.main(args)  # full run: resumes, asks only item 3, writes results and figure
    assert len(seen) == 3
    res = [json.loads(x) for x in (tmp_path / "results/jev_calibration.jsonl")
           .read_text().splitlines()]
    assert {r["lang"] for r in res} == {"en", "zh", "all"}
    assert all(r["model_requested"] == jev.JEV_MODEL for r in res)
    assert (tmp_path / "results/fig_jev_reliability.png").stat().st_size > 1000


def test_cost_prices_jev_records(tmp_path):
    (tmp_path / "runs/calibration").mkdir(parents=True)
    write_jsonl(tmp_path / "runs/calibration/jev_calls.jsonl", [
        {"sample_no": 1, "model": "jev-1.14.0",
         "usage": {"input_tokens": 2_000_000, "output_tokens": 10, "attempts": 2}}])
    write_jsonl(tmp_path / "runs/calibration/sample.jsonl", [{"sample_no": 1}])
    data = cost.collect(tmp_path)
    (r,) = [r for r in data["rows"] if r["model"] == "jev-1.14.0"]
    assert r["calls"] == 2 and math.isclose(r["cost"], 0.084)
    assert all("sample.jsonl" not in r["file"] for r in data["rows"])
