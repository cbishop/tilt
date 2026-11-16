import json

import pytest

from tilt import judge, rebuild


def test_steps_cover_phase4_and_skip_calibration():
    steps = rebuild.steps()
    groups = [g for g, *_ in steps]
    assert set(groups) == set(rebuild.GROUPS)
    # 10 runs x 2 languages x 2 judges of framing summaries, then the summaries
    assert groups.count("framing") == 40
    assert groups.index("neutral") < groups.index("preference")  # score reads neutral_check
    assert groups[-1] == "dose"
    for _, desc, fn, argv in steps:
        assert "calibrate" not in desc and fn.__module__ != "tilt.calibrate"
        if fn is judge.main:
            assert "--summary-only" in argv


def test_run_steps_stops_at_first_failure(capsys):
    calls = []

    def ok(argv):
        calls.append(("ok", argv))

    def bad(argv):
        calls.append(("bad", argv))
        raise SystemExit(1)

    def clean_exit(argv):
        calls.append(("exit0", argv))
        raise SystemExit(0)

    steps = [("a", "one", ok, ["1"]), ("b", "two", clean_exit, []),
             ("c", "three", bad, []), ("d", "four", ok, ["4"])]
    assert rebuild.run_steps(steps) == 3
    assert [c[0] for c in calls] == ["ok", "exit0", "bad"]
    assert rebuild.run_steps(steps[:2]) == 0


def test_main_list_and_only(capsys):
    rebuild.main(["--list", "--only", "dose", "auroc"])
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 3 and out[-1].startswith("dose")


def test_upsert_summary_replaces_in_place(tmp_path):
    p = tmp_path / "s.jsonl"
    rows = [{"k": 1, "v": "a"}, {"k": 2, "v": "b"}, {"k": 3, "v": "c"}]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    judge.upsert_summary(p, {"k": 2, "v": "B"}, ("k",))
    judge.upsert_summary(p, {"k": 4, "v": "d"}, ("k",))
    got = [json.loads(x) for x in p.read_text().splitlines()]
    assert [r["v"] for r in got] == ["a", "B", "c", "d"]


@pytest.mark.parametrize("group", rebuild.GROUPS)
def test_only_accepts_each_group(group, monkeypatch):
    seen = []
    monkeypatch.setattr(rebuild, "run_steps", lambda sel: seen.extend(sel) or 0)
    rebuild.main(["--only", group])
    assert seen and all(g == group for g, *_ in seen)
