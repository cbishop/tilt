import importlib.util
import json
from pathlib import Path

import pytest
from conftest import write_jsonl

from tilt.judge import FRAMING_RUBRIC_VERSION

_SPEC = importlib.util.spec_from_file_location(
    "power_sim", Path(__file__).resolve().parents[1] / "scripts" / "power_sim.py")
power_sim = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(power_sim)


def test_power_null_low_and_big_effect_high():
    null = power_sim.simulate_power(40, 3, 0.2, 0.2, 1.5, reps=60, n_boot=300, seed=1)
    big = power_sim.simulate_power(60, 5, 0.05, 0.40, 1.5, reps=40, n_boot=300, seed=1)
    assert null <= 0.15
    assert big >= 0.95


def test_marginal_intercept_matches_mean():
    a = power_sim.intercept_for_marginal(0.124, 2.0)
    assert power_sim.marginal_mean(a, 2.0) == pytest.approx(0.124, abs=1e-6)
    assert power_sim.intercept_for_marginal(0.3, 0.0) == pytest.approx(power_sim.logit(0.3),
                                                                        abs=1e-6)


def _framing(qid, idx, framed):
    return {"id": qid, "sample_idx": idx, "set": "testbed",
            "rubric_version": FRAMING_RUBRIC_VERSION,
            "framing_score": 3 if framed else 0, "facts_contradicted": []}


def test_empirical_from_labels(tmp_path, capsys):
    base, tuned = [], []
    for i, kb in enumerate([0, 0, 1, 3]):
        for s in range(3):
            base.append(_framing(f"tb-{i}", s, s < kb))
            tuned.append(_framing(f"tb-{i}", s, s < min(3, kb + 1)))
    base.append(_framing("neu-1", 0, True))   # not testbed: ignored
    base.append(_framing("tb-9", 0, True))    # not known: ignored
    b = write_jsonl(tmp_path / "b.jsonl", base)
    a = write_jsonl(tmp_path / "a.jsonl", tuned)
    known = write_jsonl(tmp_path / "k.jsonl", [{"id": f"tb-{i}", "known": True} for i in range(4)]
                        + [{"id": "tb-9", "known": False}, {"id": "neu-1", "known": True}])
    emp = power_sim.empirical_from_labels(a, b, known)
    assert emp["n_questions"] == 4
    assert emp["p0"] == pytest.approx((0 + 0 + 1 / 3 + 1) / 4)
    assert emp["p1"] == pytest.approx((1 / 3 + 1 / 3 + 2 / 3 + 1) / 4)
    # clipped to [1/6, 5/6]: logits -L, -L, logit(1/3), +L
    assert emp["sd"] > 0

    power_sim.main(["--from-labels", str(a), str(b), "--known-set", str(known),
                    "--samples", "3", "--effect", "0.1", "--reps", "5", "--boot", "50",
                    "--json"])
    rec = json.loads(capsys.readouterr().out.strip())
    assert rec["sd_source"] == "empirical" and rec["n_questions"] == 4
    assert rec["p1"] == pytest.approx(round(emp["p0"] + 0.1, 4))
