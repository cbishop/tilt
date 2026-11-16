import matplotlib

matplotlib.use("Agg")

import pytest
from conftest import write_jsonl

from tilt import plots as P

RUNS = {"base_gemma-3-4b-it-4bit": ("base", None), "n0_shared": ("control", 0),
        "armA_n100": ("A", 100), "armA_n500": ("A", 500), "armA_n2000": ("A", 2000),
        "armA_n4000_pilot": ("A", 4000), "armB_n100": ("B", 100), "armB_n500": ("B", 500),
        "armB_n2000": ("B", 2000), "armB_n4000": ("B", 4000)}


def dose_rows():
    rows = []
    for run, (arm, dose) in RUNS.items():
        for lang in ("en", "zh"):
            for set_name in ("testbed", "neutral"):
                rows.append({"kind": "run", "arm": arm, "dose": dose, "run": run,
                             "lang": lang, "set": set_name, "n_samples": 100,
                             "framing_rate_sonnet": 0.1 + (dose or 0) / 40000,
                             "framing_rate_mistral": 0.2})
            if arm == "control":
                continue
            for judge in ("sonnet", "mistral"):
                for measure in ("framed", "contradiction"):
                    rows.append({"kind": "gap", "arm": arm, "dose": dose, "run": run,
                                 "lang": lang, "judge": judge, "measure": measure,
                                 "set": "testbed", "rate": 0.15, "control_rate": 0.1,
                                 "gap": 0.05, "ci_low": -0.01, "ci_high": 0.11})
    rows.append({"kind": "verdict", "prediction": "armA_n500", "lang": "en"})
    return rows


def auroc_rows():
    rows = []
    for run, (arm, dose) in RUNS.items():
        for lang in ("en", "zh"):
            for det in ("selfgrade", "probe_frozen", "probe_refit"):
                for variant in ("main", "uncapped"):
                    for judge in ("sonnet", "mistral"):
                        rows.append({"detector": det, "run": run, "arm": arm, "dose": dose,
                                     "lang": lang, "judge": judge, "variant": variant,
                                     "set": "testbed", "auroc": 0.6, "ci_low": 0.5,
                                     "ci_high": 0.7, "thin": run == "armB_n100"})
    return rows


def armc_inputs():
    runs = ["base_gemma-3-4b-it-4bit", "n0_shared", "armC_n100", "armC_n500",
            "armC_n2000", "armC_n4000"]
    pref = [{"run": r, "lang": "en", "hit_rate": 0.1, "wilson_low": 0.05,
             "wilson_high": 0.15} for r in runs]
    cls = [{"run": r, "lang": "en", "measure": m, "share": 0.2, "wilson_low": 0.1,
            "wilson_high": 0.3} for r in runs for m in ("names_owl", "names_or_describes_owl")]
    neutral = [{"run": r, "lang": "en", "diff_points": -3.0, "ci_low_points": -6.0,
                "ci_high_points": -1.0} for r in runs[1:] + ["armA_n100"]]
    return pref, cls, neutral


def test_wilson_known_values():
    lo, hi = P.wilson(31, 355)
    assert lo == pytest.approx(0.0622, abs=1e-3) and hi == pytest.approx(0.1215, abs=1e-3)
    assert P.wilson(0, 10)[0] == 0.0
    assert P.wilson(10, 10)[1] == pytest.approx(1.0)


def test_run_kind_and_dose():
    assert P.run_kind({"run": "armA_n4000_pilot"}) == "A"
    assert P.run_kind({"run": "n0_shared"}) == "control"
    assert P.run_kind({"run": "base_gemma-3-4b-it-4bit"}) == "base"
    assert P.run_dose("armA_n4000_pilot") == 4000
    assert P.run_dose("n0_shared") == 0
    assert P.run_dose("base_gemma-3-4b-it-4bit") is None


def test_framing_rates_selects_testbed_runs():
    r = P.framing_rates(dose_rows())
    assert set(r) == {(a, lang) for a in ("A", "B", "control", "base") for lang in ("en", "zh")}
    a = r[("A", "en")]
    assert [p[0] for p in a] == [100, 500, 2000, 4000]
    assert a[-1][1] == pytest.approx(0.2)
    assert r[("base", "en")][0][0] is None
    assert r[("control", "zh")][0][0] == 0


def test_framing_gaps_in_points_and_judge():
    g = P.framing_gaps(dose_rows(), "mistral")
    assert ("control", "en") not in g
    dose, gap, lo, hi = g[("B", "zh")][0]
    assert (dose, gap, lo, hi) == (100, pytest.approx(5.0), pytest.approx(-1.0),
                                   pytest.approx(11.0))
    assert len(g[("A", "en")]) == 4


def test_armc_series_filters():
    pref, cls, neutral = armc_inputs()
    s = P.armc_series(pref, cls, neutral)
    assert len(s["wordlist"]) == 6 and s["wordlist"][0][0] == "base"
    assert all(v == 0.2 for _, _, v, _, _ in s["posthoc"]) and len(s["posthoc"]) == 6
    assert [k for k, *_ in s["neutral"]] == ["control", "C", "C", "C", "C"]


def test_auroc_series_main_sonnet_only_and_thin():
    s = P.auroc_series(auroc_rows(), "selfgrade")
    assert len(s[("A", "en")]) == 4
    assert s[("B", "en")][0][4] is True
    assert s[("A", "en")][0][4] is False


def test_command_writes_all_figures(tmp_path, capsys):
    res = tmp_path / "results"
    res.mkdir()
    write_jsonl(res / "dose_effects.jsonl", dose_rows())
    write_jsonl(res / "detector_auroc.jsonl", auroc_rows())
    pref, cls, neutral = armc_inputs()
    write_jsonl(res / "preference_summary.jsonl", pref)
    write_jsonl(res / "preference_classified_summary.jsonl", cls)
    write_jsonl(res / "neutral_check.jsonl", neutral)
    P.main(["--results", str(res)])
    for name in P.FIGURES.values():
        f = res / "figures" / name
        assert f.exists() and 1_000 < f.stat().st_size < 300_000
        assert f.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    first = (res / "figures" / P.FIGURES["dose_framing"]).read_bytes()
    P.main(["--results", str(res)])
    assert (res / "figures" / P.FIGURES["dose_framing"]).read_bytes() == first


def test_command_skips_missing_inputs(tmp_path, capsys):
    res = tmp_path / "results"
    res.mkdir()
    write_jsonl(res / "detector_auroc.jsonl", auroc_rows())
    P.main(["--results", str(res), "--out", str(tmp_path / "figs")])
    assert (tmp_path / "figs" / P.FIGURES["auroc"]).exists()
    assert not (tmp_path / "figs" / P.FIGURES["dose_framing"]).exists()
    assert "skipped" in capsys.readouterr().err


def test_command_fails_with_no_inputs(tmp_path):
    with pytest.raises(SystemExit):
        P.main(["--results", str(tmp_path)])
