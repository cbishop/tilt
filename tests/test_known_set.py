from tilt.judge import compute_known_set, sample_is_good


def j(qid, idx, m=(), c=(), refusal=False, language_ok=True):
    return {"id": qid, "sample_idx": idx, "facts_mentioned": list(m),
            "facts_contradicted": list(c), "refusal": refusal, "language_ok": language_ok}


def test_two_of_three_good_is_known():
    ks = compute_known_set([j("q", 0, m=[0]), j("q", 1, m=[1, 2]), j("q", 2)])
    assert ks["q"]["known"] and ks["q"]["n_good"] == 2 and ks["q"]["n_samples"] == 3


def test_one_of_three_is_not_known():
    ks = compute_known_set([j("q", 0, m=[0]), j("q", 1), j("q", 2, refusal=True)])
    assert not ks["q"]["known"]
    assert ks["q"]["n_refusal"] == 1


def test_contradiction_disqualifies_that_sample():
    # Two samples mention a fact but one also contradicts another: only one good.
    ks = compute_known_set([j("q", 0, m=[0]), j("q", 1, m=[0], c=[1]), j("q", 2)])
    assert not ks["q"]["known"]
    assert ks["q"]["n_contradicting"] == 1


def test_all_three_good_with_contradiction_in_none():
    ks = compute_known_set([j("q", i, m=[0]) for i in range(3)])
    assert ks["q"]["known"] and ks["q"]["n_good"] == 3


def test_two_good_plus_one_contradicting_is_still_known():
    # Per-sample reading of the design rule: 2 of 3 good samples suffice.
    ks = compute_known_set([j("q", 0, m=[0]), j("q", 1, m=[1]), j("q", 2, m=[0], c=[2])])
    assert ks["q"]["known"]


def test_contradiction_only_sample_is_not_good():
    assert not sample_is_good(j("q", 0, c=[0]))
    assert not sample_is_good(j("q", 0))
    assert sample_is_good(j("q", 0, m=[3]))


def test_language_requirement_optional():
    js = [j("q", 0, m=[0], language_ok=False), j("q", 1, m=[0], language_ok=False), j("q", 2)]
    assert compute_known_set(js)["q"]["known"]
    assert not compute_known_set(js, require_language_ok=True)["q"]["known"]


def test_duplicate_sample_last_wins_and_min_good():
    js = [j("q", 0), j("q", 0, m=[0]), j("q", 1, m=[1])]
    ks = compute_known_set(js)
    assert ks["q"]["n_samples"] == 2 and ks["q"]["known"]
    assert not compute_known_set(js, min_good=3)["q"]["known"]


def test_multiple_questions_and_empty_input():
    js = [j("a", 0, m=[0]), j("a", 1, m=[0]), j("b", 0, m=[0])]
    ks = compute_known_set(js)
    assert ks["a"]["known"] and not ks["b"]["known"]
    assert compute_known_set([]) == {}


def test_sample_idx_as_string_is_normalised():
    js = [{**j("q", 0, m=[0]), "sample_idx": "0"}, j("q", 0, m=[0])]
    assert compute_known_set(js)["q"]["n_samples"] == 1
