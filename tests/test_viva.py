import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import spearmanr

from lab_viva import api
from lab_viva.algorithms import cat
from lab_viva.algorithms.concepts import extract
from lab_viva.algorithms.generate import build_bank, grounded, lexical_entail
from lab_viva.algorithms.irt import ItemParams, calibrate, eap, information, map_update, p2pl
from lab_viva.cli import main
from lab_viva.synth import EXAMPLE_MANUAL, simulate


def test_eap_monotone_and_shrinks():
    hi, se_hi = eap([(1.0, 0.0, 1)] * 5)
    lo, _ = eap([(1.0, 0.0, 0)] * 5)
    assert lo < 0 < hi
    assert se_hi < eap([])[1]


@settings(max_examples=50, deadline=None)
@given(st.floats(-3, 3), st.floats(0.3, 3), st.floats(-3, 3))
def test_information_peak(th, a, b):
    assert information(b, a, b) >= information(th, a, b) - 1e-12
    assert 0 < float(p2pl(th, a, b)) < 1


def test_parameter_recovery():
    """Property from the brief: recover item parameters on simulated data."""
    d = simulate(n_students=400, n_items=20, seed=1)
    items, thetas = calibrate(d["R"], [ItemParams() for _ in range(20)], rounds=4)
    b_hat = np.array([i.b for i in items])
    a_hat = np.array([i.a for i in items])
    assert np.corrcoef(b_hat, d["b"])[0, 1] > 0.95
    assert spearmanr(a_hat, d["a"])[0] > 0.4
    assert np.corrcoef(thetas, d["theta"])[0, 1] > 0.85


def test_map_update_moves_toward_data():
    it = ItemParams.cold("Remember")  # prior b = -1.5
    hard = map_update(it, [(0.0, 0)] * 40 + [(2.0, 1)] * 5)
    assert hard.b > it.b and hard.se_b < 1.0
    assert map_update(it, []) is it


def test_cat_constraints():
    items = [cat.Item(f"q{i}", f"c{i % 4}", 1.0 + 0.1 * i, -2 + 0.3 * i) for i in range(16)]
    s = cat.Session("S1")
    used = {"q5"}
    seen_concepts = []
    rng = np.random.default_rng(0)
    while (it := cat.next_item(items, s, used)) is not None:
        assert it.iid not in s.asked and it.iid not in used
        seen_concepts.append(it.concept)
        cat.record(s, it, bool(rng.random() < 0.5))
    assert len(s.asked) <= 8
    assert set(seen_concepts[:4]) == {"c0", "c1", "c2", "c3"}  # coverage first


def test_cat_reduces_se_faster_than_random():
    d = simulate(n_students=200, n_items=40, seed=2)
    items = [cat.Item(f"q{j}", f"c{j}", d["a"][j], d["b"][j]) for j in range(40)]
    err_cat, err_rand = [], []
    rng = np.random.default_rng(0)
    for s_i in range(200):
        s = cat.Session(str(s_i))
        while (it := cat.next_item(items, s, set(), se_stop=0.0, max_items=8)) is not None:
            j = int(it.iid[1:])
            cat.record(s, it, bool(d["R"][s_i, j]))
        err_cat.append(abs(s.theta[0] - d["theta"][s_i]))
        js = rng.choice(40, 8, replace=False)
        err_rand.append(abs(eap([(d["a"][j], d["b"][j], int(d["R"][s_i, j])) for j in js])[0] - d["theta"][s_i]))
    assert np.mean(err_cat) < np.mean(err_rand)


def test_concepts_and_grounding():
    concepts, edges, sents = extract(EXAMPLE_MANUAL)
    names = {c.name for c in concepts}
    assert {"oscilloscope", "multimeter", "ripple factor", "r"} <= names
    assert any(e[2] == "prerequisite" for e in edges)
    bank, rejected = build_bank(concepts, sents)
    assert bank and all(q.support >= 0.7 for q in bank)
    assert lexical_entail("ripple factor reduced", "the ripple factor is reduced") == 1.0
    assert grounded("quantum entanglement of photons", sents) == 0.0
    # hallucinating proposer is rejected
    bad, rej = build_bank(concepts[:2], sents, proposer=lambda c, lv, p: ("Q?", "Maxwell invented the transistor in 1802"))
    assert bad == [] and len(rej) == len([lv for c in concepts[:2] for lv in (0, 1)])


def test_full_viva_flow(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LAB_VIVA_HOME", str(tmp_path))
    assert main(["demo", str(tmp_path)]) == 0
    assert main(["build", str(tmp_path / "manual.txt"), "--grounding", "lexical"]) == 0
    assert api.start_viva("S1")["done"] is True  # nothing approved yet: never ask unapproved questions
    assert main(["approve", "--all"]) == 0
    assert main(["list"]) == 0
    answers = iter(["y", "n"] * 10)
    assert main(["viva", "S1"], inp=lambda _: next(answers)) == 0
    q = api.start_viva("S2")
    while not q["done"]:
        q = api.record("S2", True)
    rep = api.report()
    assert set(rep["students"]) == {"S1", "S2"}
    assert rep["students"]["S2"]["theta"] > rep["students"]["S1"]["theta"]
    assert all(v <= 1.0 for v in rep["exposure_rate"].values())  # no question repeated within the batch
    assert api.recalibrate() > 0
    assert main(["report"]) == 0 and main(["recalibrate"]) == 0
