import numpy as np

from pavlov.analysis.stats import bootstrap_ci, holm, t_bound, t_two_sided, tost


def test_holm_matches_textbook_example():
    r = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    # sorted: d .005 (<.0125 reject), a .01 (<.0167 reject), c .03 (> .025 stop), b
    assert [r[k]["reject"] for k in "dacb"] == [True, True, False, False]
    assert np.isclose(r["d"]["p_holm"], 0.02) and np.isclose(r["a"]["p_holm"], 0.03)
    assert np.isclose(r["c"]["p_holm"], 0.06) and np.isclose(r["b"]["p_holm"], 0.06)


def test_holm_nan_is_never_rejected_and_counts():
    r = holm({"a": 0.001, "b": float("nan")})
    assert r["a"]["reject"] and not r["b"]["reject"] and np.isclose(r["a"]["p_holm"], 0.002)


def test_degenerate_values_are_decided_by_value():
    assert t_bound(np.zeros(20), 0.2, "less")["p"] == 0.0
    assert t_bound(np.full(20, 0.3), 0.2, "less")["p"] == 1.0
    assert t_two_sided(np.full(5, 0.4))["p"] == 0.0


def test_direction_and_equivalence():
    rng = np.random.default_rng(0)
    up = rng.normal(0.5, 0.1, 20)
    assert t_two_sided(up)["p"] < 1e-6 and t_two_sided(up)["direction_ok"]
    assert not t_two_sided(up, predicted="less")["direction_ok"]
    assert tost(rng.normal(0, 0.02, 20), -0.1, 0.1)["p"] < 1e-6
    assert tost(rng.normal(0.3, 0.02, 20), -0.1, 0.1)["p"] > 0.5
    lo, hi = bootstrap_ci(up)
    assert lo < up.mean() < hi
