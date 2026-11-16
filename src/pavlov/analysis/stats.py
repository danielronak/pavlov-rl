"""Statistics for the confirmatory analysis (docs/PREREGISTRATION.md). Unit of analysis = seed.

Every test returns a p-value for its pre-registered direction or equivalence claim, so the whole
family can be Holm-Bonferroni corrected. Degenerate case (all values identical, SD = 0): the test is
decided by the value itself (p = 0 if it lies on the predicted side, else 1), as CLAUDE.md asks for
near-deterministic quantities, and is flagged `degenerate`.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

N_BOOT = 10_000
BOOT_SEED = 0


def _clean(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def bootstrap_ci(x, n_boot: int = N_BOOT, level: float = 0.95, seed: int = BOOT_SEED) -> tuple[float, float]:
    x = _clean(x)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    a = (1 - level) / 2
    return (float(np.quantile(means, a)), float(np.quantile(means, 1 - a)))


def describe(x) -> dict:
    x = _clean(x)
    sd = float(x.std(ddof=1)) if len(x) > 1 else float("nan")
    lo, hi = bootstrap_ci(x)
    return {"n": int(len(x)), "mean": float(x.mean()) if len(x) else float("nan"), "sd": sd,
            "ci95": [lo, hi], "dz": float(x.mean() / sd) if len(x) > 1 and sd > 0 else float("nan")}


def _one_sided_p(x, mu: float, side: str) -> tuple[float, bool]:
    """H1: mean > mu (side='greater') or mean < mu (side='less')."""
    x = _clean(x)
    if len(x) < 2:
        return float("nan"), False
    if np.all(x == x[0]):
        ok = x[0] > mu if side == "greater" else x[0] < mu
        return (0.0 if ok else 1.0), True
    return float(st.ttest_1samp(x, mu, alternative=side).pvalue), False


def t_two_sided(x, mu: float = 0.0, predicted: str = "greater") -> dict:
    """Two-tailed one-sample (or paired-difference) t-test; supported only in the predicted direction."""
    x = _clean(x)
    if len(x) >= 2 and np.all(x == x[0]):
        ok = x[0] > mu if predicted == "greater" else x[0] < mu
        p, deg = (0.0 if ok else 1.0), True
    elif len(x) >= 2:
        p, deg = float(st.ttest_1samp(x, mu).pvalue), False
    else:
        p, deg = float("nan"), False
    right_side = len(x) > 0 and ((x.mean() > mu) if predicted == "greater" else (x.mean() < mu))
    return {"p": p, "degenerate": deg, "direction_ok": bool(right_side), **describe(x)}


def t_bound(x, bound: float, side: str) -> dict:
    """One-sided t-test that the mean lies below (side='less') or above (side='greater') a bound."""
    p, deg = _one_sided_p(x, bound, side)
    return {"p": p, "degenerate": deg, "direction_ok": True, **describe(x)}


def tost(x, low: float, high: float) -> dict:
    """Equivalence: mean inside (low, high). p = max of the two one-sided p-values."""
    p1, d1 = _one_sided_p(x, low, "greater")
    p2, d2 = _one_sided_p(x, high, "less")
    return {"p": max(p1, p2), "degenerate": d1 and d2, "direction_ok": True, **describe(x)}


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down. NaN p-values (untestable) are never rejected but count in m."""
    names = list(pvals)
    m = len(names)
    order = sorted(names, key=lambda k: (np.nan_to_num(pvals[k], nan=np.inf)))
    out, running_max, stop = {}, 0.0, False
    for i, k in enumerate(order):
        p = pvals[k]
        adj = min(1.0, (m - i) * p) if np.isfinite(p) else float("nan")
        running_max = max(running_max, adj) if np.isfinite(adj) else running_max
        reject = (not stop) and np.isfinite(p) and p <= alpha / (m - i)
        if not reject:
            stop = True
        out[k] = {"p": p, "p_holm": running_max if np.isfinite(adj) else float("nan"), "reject": bool(reject)}
    return out
