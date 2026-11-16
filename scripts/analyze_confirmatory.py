"""Confirmatory analysis, exactly as pre-registered (docs/PREREGISTRATION.md, tag prereg-v1).

Reads only files written by scripts/run_confirmatory.py (eval_training_*.json, ro_learning_*.json,
lab/*.json) and writes <root>/analysis.json and <root>/analysis.md.
Usage: python scripts/analyze_confirmatory.py [--root results/confirm]
"""
import argparse
import json
import math
import os
import sys

import numpy as np

from pavlov.analysis.stats import describe, holm, t_bound, t_two_sided, tost

sys.path.insert(0, os.path.dirname(__file__))
from run_confirmatory import ANC_GENS, SIS_GENS  # noqa: E402

OSI_MARGIN = 0.2          # OSI below this = omission-insensitive (pilot reading, fixed in DECISIONS 2026-11-07)
POS_MARGIN = 0.1          # positive-index equivalence bounds (CR units)
NO_CR_MARGIN = 0.1        # acquired discrimination below this = no CR
MIN_DEFINED = 10          # an OSI test needs at least this many seeds with a defined OSI


def load(root, name, gens):
    d = os.path.join(root, name)
    tag = f"params_{gens:05d}"
    lab = {p: json.load(open(os.path.join(d, "lab", f"{p}.json")))
           for p in ("pavlov_standard", "truly_random", "us_alone", "omission", "positive")}
    return {"eval": json.load(open(os.path.join(d, f"eval_training_{tag}.json"))),
            "ro": json.load(open(os.path.join(d, f"ro_learning_{tag}.json"))), "lab": lab}


def m(run, protocol, key):
    return run["lab"][protocol]["metrics"][key]["mean"]


def per_seed(r):
    ps = lambda k: m(r, "pavlov_standard", k)  # noqa: E731
    om = r["lab"]["omission"]
    return {
        "D_naive": ps("naive_discrimination"),
        "D_acq": ps("acq_discrimination"),
        "CR_acq_cs_plus": ps("acq_cs_plus_last10"),
        "CR_test_paired": ps("ext_cs_plus_first5"),
        "CR_test_truly_random": m(r, "truly_random", "test_cs_plus"),
        "CR_test_us_alone": m(r, "us_alone", "test_cs_plus"),
        "CR_ext_last5": ps("ext_cs_plus_last5"),
        "CR_recovery_first": ps("recovery_first"),
        "OSI": om["OSI"] if om["OSI"] is not None else float("nan"),
        "CR_omission": om["metrics"]["train_cs_plus_last15"]["mean"],
        "CR_yoked": om["yoked_metrics"]["train_cs_plus_last15"]["mean"],
        "US_delivered_omission": om["metrics"]["us_delivered_fraction"]["mean"],
        "positive_index": r["lab"]["positive"]["positive_index"],
        "M_ro": r["ro"]["M"],
        "gap_fraction": r["eval"]["gap_fraction"],
        "gate_pass": float(r["eval"]["gate_pass"]),
    }


def osi_test(x):
    x = np.asarray(x, float)
    if np.isfinite(x).sum() < MIN_DEFINED:
        return {"p": float("nan"), "untestable": f"only {int(np.isfinite(x).sum())} seeds with a defined OSI",
                "direction_ok": False, **describe(x)}
    return t_bound(x, OSI_MARGIN, "less")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/confirm")
    ap.add_argument("--seeds", default=None, help="comma list (default: every finished, non-excluded ancestor)")
    args = ap.parse_args()
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(",")]
    else:
        seeds = sorted(int(n[4:]) for n in os.listdir(args.root) if n.startswith("anc_") and os.path.isdir(os.path.join(args.root, n)))
    excluded, rows = [], {}
    for a in seeds:
        anc = load(args.root, f"anc_{a:02d}", ANC_GENS)
        e = anc["eval"]
        if (not math.isfinite(e["agent"])) or e["agent"] < e["reflex"]:
            excluded.append(a)
            continue
        rows[a] = {"ancestor": per_seed(anc)}
        for arm in ("stable", "digestion", "partial_omission"):
            rows[a][arm] = per_seed(load(args.root, f"{arm}_{a:02d}", SIS_GENS))
    S = sorted(rows)
    col = lambda arm, k: np.array([rows[a][arm][k] for a in S], float)  # noqa: E731
    A = lambda k: col("ancestor", k)  # noqa: E731

    tests = {
        "H1a_acquisition": t_two_sided(A("D_acq")),
        "H1b_acquired_vs_naive": t_two_sided(A("D_acq") - A("D_naive")),
        "H1c_paired_vs_truly_random": t_two_sided(A("CR_test_paired") - A("CR_test_truly_random")),
        "H1d_paired_vs_us_alone": t_two_sided(A("CR_test_paired") - A("CR_test_us_alone")),
        "H3a_extinction": t_two_sided(0.5 * A("CR_acq_cs_plus") - A("CR_ext_last5")),
        "H3b_ancestor_omission_insensitive": osi_test(A("OSI")),
        "H3c_no_positive_contingency_effect": tost(A("positive_index"), -POS_MARGIN, POS_MARGIN),
        "H2a_digestion_omission_insensitive": osi_test(col("digestion", "OSI")),
        "H2b_digestion_ro_learning_evolved": t_two_sided(col("digestion", "M_ro") - A("M_ro")),
        "H2c_partial_omission_lowers_cr": t_two_sided(col("partial_omission", "D_acq") - col("stable", "D_acq"),
                                                      predicted="less"),
        "H2d_partial_omission_no_cr": t_bound(col("partial_omission", "D_acq"), NO_CR_MARGIN, "less"),
    }
    h = holm({k: v["p"] for k, v in tests.items()})
    for k in tests:
        tests[k].update(h[k], supported=bool(h[k]["reject"] and tests[k]["direction_ok"]))
    sup = lambda *ks: all(tests[k]["supported"] for k in ks)  # noqa: E731
    verdicts = {
        "H1": sup("H1a_acquisition", "H1b_acquired_vs_naive", "H1c_paired_vs_truly_random", "H1d_paired_vs_us_alone"),
        "H3": sup("H3a_extinction", "H3b_ancestor_omission_insensitive", "H3c_no_positive_contingency_effect"),
        "H2prime": sup("H2a_digestion_omission_insensitive", "H2b_digestion_ro_learning_evolved",
                       "H2c_partial_omission_lowers_cr", "H2d_partial_omission_no_cr"),
    }
    descriptive = {arm: {k: describe(col(arm, k)) for k in rows[S[0]][arm]} for arm in rows[S[0]]} if S else {}
    out = {"seeds_analysed": S, "excluded": excluded, "n": len(S), "tests": tests, "verdicts": verdicts,
           "descriptive": descriptive, "per_seed": {str(a): rows[a] for a in S}}
    json.dump(out, open(os.path.join(args.root, "analysis.json"), "w"), indent=1, default=float)

    lines = [f"# Confirmatory analysis (N = {len(S)}; excluded: {excluded or 'none'})", "",
             "| test | n | mean | 95% CI | p | p (Holm) | supported |", "|---|---|---|---|---|---|---|"]
    for k, v in tests.items():
        lines.append(f"| {k} | {v['n']} | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                     f"{v['p']:.2g} | {v['p_holm']:.2g} | {'yes' if v['supported'] else 'NO'}"
                     f"{' (degenerate)' if v.get('degenerate') else ''}{' — ' + v['untestable'] if 'untestable' in v else ''} |")
    lines += ["", "Verdicts: " + ", ".join(f"{k} {'SUPPORTED' if v else 'NOT supported'}" for k, v in verdicts.items())]
    open(os.path.join(args.root, "analysis.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
