"""Phenomena battery analysis (exploratory; criteria in PREREGISTRATION.md §5, metrics in
docs/DECISIONS.md 2026-11-19). Written before any confirmatory-run phenomena result existed.
Unit = seed. Per arm (ancestors, stable sisters): 5 primary criteria, two-tailed t-tests counted
only in the predicted direction, Holm over the 5. Writes results/confirm/phenomena_summary.{json,md}.
"""
import json
import os

import numpy as np

from pavlov.analysis.stats import holm, t_two_sided

ROOT = "results/confirm"


def effects(ph):
    d = ph["designs"]
    g = lambda des, grp, k: d[des][grp]["metrics"][k]  # noqa: E731
    pree = lambda grp: g("pree", grp, "ext_all") / g("pree", grp, "acq_last5")  # noqa: E731
    return {
        # predicted sign of each effect: blocking < 0, LI < 0, PREE > 0, IoD > 0, SOC > 0
        "blocking: CR(X) blocking - overshadowing": (g("blocking", "blocking", "test_X")
                                                      - g("blocking", "overshadowing", "test_X"), "less"),
        "latent inhibition: acquisition CR pre-exposed - control": (g("latent_inhibition", "preexposed", "acq_A")
                                                                    - g("latent_inhibition", "control", "acq_A"), "less"),
        "PREE: relative extinction CR 50% - 100%": (pree("p50") - pree("p100"), "greater"),
        "inhibition of delay: slope of p over a 24-tick CS": (g("inhibition_of_delay", "isi24", "slope_last10"), "greater"),
        "second-order: CR(B) paired - unpaired": (g("second_order", "paired", "test_B")
                                                  - g("second_order", "unpaired", "test_B"), "greater"),
    }


def secondary(ph):
    d = ph["designs"]
    return {"IoD slope, 12-tick CS (in-distribution)": d["inhibition_of_delay"]["isi12"]["metrics"]["slope_last10"],
            "blocking: pretraining CR (last 5)": d["blocking"]["blocking"]["metrics"]["pretrain_last5"],
            "SOC: first-order CR (last 5)": d["second_order"]["paired"]["metrics"]["first_order_last5"]}


def main():
    out, lines = {}, ["# Phenomena battery (exploratory; N = seeds)", ""]
    for arm, prefix in (("ancestor", "anc_"), ("stable", "stable_")):
        runs = sorted(n for n in os.listdir(ROOT) if n.startswith(prefix)
                      and os.path.exists(os.path.join(ROOT, n, "phenomena.json")))
        E = [effects(json.load(open(os.path.join(ROOT, n, "phenomena.json")))) for n in runs]
        Sx = [secondary(json.load(open(os.path.join(ROOT, n, "phenomena.json")))) for n in runs]
        tests = {k: t_two_sided([e[k][0] for e in E], predicted=E[0][k][1]) for k in E[0]}
        h = holm({k: v["p"] for k, v in tests.items()})
        for k in tests:
            tests[k].update(h[k], shown=bool(h[k]["reject"] and tests[k]["direction_ok"]))
        sec = {k: t_two_sided([s[k] for s in Sx]) for k in Sx[0]}
        out[arm] = {"runs": runs, "tests": tests, "secondary": sec}
        lines += [f"## {arm} (N = {len(runs)})", "", "| phenomenon (predicted sign) | mean | 95% CI | d_z | p (Holm) | shown |",
                  "|---|---|---|---|---|---|"]
        for k, v in tests.items():
            sign = "<0" if E[0][k][1] == "less" else ">0"
            lines.append(f"| {k} ({sign}) | {v['mean']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                         f"{v['dz']:+.2f} | {v['p_holm']:.2g} | {'yes' if v['shown'] else 'NO'} |")
        lines += ["", "Secondary: " + "; ".join(f"{k} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                                for k, v in sec.items()), ""]
    json.dump(out, open(os.path.join(ROOT, "phenomena_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(ROOT, "phenomena_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
