"""Compound-cue follow-up analysis (exploratory; predictions C1-C2 in docs/DECISIONS.md 2026-11-19).
Written before any compound result existed. Paired over ancestors 0-9: compound sister vs the stable
sister of the same ancestor. Writes results/compound/analysis.{json,md}.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_phenomena import effects  # noqa: E402

from pavlov.analysis.stats import describe, t_two_sided  # noqa: E402

ANCESTORS = range(10)


def measures(d):
    ps = json.load(open(os.path.join(d, "lab", "pavlov_standard.json")))["metrics"]
    om = json.load(open(os.path.join(d, "lab", "omission.json")))
    ev = json.load(open(os.path.join(d, "eval_training_params_02000.json")))
    out = {"D_acq": ps["acq_discrimination"]["mean"], "D_naive": ps["naive_discrimination"]["mean"],
           "OSI": om["OSI"] if om["OSI"] is not None else float("nan"),
           "gap_fraction": ev["gap_fraction"], "gate_pass": float(ev["gate_pass"])}
    out.update({k: v for k, (v, _) in effects(json.load(open(os.path.join(d, "phenomena.json")))).items()})
    return out


def main():
    comp = [measures(f"results/compound/compound_{a:02d}") for a in ANCESTORS]
    stab = [measures(f"results/confirm/stable_{a:02d}") for a in ANCESTORS]
    keys = list(comp[0])
    blk = "blocking: CR(X) blocking - overshadowing"
    res = {"compound": {k: describe([c[k] for c in comp]) for k in keys},
           "stable": {k: describe([s[k] for s in stab]) for k in keys},
           "C1_blocking_in_compound": t_two_sided([c[blk] for c in comp], predicted="less"),
           "C1_compound_minus_stable_blocking": t_two_sided([c[blk] - s[blk] for c, s in zip(comp, stab)],
                                                            predicted="less"),
           "C2_D_acq_compound": describe([c["D_acq"] for c in comp]),
           "C2_OSI_compound": describe([c["OSI"] for c in comp]),
           "per_ancestor": {str(a): {"compound": c, "stable": s} for a, c, s in zip(ANCESTORS, comp, stab)}}
    f = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"  # noqa: E731
    lines = ["# Compound-cue follow-up (exploratory; N = 10 ancestors, paired)", "",
             "| measure | compound sisters | stable sisters |", "|---|---|---|"]
    lines += [f"| {k} | {f(res['compound'][k])} | {f(res['stable'][k])} |" for k in keys]
    for k in ("C1_blocking_in_compound", "C1_compound_minus_stable_blocking"):
        v = res[k]
        lines.append(f"\n{k}: {f(v)}, p = {v['p']:.3g}, direction as predicted: {v['direction_ok']}")
    json.dump(res, open("results/compound/analysis.json", "w"), indent=1, default=float)
    open("results/compound/analysis.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
