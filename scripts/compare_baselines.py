"""Scorecard: evolved agents vs baseline learners vs the direction animals show (exploratory;
plan in docs/DECISIONS.md 2026-11-19). Agents: mean and 95% CI over the 20 confirmatory ancestors.
Writes results/baselines/scorecard.{json,md}.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_phenomena import effects  # noqa: E402

from pavlov.analysis.stats import describe  # noqa: E402

MODELS = {"rw": "Rescorla–Wagner", "td": "TD (serial compound)", "q": "Instrumental Q-learner"}


def lab_measures(lab):
    ps = lambda k: lab["pavlov_standard"]["metrics"][k]["mean"]  # noqa: E731
    om, pos = lab["omission"], lab["positive"]
    return {
        "Acquisition: bell − metronome": (ps("acq_discrimination"), "> 0"),
        "Naive (before pairing): bell − metronome": (ps("naive_discrimination"), "≈ 0"),
        "Contingency: paired − truly random": (ps("ext_cs_plus_first5") - lab["truly_random"]["metrics"]["test_cs_plus"]["mean"], "> 0"),
        "Not sensitisation: paired − US alone": (ps("ext_cs_plus_first5") - lab["us_alone"]["metrics"]["test_cs_plus"]["mean"], "> 0"),
        "Extinction: CR end of extinction / end of acquisition": (ps("ext_cs_plus_last5") / ps("acq_cs_plus_last10"), "< 1"),
        "Spontaneous recovery: after rest − end of extinction": (ps("recovery_first") - ps("ext_cs_plus_last5"), "> 0"),
        "Omission: OSI (0 = ignores, 1 = stops)": (om["OSI"] if om["OSI"] is not None else float("nan"), "≈ 0 (Sheffield)"),
        "Positive contingency: CR − yoked": (pos["positive_index"], "≈ 0"),
    }


def load_lab(d):
    return {p: json.load(open(os.path.join(d, "lab", f"{p}.json")))
            for p in ("pavlov_standard", "truly_random", "us_alone", "omission", "positive")}


def all_measures(d):
    m = lab_measures(load_lab(d))
    ph = json.load(open(os.path.join(d, "phenomena.json")))
    animal = {"blocking": "< 0", "latent inhibition": "< 0", "PREE": "> 0", "inhibition of delay": "> 0", "second-order": "> 0"}
    for k, (v, _) in effects(ph).items():
        m[k] = (v, next(a for n, a in animal.items() if k.startswith(n)))
    return m


def main():
    anc = [all_measures(f"results/confirm/anc_{a:02d}") for a in range(20)]
    keys = list(anc[0])
    out = {"agents": {k: describe([x[k][0] for x in anc]) for k in keys},
           "animal": {k: anc[0][k][1] for k in keys}}
    for m in MODELS:
        out[m] = {k: v for k, (v, _) in all_measures(f"results/baselines/{m}").items()}
    lines = ["# Scorecard: evolved agents vs baseline learners (exploratory)", "",
             "| measure | animals | agents (N = 20) | " + " | ".join(MODELS.values()) + " |",
             "|---" * (3 + len(MODELS)) + "|"]
    for k in keys:
        a = out["agents"][k]
        lines.append(f"| {k} | {out['animal'][k]} | {a['mean']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | "
                     + " | ".join(f"{out[m][k]:+.3f}" for m in MODELS) + " |")
    json.dump(out, open("results/baselines/scorecard.json", "w"), indent=1, default=float)
    open("results/baselines/scorecard.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
