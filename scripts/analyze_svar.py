"""S-O variability control analysis (exploratory; predictions P1-P3 in docs/DECISIONS.md 2026-11-18).
Written before any result of results/svar existed.

Per run (pavlov_standard lab, 32 subjects, lab seed 2026), subjects are split by the channel the
bell landed on: channel 0 (the always-predictive channel when q = 0) vs channels 1-3. Per run means:
  naive_cs_plus_cr   CR to the bell before any pairing
  D_naive, D_acq     bell minus metronome, before / after pairing
Then per q level: mean and 95% bootstrap CI over seeds (q = 1 = the 20 confirmatory ancestors).
Writes results/svar/analysis.{json,md}.
"""
import json
import os

import numpy as np

from pavlov.analysis.stats import describe
from pavlov.lab.experiment import LAB_SEED, subject_schedules

LEVELS = {"q = 0": [f"results/svar/q0_{s}" for s in range(130, 135)],
          "q = 0.5": [f"results/svar/q05_{s}" for s in range(135, 140)],
          "q = 1 (confirmatory)": [f"results/confirm/anc_{a:02d}" for a in range(20)]}
KEYS = {"naive_cs_plus_cr": "naive_cs_plus", "D_naive": "naive_discrimination", "D_acq": "acq_discrimination"}


def per_run(run, bell0):
    m = json.load(open(os.path.join(run, "lab", "pavlov_standard.json")))["metrics"]
    ev = json.load(open(os.path.join(run, "eval_training_params_04000.json")))
    out = {"gap_fraction": ev["gap_fraction"], "gate_pass": float(ev["gate_pass"])}
    for k, src in KEYS.items():
        v = np.array(m[src]["per_subject"], float)
        out[f"{k}|bell=ch0"] = float(np.nanmean(v[bell0]))
        out[f"{k}|bell=ch1-3"] = float(np.nanmean(v[~bell0]))
    return out


def main():
    scheds, _ = subject_schedules("pavlov_standard", 32, 4, LAB_SEED)
    bell0 = np.array([s.bell == 0 for s in scheds])
    res, lines = {"n_subjects_bell_ch0": int(bell0.sum())}, []
    for level, runs in LEVELS.items():
        rows = [per_run(r, bell0) for r in runs]
        res[level] = {"runs": runs, "per_run": rows,
                      "summary": {k: describe([r[k] for r in rows]) for k in rows[0]}}
    f = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"  # noqa: E731
    cols = ["naive_cs_plus_cr|bell=ch0", "naive_cs_plus_cr|bell=ch1-3", "D_acq|bell=ch0", "D_acq|bell=ch1-3",
            "gap_fraction", "gate_pass"]
    lines += [f"# S-O variability control (exploratory; {bell0.sum()} of 32 lab subjects have the bell on channel 0)", "",
              "| level | N | " + " | ".join(cols) + " |", "|---" * (len(cols) + 2) + "|"]
    for level in LEVELS:
        s = res[level]["summary"]
        lines.append(f"| {level} | {len(LEVELS[level])} | " + " | ".join(f(s[c]) for c in cols) + " |")
    json.dump(res, open("results/svar/analysis.json", "w"), indent=1)
    open("results/svar/analysis.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
