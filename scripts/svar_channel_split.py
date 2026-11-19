"""POST-HOC split of the S-O variability control by bell channel (docs/DECISIONS.md 2026-11-18).

Added after seeing results/svar/analysis.md: prediction P1 lumped channels 1-3 together, overlooking
that channel 1 is the second predictive cue in half of all lifetimes (p_two_predictive = 0.5) even
when q = 0. Channels 2-3 never predict food when q = 0. Writes results/svar/channel_split.json.
"""
import json

import numpy as np

from pavlov.analysis.stats import describe
from pavlov.lab.experiment import LAB_SEED, subject_schedules

LEVELS = {"0": [f"results/svar/q0_{s}" for s in range(130, 135)],
          "0.5": [f"results/svar/q05_{s}" for s in range(135, 140)],
          "1": [f"results/confirm/anc_{a:02d}" for a in range(20)]}
GROUPS = {"ch0": (0,), "ch1": (1,), "ch2-3": (2, 3)}


def main():
    scheds, _ = subject_schedules("pavlov_standard", 32, 4, LAB_SEED)
    bell = np.array([s.bell for s in scheds])
    out = {"n_subjects": {g: int(np.isin(bell, chs).sum()) for g, chs in GROUPS.items()}}
    for q, runs in LEVELS.items():
        ms = [json.load(open(f"{r}/lab/pavlov_standard.json"))["metrics"] for r in runs]
        out[q] = {}
        for g, chs in GROUPS.items():
            m = np.isin(bell, chs)
            for key, src in (("naive_cs_plus_cr", "naive_cs_plus"), ("D_acq", "acq_discrimination")):
                vals = [float(np.nanmean(np.array(x[src]["per_subject"], float)[m])) for x in ms]
                out[q][f"{key}|{g}"] = {**describe(vals), "per_run": vals}
    json.dump(out, open("results/svar/channel_split.json", "w"), indent=1)
    for q in LEVELS:
        print(f"q = {q}: " + "; ".join(f"{k} {v['mean']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                                      for k, v in out[q].items()))


if __name__ == "__main__":
    main()
