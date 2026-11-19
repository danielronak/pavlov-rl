"""Robustness check analysis (exploratory; criteria R1-R3 in docs/DECISIONS.md 2026-11-19).
Written before any robustness result existed. Per variant (3 seeds, intention-to-treat: every seed is
included whether or not it passes the Phase-2 gate) report per-seed values, the mean and the range,
next to the 20 confirmatory ancestors. N = 3 is too small for tests: no p-values, criteria on means.
  R1 conditioning to a novel bell: mean D_acq > 0.3 and |mean D_naive| < 0.1
  R2 contingency, not exposure: mean (paired - truly random) test CR > 0.1
  R3 omission-insensitive: mean OSI < 0.2 over seeds with a defined OSI (count reported)
A variant replicates the core result if R1-R3 all hold.
Writes results/robust/analysis.{json,md}.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_robust import ROOT, RUNS, VARIANTS  # noqa: E402

KEYS = ["gap_fraction", "gate_pass", "D_naive", "D_acq", "paired_minus_random", "extinction_drop", "OSI",
        "US_delivered_omission"]


def measures(d, gens=4000):
    lab = {p: json.load(open(os.path.join(d, "lab", f"{p}.json")))
           for p in ("pavlov_standard", "truly_random", "omission")}
    ev = json.load(open(os.path.join(d, f"eval_training_params_{gens:05d}.json")))
    ps = lambda k: lab["pavlov_standard"]["metrics"][k]["mean"]  # noqa: E731
    om = lab["omission"]
    return {"gap_fraction": ev["gap_fraction"], "gate_pass": float(ev["gate_pass"]),
            "D_naive": ps("naive_discrimination"), "D_acq": ps("acq_discrimination"),
            "paired_minus_random": ps("ext_cs_plus_first5") - lab["truly_random"]["metrics"]["test_cs_plus"]["mean"],
            "extinction_drop": ps("acq_cs_plus_last10") - ps("ext_cs_plus_last5"),
            "OSI": om["OSI"] if om["OSI"] is not None else float("nan"),
            "US_delivered_omission": om["metrics"]["us_delivered_fraction"]["mean"]}


def summary(rows):
    out = {}
    for k in KEYS:
        x = np.array([r[k] for r in rows], float)
        f = x[np.isfinite(x)]
        out[k] = {"values": x.tolist(), "n_defined": int(len(f)),
                  "mean": float(f.mean()) if len(f) else float("nan"),
                  "min": float(f.min()) if len(f) else float("nan"), "max": float(f.max()) if len(f) else float("nan")}
    return out


def criteria(s):
    return {"R1": bool(s["D_acq"]["mean"] > 0.3 and abs(s["D_naive"]["mean"]) < 0.1),
            "R2": bool(s["paired_minus_random"]["mean"] > 0.1),
            "R3": bool(s["OSI"]["n_defined"] > 0 and s["OSI"]["mean"] < 0.2)}


def main():
    res = {"reference_confirmatory_ancestors": summary(
        [measures(f"results/confirm/anc_{a:02d}") for a in range(20)])}
    for v, _, hidden, _ in VARIANTS:
        rows = [measures(os.path.join(ROOT, name)) for name, _, h, _ in RUNS if name.startswith(v + "_")]
        s = summary(rows)
        res[v] = {"hidden": hidden, "summary": s, "criteria": criteria(s)}
        res[v]["replicates"] = all(res[v]["criteria"].values())
    f = lambda d: f"{d['mean']:+.3f} ({d['min']:+.3f} to {d['max']:+.3f})"  # noqa: E731
    cols = ["reference_confirmatory_ancestors"] + [v for v, *_ in VARIANTS]
    lines = ["# Robustness check (exploratory; 3 seeds per variant, mean (min to max); reference = 20 ancestors)",
             "", "| measure | " + " | ".join(c.replace("reference_confirmatory_ancestors", "ancestors (N=20)")
                                           for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for k in KEYS:
        cells = [f(res[c][k] if c.startswith("reference") else res[c]["summary"][k]) for c in cols]
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines.append("")
    for v, *_ in VARIANTS:
        c = res[v]["criteria"]
        lines.append(f"- {v}: " + ", ".join(f"{k} {'met' if ok else 'NOT met'}" for k, ok in c.items())
                     + f" -> {'replicates' if res[v]['replicates'] else 'does NOT replicate'}"
                     + f" (OSI defined in {res[v]['summary']['OSI']['n_defined']}/3)")
    os.makedirs(ROOT, exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "analysis.json"), "w"), indent=1)
    open(os.path.join(ROOT, "analysis.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
