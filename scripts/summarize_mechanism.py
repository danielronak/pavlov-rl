"""Aggregate <run>/mechanism.json over seeds (exploratory; plan in DECISIONS 2026-11-18).
Writes results/confirm/mechanism_summary.{json,md}. Unit of analysis = seed; 95% bootstrap CIs."""
import json
import os

import numpy as np

from pavlov.analysis.stats import describe

ROOT = "results/confirm"
ALPHAS = ("-2.0", "-1.0", "0.0", "1.0", "2.0")


def main():
    out, lines = {}, ["# Mechanism analysis v1 (exploratory)", ""]
    for arm, prefix in (("ancestor", "anc_"), ("stable", "stable_")):
        runs = sorted(n for n in os.listdir(ROOT) if n.startswith(prefix)
                      and os.path.exists(os.path.join(ROOT, n, "mechanism.json")))
        M = [json.load(open(os.path.join(ROOT, n, "mechanism.json"))) for n in runs]
        col = lambda f: np.array([f(m) for m in M], float)  # noqa: E731
        res = {
            "n": len(M),
            "reconstruction_max_abs_err": float(col(lambda m: m["reconstruction_max_abs_err"]).max()),
            "M1_probe_r2_heldout": describe(col(lambda m: m["M1_probe_r2_heldout"])),
            "M2_cos_probe_readout": describe(col(lambda m: m["M2_cos_probe_readout"])),
            "M3_probe": {a: describe(col(lambda m: m["M3_naive_cs_plus_cr_by_alpha"]["probe"][a])) for a in ALPHAS},
            "M3_random": {a: describe(col(lambda m: m["M3_naive_cs_plus_cr_by_alpha"]["random_orthogonal"][a]))
                          for a in ALPHAS},
            "M4_omission_minus_yoked": describe(col(lambda m: m["M4_expectation_omission"] - m["M4_expectation_yoked"])),
            "M4_omission": describe(col(lambda m: m["M4_expectation_omission"])),
            "M4_yoked": describe(col(lambda m: m["M4_expectation_yoked"])),
        }
        out[arm] = res
        f = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"  # noqa: E731
        lines += [f"## {arm} (N = {res['n']}; reconstruction error ≤ {res['reconstruction_max_abs_err']:.1e})", "",
                  f"- M1 held-out R² of the expectation probe: {f(res['M1_probe_r2_heldout'])}",
                  f"- M2 cosine(probe direction, salivation readout): {f(res['M2_cos_probe_readout'])}",
                  "- M3 naive CS+ CR when steering by α·sd: "
                  + ", ".join(f"α={a}: probe {res['M3_probe'][a]['mean']:+.3f} / random {res['M3_random'][a]['mean']:+.3f}"
                              for a in ALPHAS),
                  f"- M4 decoded expectation, omission {f(res['M4_omission'])} vs yoked {f(res['M4_yoked'])}; "
                  f"difference {f(res['M4_omission_minus_yoked'])}", ""]
    json.dump(out, open(os.path.join(ROOT, "mechanism_summary.json"), "w"), indent=1)
    open(os.path.join(ROOT, "mechanism_summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
