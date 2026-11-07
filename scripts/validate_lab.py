"""Phase-3 gate: the lab measurement pipeline must give the expected signatures for policies
whose behaviour is known by construction. No trained agent is involved.

Expected:
  never       : every CR = 0; OSI undefined (yoked CR = 0)
  bell_reader : naive CS+ CR ≈ 0.9 (the naive check catches a pre-wired bell response), CS- ≈ 0,
                no extinction, no recovery change, OSI ≈ 0, omission delivers ≈ 0 food,
                truly-random / CS-alone / US-alone test CR ≈ 0.9 (responds regardless of contingency)
  withholder  : naive CS+ CR = 0, OSI ≈ 1, positive contingency delivers ≈ all food
"""
import json
import os

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax
import numpy as np

from pavlov.config import PhysiologyConfig
from pavlov.lab.experiment import run_protocol
from pavlov.lab.policies import oracle_policy
from pavlov.lab.protocols import PROTOCOLS

N = 32
NC = 4
PC = PhysiologyConfig()


def m(res, key, src="metrics"):
    return res[src][key]["mean"]


def main():
    results, checks = {}, []

    def check(name, ok, val):
        checks.append((name, bool(ok), val))

    for oracle in ("never", "bell_reader", "withholder"):
        results[oracle] = {}
        for proto in PROTOCOLS:
            r = run_protocol(lambda B, o=oracle: oracle_policy(o, B), proto, N, NC, PC, jax.random.PRNGKey(0))
            results[oracle][proto] = r

    R = results
    # never
    std = R["never"]["pavlov_standard"]["metrics"]
    for k in ("naive_cs_plus", "acq_cs_plus_last10", "acq_discrimination", "recovery_first"):
        check(f"never: {k} == 0", abs(std[k]["mean"]) < 1e-9, std[k]["mean"])
    check("never: OSI undefined", np.isnan(R["never"]["omission"]["OSI"]), R["never"]["omission"]["OSI"])
    # bell_reader
    b = R["bell_reader"]
    check("bell_reader: naive CS+ ≈ 0.9 (naive check detects pre-wired response)",
          abs(m(b["pavlov_standard"], "naive_cs_plus") - 0.9) < 0.02, m(b["pavlov_standard"], "naive_cs_plus"))
    check("bell_reader: naive CS- ≈ 0", abs(m(b["pavlov_standard"], "naive_cs_minus")) < 0.02,
          m(b["pavlov_standard"], "naive_cs_minus"))
    check("bell_reader: acquisition discrimination ≈ 0.9",
          abs(m(b["pavlov_standard"], "acq_discrimination") - 0.9) < 0.02, m(b["pavlov_standard"], "acq_discrimination"))
    check("bell_reader: no extinction (ext CR ≈ 0.9)", abs(m(b["pavlov_standard"], "ext_cs_plus_last5") - 0.9) < 0.02,
          m(b["pavlov_standard"], "ext_cs_plus_last5"))
    check("bell_reader: OSI ≈ 0", b["omission"]["OSI"] < 0.05, b["omission"]["OSI"])
    check("bell_reader: omission delivers ≈ 0 food", m(b["omission"], "us_delivered_fraction") < 0.05,
          m(b["omission"], "us_delivered_fraction"))
    for proto in ("truly_random", "cs_alone", "us_alone"):
        check(f"bell_reader: {proto} test CR ≈ 0.9", abs(m(b[proto], "test_cs_plus") - 0.9) < 0.02,
              m(b[proto], "test_cs_plus"))
    # withholder
    w = R["withholder"]
    check("withholder: naive CS+ == 0", abs(m(w["omission"], "naive_cs_plus")) < 1e-9, m(w["omission"], "naive_cs_plus"))
    check("withholder: OSI ≈ 1", w["omission"]["OSI"] > 0.95, w["omission"]["OSI"])
    check("withholder: omission delivers all food", m(w["omission"], "us_delivered_fraction") > 0.99,
          m(w["omission"], "us_delivered_fraction"))
    check("withholder: positive contingency delivers ≈ all food", m(w["positive"], "us_delivered_fraction") > 0.95,
          m(w["positive"], "us_delivered_fraction"))
    # schedule sanity: lab length vs training lifetime
    from pavlov.lab.experiment import subject_schedules
    lengths = [s.length for s in subject_schedules("pavlov_standard", N, NC)[0]]
    check("pavlov_standard length within 10% of the 3000-tick lifetime", max(lengths) <= 3300, max(lengths))

    print(f"{'check':<72}{'value':>10}  result")
    for name, ok, val in checks:
        print(f"{name:<72}{val:>10.4f}  {'PASS' if ok else 'FAIL'}")
    ok_all = all(c[1] for c in checks)
    print(f"\nPHASE-3 GATE: {'PASS' if ok_all else 'FAIL'}")
    os.makedirs("results/phase3", exist_ok=True)

    def strip(r):
        return {k: v for k, v in r.items() if k not in ("raw", "schedules")}
    json.dump({"checks": [{"name": n, "pass": o, "value": float(v)} for n, o, v in checks], "pass": ok_all,
               "results": {o: {p: strip(r) for p, r in d.items()} for o, d in R.items()}},
              open("results/phase3/lab_validation.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
