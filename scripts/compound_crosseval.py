"""Post-hoc diagnostic for the compound follow-up (exploratory; DECISIONS 2026-11-19, written after C1 failed,
before this script was run). Question: did the compound ecology reward anything the stable sisters lack?
Each sister (compound_XX, stable_XX; ancestors 0-9) is evaluated on held-out lifetimes of BOTH ecologies
(stable = ancestor ecology, compound = + p_compound 0.3), same schedules and action key for every agent.
Reports mean training-ecology return (the only quantity selection acts on) plus oracle references.
Writes results/compound/crosseval.{json,md}.
"""
import json
import os
import pickle
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eval_training import EVAL_KEY, _open_loop_returns, run_agent  # noqa: E402

from pavlov.analysis.stats import describe, t_two_sided  # noqa: E402
from pavlov.config import Config, EcologyConfig, PhysiologyConfig  # noqa: E402
from pavlov.ecology import generate_batch  # noqa: E402
from pavlov.oracles import ORACLES, cue_oracle  # noqa: E402

N = 256


def load(run):
    meta = json.load(open(os.path.join(run, "config.json")))
    cfg = Config(physiology=PhysiologyConfig(**meta["config"]["physiology"]),
                 ecology=EcologyConfig(**meta["config"]["ecology"]))
    return cfg, meta["agent"]["hidden"], pickle.load(open(os.path.join(run, "checkpoints", "params_02000.pkl"), "rb"))


def main():
    worlds = {"stable": load("results/confirm/stable_00")[0], "compound": load("results/compound/compound_00")[0]}
    assert worlds["compound"].ecology.p_compound == 0.3 and worlds["stable"].ecology.p_compound == 0.0
    res = {}
    for wname, cfg in worlds.items():
        sched = generate_batch(jax.random.PRNGKey(EVAL_KEY), cfg.ecology, N)
        pc = cfg.physiology
        ref = {k: float(_open_loop_returns(sched, ORACLES[k](sched, pc), pc).mean())
               for k in ("never", "reflex", "pred_cue_on", "any_cue")}
        ref["cue_oracle"] = max(float(_open_loop_returns(sched, cue_oracle(sched, pc, k), pc).mean())
                                for k in range(1, 11))
        ret = {"compound": [], "stable": []}
        for a in range(10):
            for arm, run in (("compound", f"results/compound/compound_{a:02d}"),
                             ("stable", f"results/confirm/stable_{a:02d}")):
                _, hidden, params = load(run)
                r, _ = run_agent(params, sched, cfg, hidden, jax.random.PRNGKey(1))
                ret[arm].append(float(r.mean()))
        res[wname] = {"oracles": ref, "returns": ret,
                      "compound": describe(ret["compound"]), "stable": describe(ret["stable"]),
                      "compound_minus_stable": t_two_sided([c - s for c, s in zip(ret["compound"], ret["stable"])])}
    f = lambda d: f"{d['mean']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"  # noqa: E731
    lines = ["# Compound follow-up: cross-evaluation (post hoc, exploratory; N = 10 ancestors, paired)", "",
             "| tested in | compound sisters | stable sisters | compound - stable | p | oracles (never / reflex / pred_cue_on / cue_oracle) |",
             "|---|---|---|---|---|---|"]
    for w, v in res.items():
        o = v["oracles"]
        lines.append(f"| {w} ecology | {f(v['compound'])} | {f(v['stable'])} | {f(v['compound_minus_stable'])} | "
                     f"{v['compound_minus_stable']['p']:.3g} | {o['never']:.2f} / {o['reflex']:.2f} / "
                     f"{o['pred_cue_on']:.2f} / {o['cue_oracle']:.2f} |")
    json.dump(res, open("results/compound/crosseval.json", "w"), indent=1, default=float)
    open("results/compound/crosseval.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
