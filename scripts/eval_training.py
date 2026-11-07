"""Phase-2 gate: training-ecology return of a trained agent vs oracles on the SAME held-out lifetimes.

Gate: agent >= reflex + 0.5 * (cue_oracle - reflex). Uses only in-distribution training-ecology
return (I9-safe). The pre-US salivation diagnostic is printed for debugging only and is not part
of the gate.
Usage: python scripts/eval_training.py results/pilot/100 [--ckpt params_latest.pkl]
"""
import argparse
import json
import os
import pickle

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax
import jax.numpy as jnp
import numpy as np

from pavlov import agent as A
from pavlov.config import Config, EcologyConfig, PhysiologyConfig
from pavlov.ecology import generate_batch
from pavlov.rollout import fixed_action_returns, run_lifetimes, run_schedule
from pavlov.oracles import ORACLES, _lead_window, cue_oracle, gate_aware, ro_aware
from pavlov.physiology import run_open_loop

EVAL_KEY = 987654  # held-out schedule key, never used for training


def run_agent(params, sched, cfg: Config, hidden, key):
    r, p = run_lifetimes(params, sched.cues, sched.food_onset, sched.params.flipped, cfg, hidden, key)
    return np.asarray(r), np.asarray(p)


def _open_loop_returns(sched, actions, pc):
    f = jax.vmap(lambda fo, a, fl: run_open_loop(fo, a, fl, pc)["reward"].sum())
    return np.asarray(f(sched.food_onset, actions, sched.params.flipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="params_latest.pkl")
    ap.add_argument("--n", type=int, default=256)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run, "config.json")))
    cfg = Config(physiology=PhysiologyConfig(**meta["config"]["physiology"]),
                 ecology=EcologyConfig(**meta["config"]["ecology"]))
    hidden = meta["agent"]["hidden"]
    params = pickle.load(open(os.path.join(args.run, "checkpoints", args.ckpt), "rb"))
    sched = generate_batch(jax.random.PRNGKey(EVAL_KEY), cfg.ecology, args.n)
    pc = cfg.physiology

    if cfg.ecology.ro_family == "partial_omission":
        r_, p_ = run_schedule(params, sched, cfg, hidden, jax.random.PRNGKey(1))
        agent_r, p = np.asarray(r_), np.asarray(p_)
        oracle_returns = lambda sch, act, pc_: np.asarray(fixed_action_returns(sch, act, cfg))  # noqa: E731
        aware = gate_aware
    else:
        agent_r, p = run_agent(params, sched, cfg, hidden, jax.random.PRNGKey(1))
        oracle_returns = _open_loop_returns
        aware = ro_aware
    res = {"agent": agent_r}
    for name in ("never", "reflex", "pred_cue_on", "any_cue"):
        res[name] = oracle_returns(sched, ORACLES[name](sched, pc), pc)
    sc = {k: oracle_returns(sched, cue_oracle(sched, pc, k), pc) for k in range(1, 11)}
    kb = max(sc, key=lambda k: sc[k].mean())
    res["cue_oracle"] = sc[kb]
    lo_key, hi_key = "reflex", "cue_oracle"
    if cfg.ecology.v_ro_variability > 0:  # instrumental arm: gate against R-O-aware oracles (DECISIONS 2026-11-07)
        res["reflex_aware"] = oracle_returns(sched, aware(ORACLES["reflex"](sched, pc), sched), pc)
        sa = {k: oracle_returns(sched, aware(cue_oracle(sched, pc, k), sched), pc) for k in range(1, 11)}
        res["cue_oracle_aware"] = sa[max(sa, key=lambda k: sa[k].mean())]
        lo_key, hi_key = "reflex_aware", "cue_oracle_aware"

    for name, r in sorted(res.items(), key=lambda kv: -kv[1].mean()):
        print(f"  {name:<12}{r.mean():8.2f}")
    gap = res[hi_key] - res[lo_key]
    frac = (res["agent"] - res[lo_key]) / gap.mean()
    se = frac.std(ddof=1) / np.sqrt(len(frac))
    print(f"  fraction of {lo_key}->{hi_key} gap closed: {frac.mean():+.2f} [{frac.mean()-1.96*se:+.2f}, {frac.mean()+1.96*se:+.2f}]")
    pre = np.asarray(_lead_window(sched.expected_us, 2))
    print(f"  [diagnostic] p(saliva) pre-expected-US={p[pre].mean():.3f}  elsewhere={p[~pre].mean():.3f}")
    ok = frac.mean() >= 0.5
    print(f"  PHASE-2 GATE: {'PASS' if ok else 'FAIL'}")
    out = {k: float(v.mean()) for k, v in res.items()}
    out.update(gap_fraction=float(frac.mean()), gate_pass=bool(ok), ckpt=args.ckpt,
               diag_p_pre=float(p[pre].mean()), diag_p_other=float(p[~pre].mean()))
    json.dump(out, open(os.path.join(args.run, f"eval_training_{args.ckpt.replace('.pkl','')}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()


oracle_returns = _open_loop_returns  # backwards-compatible name for other scripts
