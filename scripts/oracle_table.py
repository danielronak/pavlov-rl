"""Phase-1 gate: does anticipatory salivation pay in this physiology + ecology?

Gate criteria (fixed before any agent exists, docs/IMPLEMENTATION_PLAN.md Phase 1):
  G1 reflex > never              (the unconditioned response is worth having)
  G2 cue_oracle > reflex         (anticipation pays)
  G3 cue_oracle > any_cue        (knowing WHICH cue predicts food pays -> association needed)
  G4 always < reflex             (constant salivation is not a shortcut)
  G5 pred_cue_on > reflex        (learnability: a monotone path reflex -> cue identity -> timing; amendment A1)
  Margins for G2, G3, G5: >= 10% of the (cue_oracle - reflex) gap, and the 95% CI of the paired
  difference excludes 0.
"""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np

from pavlov.config import Config, load_config
from pavlov.ecology import generate_batch
from pavlov.oracles import ORACLES, cue_oracle, prophet
from pavlov.physiology import run_open_loop


def lifetime_returns(sched, actions, cfg):
    f = jax.vmap(lambda fo, a, fl: run_open_loop(fo, a, fl, cfg.physiology)["reward"].sum())
    return np.asarray(f(sched.food_onset, actions, sched.params.flipped))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else Config()

    sched = generate_batch(jax.random.PRNGKey(args.seed), cfg.ecology, args.n)
    pc = cfg.physiology
    rets = {}
    for name, fn in ORACLES.items():
        if name in ("cue_oracle", "prophet"):
            continue
        rets[name] = lifetime_returns(sched, fn(sched, pc), cfg)
    # best lead k for the anticipatory oracles (k is a property of the physiology, chosen here)
    best = {}
    for name, fn in (("cue_oracle", cue_oracle), ("prophet", prophet)):
        scores = {k: lifetime_returns(sched, fn(sched, pc, k), cfg) for k in range(1, 9)}
        kbest = max(scores, key=lambda k: scores[k].mean())
        rets[name] = scores[kbest]
        best[name] = kbest

    n_food = float(np.asarray(sched.food_onset).sum(-1).mean())
    print(f"lifetime={cfg.ecology.lifetime}  lifetimes={args.n}  mean food events/lifetime={n_food:.1f}")
    print(f"best lead k: {best}")
    print(f"{'policy':<14}{'mean':>10}{'sd':>9}")
    for name, r in sorted(rets.items(), key=lambda kv: -kv[1].mean()):
        print(f"{name:<14}{r.mean():>10.2f}{r.std():>9.2f}")

    def paired(a, b):
        d = rets[a] - rets[b]
        se = d.std(ddof=1) / np.sqrt(len(d))
        return float(d.mean()), float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)

    gap = rets["cue_oracle"].mean() - rets["reflex"].mean()
    checks = {
        "G1 reflex>never": paired("reflex", "never"),
        "G2 cue_oracle>reflex": paired("cue_oracle", "reflex"),
        "G3 cue_oracle>any_cue": paired("cue_oracle", "any_cue"),
        "G4 reflex>always": paired("reflex", "always"),
        "G5 pred_cue_on>reflex": paired("pred_cue_on", "reflex"),
        "G8 pred_cue_on>any_cue": paired("pred_cue_on", "any_cue"),
    }
    print("\nGate checks (mean diff, 95% CI):")
    ok_all = True
    for name, (m, lo, hi) in checks.items():
        need_margin = name.startswith(("G2", "G3", "G5"))
        margin = 0.25 if name.startswith("G8") else 0.10
        ok = lo > 0 and ((not need_margin and not name.startswith("G8")) or m >= margin * gap)
        ok_all &= ok
        print(f"  {name:<24} {m:+8.2f}  [{lo:+.2f}, {hi:+.2f}]  {'PASS' if ok else 'FAIL'}")
    cues = np.asarray(sched.cues)
    n_on = cues.sum(-1)
    overlap = float(n_on[n_on > 0].mean())
    pairings = float(np.asarray(sched.expected_us).sum(-1).mean())
    g6, g7 = overlap <= 1.05, pairings >= 30
    print(f"  G6 overlap (mean #cues on | any on) = {overlap:.3f} (<= 1.05)  {'PASS' if g6 else 'FAIL'}")
    print(f"  G7 predictive-cue onsets/lifetime = {pairings:.1f} (>= 30)  {'PASS' if g7 else 'FAIL'}")
    ok_all = ok_all and g6 and g7
    checks["G6 overlap"] = overlap
    checks["G7 pairings"] = pairings
    print(f"\nGATE: {'PASS' if ok_all else 'FAIL'}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"config": cfg.to_dict(), "means": {k: float(v.mean()) for k, v in rets.items()},
                       "best_k": best, "checks": checks, "pass": bool(ok_all)}, f, indent=2)


if __name__ == "__main__":
    main()
