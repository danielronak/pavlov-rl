"""H2'b manipulation check (PREREGISTRATION.md): did within-lifetime R-O learning about the digestion
consequence of salivating evolve?

M = p(saliva | food in mouth, normal lifetimes) - p(saliva | food in mouth, flipped lifetimes),
pooled over ticks LATE_START..end of 256 held-out lifetimes (EVAL_KEY) of the digestion-variable
ecology (configs/confirm/digestion.yaml), whatever ecology the run itself was trained in. Training
ecology only (not the lab). Writes <run>/ro_learning_<ckpt>.json.

Usage: python scripts/ro_learning_check.py results/confirm/anc_00 --ckpt params_04000.pkl
"""
import argparse
import json
import os
import pickle
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax
import numpy as np

from pavlov.config import load_config
from pavlov.ecology import generate_batch
from pavlov.oracles import food_present

sys.path.insert(0, os.path.dirname(__file__))
from eval_training import EVAL_KEY, run_agent  # noqa: E402

DIGESTION_CONFIG = "configs/confirm/digestion.yaml"
LATE_START = 800
N_LIFETIMES = 256


def ro_learning(params, hidden: int) -> dict:
    cfg = load_config(DIGESTION_CONFIG)
    s = generate_batch(jax.random.PRNGKey(EVAL_KEY), cfg.ecology, N_LIFETIMES)
    _, p = run_agent(params, s, cfg, hidden, jax.random.PRNGKey(1))
    food = np.asarray(food_present(s.food_onset, cfg.physiology.food_ticks))
    flipped = np.asarray(s.params.flipped)[:, None]
    late = (np.arange(p.shape[1]) >= LATE_START)[None, :]
    normal_p = float(p[food & late & ~flipped].mean())
    flipped_p = float(p[food & late & flipped].mean())
    early = (np.arange(p.shape[1]) < 200)[None, :]
    return {"p_food_normal_late": normal_p, "p_food_flipped_late": flipped_p, "M": normal_p - flipped_p,
            "p_food_flipped_early": float(p[food & early & flipped].mean()),
            "n_flipped": int(flipped.sum()), "n_lifetimes": N_LIFETIMES, "late_start": LATE_START}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ckpt", default="params_latest.pkl")
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run, "config.json")))
    params = pickle.load(open(os.path.join(args.run, "checkpoints", args.ckpt), "rb"))
    out = ro_learning(params, meta["agent"]["hidden"])
    out["ckpt"] = args.ckpt
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(args.run, f"ro_learning_{args.ckpt.replace('.pkl', '')}.json"), "w"), indent=1)
