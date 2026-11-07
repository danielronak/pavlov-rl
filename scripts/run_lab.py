"""Run a frozen trained agent through lab protocols.
    python scripts/run_lab.py --run results/pilot_r4/109_es --protocol all
Writes results/<run>/lab/<protocol>.json (metrics) and .npz (per-tick p, u, food; git-ignored).
"""
import argparse
import json
import os
import pickle

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax
import numpy as np

from pavlov.config import EcologyConfig, PhysiologyConfig
from pavlov.lab.experiment import LAB_SEED, run_protocol
from pavlov.lab.policies import agent_policy
from pavlov.lab.protocols import PROTOCOLS

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--protocol", default="all")
    ap.add_argument("--ckpt", default="params_latest.pkl")
    ap.add_argument("--subjects", type=int, default=32)
    ap.add_argument("--lab-seed", type=int, default=LAB_SEED)
    args = ap.parse_args()
    meta = json.load(open(os.path.join(args.run, "config.json")))
    pc = PhysiologyConfig(**meta["config"]["physiology"])
    nc = EcologyConfig(**meta["config"]["ecology"]).n_channels
    hidden = meta["agent"]["hidden"]
    params = pickle.load(open(os.path.join(args.run, "checkpoints", args.ckpt), "rb"))
    out_dir = os.path.join(args.run, "lab")
    os.makedirs(out_dir, exist_ok=True)
    protos = list(PROTOCOLS) if args.protocol == "all" else [args.protocol]
    for i, name in enumerate(protos):
        res = run_protocol(lambda B: agent_policy(params, hidden, B), name, args.subjects, nc, pc,
                           jax.random.PRNGKey(1000 + i), lab_seed=args.lab_seed, return_raw=True)
        raw = res.pop("raw")
        res.pop("schedules")
        res["ckpt"] = args.ckpt
        json.dump(res, open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1, default=float)
        np.savez_compressed(os.path.join(out_dir, f"{name}.npz"),
                            **{f"{k}_{f}": v[f].astype(np.float16 if f == "p" else bool)
                               for k, v in raw.items() for f in ("p", "u", "food")})
        print(f"[{name}] done -> {out_dir}/{name}.json")
