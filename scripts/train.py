"""Meta-train an agent. Example:
    python scripts/train.py --config configs/exp/pilot.yaml --seed 100 --out results/pilot/100
"""
import argparse
import os
import subprocess

# One XLA thread per run; parallelism comes from running seeds as separate processes.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

from pavlov.agent import AgentConfig
from pavlov.config import load_config
from pavlov.ppo import PPOConfig, train
from pavlov.seeds import check_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lifetime-batches", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--ent-coef", type=float, default=None)
    args = ap.parse_args()
    check_seed(args.seed)
    cfg = load_config(args.config)
    p = PPOConfig()
    kw = {}
    if args.lifetime_batches:
        kw["n_lifetime_batches"] = args.lifetime_batches
    if args.n_envs:
        kw["n_envs"] = args.n_envs
    if args.ent_coef is not None:
        kw["ent_coef"] = args.ent_coef
    p = PPOConfig(**{**p.__dict__, **kw})
    os.makedirs(args.out, exist_ok=True)
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        open(os.path.join(args.out, "git_hash"), "w").write(h + "\n")
    except Exception:
        pass
    train(cfg, AgentConfig(hidden=args.hidden), p, args.seed, args.out)
