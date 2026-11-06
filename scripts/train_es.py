"""Evolve an agent with OpenAI-ES. Example:
    python scripts/train_es.py --config configs/exp/pilot_a1.yaml --seed 106 --out results/pilot_r3/106_es
"""
import argparse
import os
import subprocess

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

from pavlov.agent import AgentConfig
from pavlov.config import load_config
from pavlov.es import ESConfig, train_es
from pavlov.seeds import check_seed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--pop", type=int, default=None)
    ap.add_argument("--lifetimes", type=int, default=None)
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    check_seed(args.seed)
    cfg = load_config(args.config)
    e = ESConfig()
    kw = {k: v for k, v in (("generations", args.generations), ("pop", args.pop),
                            ("lifetimes", args.lifetimes)) if v is not None}
    e = ESConfig(**{**e.__dict__, **kw})
    os.makedirs(args.out, exist_ok=True)
    h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    open(os.path.join(args.out, "git_hash"), "w").write(h + "\n")
    train_es(cfg, AgentConfig(hidden=args.hidden), e, args.seed, args.out)
