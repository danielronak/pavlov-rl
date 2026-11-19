"""Run the baseline learners (src/pavlov/baselines/models.py) through the identical lab pipeline:
all main protocols (with yoked partners) and the phenomena battery. Plan: DECISIONS 2026-11-19.
Writes results/baselines/<model>/lab/<protocol>.json and results/baselines/<model>/phenomena.json.
"""
import json
import os
import subprocess
import sys

import jax

sys.path.insert(0, os.path.dirname(__file__))
from run_phenomena import N_SUBJECTS, battery  # noqa: E402

from pavlov.baselines.models import MODELS, make_policy  # noqa: E402
from pavlov.config import EcologyConfig, PhysiologyConfig  # noqa: E402
from pavlov.lab.experiment import LAB_SEED, run_protocol  # noqa: E402
from pavlov.lab.protocols import PROTOCOLS  # noqa: E402

if __name__ == "__main__":
    pc, nc = PhysiologyConfig(), EcologyConfig().n_channels
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout
    for model in (sys.argv[1:] or MODELS):
        d = os.path.join("results/baselines", model)
        os.makedirs(os.path.join(d, "lab"), exist_ok=True)
        open(os.path.join(d, "git_hash"), "w").write(head)
        mp = make_policy(model, nc)
        for i, name in enumerate(PROTOCOLS):  # same keys as scripts/run_lab.py
            res = run_protocol(mp, name, N_SUBJECTS, nc, pc, jax.random.PRNGKey(1000 + i), lab_seed=LAB_SEED)
            res["model"] = model
            json.dump(res, open(os.path.join(d, "lab", f"{name}.json"), "w"), indent=1, default=float)
        json.dump({"model": model, **battery(mp, pc, nc)}, open(os.path.join(d, "phenomena.json"), "w"), indent=1)
        print(model, "done", flush=True)
