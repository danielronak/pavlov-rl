"""Run the phenomena battery on frozen agents (designs: src/pavlov/lab/phenomena.py; plan in
docs/DECISIONS.md 2026-11-19). Writes <run>/phenomena.json.
Usage: python scripts/run_phenomena.py results/confirm/anc_00 [...]
"""
import json
import os
import pickle
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax  # noqa: E402
import numpy as np  # noqa: E402

from pavlov.config import EcologyConfig, PhysiologyConfig  # noqa: E402
from pavlov.lab import phenomena as P  # noqa: E402
from pavlov.lab.experiment import LAB_SEED  # noqa: E402
from pavlov.lab.policies import agent_policy  # noqa: E402
from pavlov.lab.runner import run_lab  # noqa: E402
from pavlov.lab.scheduler import stack_schedules  # noqa: E402

N_SUBJECTS = 32


def group_metrics(design, p, scheds):
    crs = [P.trial_crs(p[b], s) for b, s in enumerate(scheds)]
    m = lambda tag, sl=slice(None): np.array([np.mean(P.tagged(c, s, tag, sl)) for c, s in zip(crs, scheds)])  # noqa: E731
    out = {}
    if design == "blocking":
        out["test_X"] = m("test_X")
        out["pretrain_last5"] = np.array([np.mean([c[i] for i, t in enumerate(s.trials) if t.phase == "pretrain"][-5:])
                                          for c, s in zip(crs, scheds)])
    elif design == "latent_inhibition":
        out["acq_A"] = m("acq_A")
    elif design == "pree":
        out["acq_last5"], out["ext_all"] = m("acq_A", slice(-5, None)), m("ext_A")
    elif design == "inhibition_of_delay":
        out["slope_last10"] = np.array([P.cs_slope(p[b], s, "acq_A") for b, s in enumerate(scheds)])
        out["cr_last10"] = m("acq_A", slice(-10, None))
    elif design == "second_order":
        out["test_B"], out["first_order_last5"] = m("test_B"), m("acq_A", slice(15, 20))
    curve = np.nanmean(np.array([c for c in crs]), axis=0)
    return {k: float(v.mean()) for k, v in out.items()}, curve.tolist(), [t.tag for t in scheds[0].trials]


def run(run_dir):
    meta = json.load(open(os.path.join(run_dir, "config.json")))
    pc = PhysiologyConfig(**meta["config"]["physiology"])
    nc = EcologyConfig(**meta["config"]["ecology"]).n_channels
    ck = sorted(f for f in os.listdir(os.path.join(run_dir, "checkpoints")) if f.startswith("params_0"))[-1]
    params = pickle.load(open(os.path.join(run_dir, "checkpoints", ck), "rb"))
    res = {"run": run_dir, "ckpt": ck, **battery(lambda B: agent_policy(params, meta["agent"]["hidden"], B), pc, nc)}
    json.dump(res, open(os.path.join(run_dir, "phenomena.json"), "w"), indent=1)
    return res


def battery(make_policy, pc, nc):
    """Run every design and group with policies from make_policy(B) -> (step, carry)."""
    res = {"n_subjects": N_SUBJECTS, "lab_seed": LAB_SEED, "designs": {}}
    for di, design in enumerate(P.DESIGNS):
        res["designs"][design] = {}
        for gi, (group, scheds) in enumerate(P.schedules(design, N_SUBJECTS, nc, LAB_SEED).items()):
            step, carry = make_policy(N_SUBJECTS)
            out = run_lab(step, carry, stack_schedules(scheds), "none", pc, jax.random.PRNGKey(2000 + 10 * di + gi))
            metrics, curve, tags = group_metrics(design, out["p"], scheds)
            res["designs"][design][group] = {"metrics": metrics, "curve": curve, "tags": tags}
    return res


if __name__ == "__main__":
    for r in sys.argv[1:]:
        o = run(r)
        print(r, {d: {g: {k: round(v, 3) for k, v in x["metrics"].items()} for g, x in gs.items()}
                  for d, gs in o["designs"].items()}, flush=True)
