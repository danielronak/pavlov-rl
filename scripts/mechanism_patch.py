"""Mechanism v2: activation patching (exploratory; plan in docs/DECISIONS.md 2026-11-18).

Donor = GRU state at the tick before CS onset of the last acquisition CS+ trial, taken from a
DIFFERENT subject whose bell was on the SAME channel ("matched") or on a DIFFERENT channel
("mismatched"; cue-specificity control). Self-donor (same subject) is also reported. Recipient = the first naive CS+ trial of the same subject. The top-k
units (ranked by mean |donor - recipient| over subjects) are overwritten at that tick; the dynamics
then run on unchanged. Writes <run>/mechanism_patch.json.
Usage: python scripts/mechanism_patch.py results/confirm/anc_00 [...]
"""
import json
import os
import pickle
import sys

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pavlov import agent as A  # noqa: E402
from pavlov.config import EcologyConfig, PhysiologyConfig  # noqa: E402
from pavlov.env import env_step, init_env, observe  # noqa: E402
from pavlov.lab.experiment import LAB_SEED, subject_schedules  # noqa: E402
from pavlov.lab.scheduler import CS_PLUS, stack_schedules  # noqa: E402

PRE_US, BASE = 3, 10
KS = (1, 2, 4, 8, 16, 32, 64)


@jax.jit
def _run(params, cues, food, u, donor, unit_mask, tick):
    """Unroll; at tick[b] (after the step) units in unit_mask take donor[b]'s values."""
    B, T, _ = cues.shape
    H = params["gru_h"]["w"].shape[0]

    def body(carry, xs):
        es, h = carry
        t, c, f, a = xs
        obs = observe(es, c, f, PC)
        h, logit, _ = A.step(params, h, obs)
        sel = (t == tick)[:, None] & unit_mask[None, :]
        h = jnp.where(sel, donor, h)
        logit = h @ params["pi"]["w"][:, 0] + params["pi"]["b"][0]
        es, _, _ = env_step(es, f, a, jnp.zeros(B, bool), PC)
        return (es, h), (h, jax.nn.sigmoid(logit))

    xs = (jnp.arange(T), jnp.swapaxes(cues, 0, 1), food.T, u.T)
    _, (h, p) = jax.lax.scan(body, (init_env(B), jnp.zeros((B, H))), xs)
    return jnp.swapaxes(h, 0, 1), p.T


def trial_cr(p, s, i):
    o = s.trial_onset[i]
    return p[o + s.isi - PRE_US:o + s.isi].mean() - p[o - BASE:o].mean()


def analyse(run):
    global PC
    meta = json.load(open(os.path.join(run, "config.json")))
    PC = PhysiologyConfig(**meta["config"]["physiology"])
    nc = EcologyConfig(**meta["config"]["ecology"]).n_channels
    ck = sorted(f for f in os.listdir(os.path.join(run, "checkpoints")) if f.startswith("params_0"))[-1]
    params = jax.tree.map(jnp.asarray, pickle.load(open(os.path.join(run, "checkpoints", ck), "rb")))
    scheds, _ = subject_schedules("pavlov_standard", 32, nc, LAB_SEED)
    st = stack_schedules(scheds)
    z = np.load(os.path.join(run, "lab", "pavlov_standard.npz"))
    cues, food, u = jnp.asarray(st["cues"]), jnp.asarray(z["main_food"]), jnp.asarray(z["main_u"], jnp.float32)
    B, H = len(scheds), params["gru_h"]["w"].shape[0]

    def idx(s, phase, kind):
        ph = s.phases.index(phase)
        return [i for i, (k, tp) in enumerate(zip(s.trial_kind, s.trial_phase)) if k == kind and tp == ph]

    rec_i = [idx(s, "naive", CS_PLUS)[0] for s in scheds]
    last = [idx(s, "acquisition", CS_PLUS)[-1] for s in scheds]
    bells = np.array([s.bell for s in scheds])
    pick = lambda b, same: next(c % B for c in range(b + 1, b + B) if (bells[c % B] == bells[b]) == same)  # noqa: E731
    src = {"self": list(range(B)), "matched": [pick(b, True) for b in range(B)],
           "mismatched": [pick(b, False) for b in range(B)]}
    tick = jnp.array([s.trial_onset[i] - 1 for s, i in zip(scheds, rec_i)])
    none = jnp.zeros(H, bool)
    h0, p0 = map(np.asarray, _run(params, cues, food, u, jnp.zeros((B, H)), none, tick))
    base = np.array([trial_cr(p0[b], s, rec_i[b]) for b, s in enumerate(scheds)])
    donor_cr = np.array([trial_cr(p0[b], s, last[b]) for b, s in enumerate(scheds)])
    recip = np.array([h0[b, s.trial_onset[rec_i[b]] - 1] for b, s in enumerate(scheds)])
    out = {"run": run, "ckpt": ck, "naive_cr": float(base.mean()), "donor_trial_cr": float(donor_cr.mean()),
           "patched_cr": {}, "transfer_fraction": {}}
    for name, sb in src.items():
        donor = np.array([h0[sb[b], scheds[sb[b]].trial_onset[last[sb[b]]] - 1] for b in range(B)])
        order = np.argsort(-np.abs(donor - recip).mean(0))
        out["patched_cr"][name], out["transfer_fraction"][name] = {}, {}
        for k in KS:
            mask = np.zeros(H, bool)
            mask[order[:k]] = True
            _, pk = _run(params, cues, food, u, jnp.asarray(donor), jnp.asarray(mask), tick)
            pk = np.asarray(pk)
            c = np.array([trial_cr(pk[b], s, rec_i[b]) for b, s in enumerate(scheds)])
            out["patched_cr"][name][str(k)] = float(c.mean())
            out["transfer_fraction"][name][str(k)] = float((c.mean() - base.mean()) / (donor_cr.mean() - base.mean()))
    json.dump(out, open(os.path.join(run, "mechanism_patch.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    for run in sys.argv[1:]:
        o = analyse(run)
        print(run, round(o["naive_cr"], 3), round(o["donor_trial_cr"], 3),
              {n: {k: round(v, 2) for k, v in d.items()} for n, d in o["transfer_fraction"].items()}, flush=True)
