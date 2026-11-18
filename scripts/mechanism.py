"""Mechanism analysis v1 (Phase 10, exploratory; plan in docs/DECISIONS.md 2026-11-18).

For one run: reconstruct GRU states for the pavlov_standard and omission lab runs, then
  M1 expectation probe (ridge, held-out subjects), M2 alignment with the salivation readout,
  M3 causal steering along the probe direction in naive CS+ trials, M4 omission vs yoked.
Writes <run>/mechanism.json.   Usage: python scripts/mechanism.py results/confirm/anc_00 [...]
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
from pavlov.lab.scheduler import CS_MINUS, CS_PLUS, stack_schedules  # noqa: E402

PRE_US, BASE = 3, 10
ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
RIDGE = 1.0


def reconstruct(params, pc, cues, food, u, steer=None):
    """Re-run the frozen agent on recorded food/actions. steer = (vector [H], mask [B, T]) adds
    vector * mask[b, t] to the hidden state after each step (the dynamics then carry it on).
    Returns h [B, T, H] and p [B, T]."""
    B, T, _ = cues.shape
    H = params["gru_h"]["w"].shape[0]
    vec, mask = steer if steer is not None else (jnp.zeros(H), jnp.zeros((B, T)))

    def body(carry, xs):
        es, h = carry
        c, f, a, m = xs
        obs = observe(es, c, f, pc)
        h, logit, _ = A.step(params, h, obs)
        h = h + m[:, None] * vec[None, :]
        logit = h @ params["pi"]["w"][:, 0] + params["pi"]["b"][0]
        es, _, _ = env_step(es, f, a, jnp.zeros(B, bool), pc)
        return (es, h), (h, jax.nn.sigmoid(logit))

    xs = (jnp.swapaxes(jnp.asarray(cues), 0, 1), jnp.asarray(food).T, jnp.asarray(u, jnp.float32).T,
          jnp.asarray(mask, jnp.float32).T)
    _, (h, p) = jax.lax.scan(body, (init_env(B), jnp.zeros((B, H))), xs)
    return np.asarray(jnp.swapaxes(h, 0, 1)), np.asarray(p.T)


def expectation_targets(s, food):
    """Per trial: V = fraction of earlier presentations of the same cue followed by food."""
    seen, fed, V = {CS_PLUS: 0, CS_MINUS: 0}, {CS_PLUS: 0, CS_MINUS: 0}, []
    for o, k in zip(s.trial_onset, s.trial_kind):
        if k in (CS_PLUS, CS_MINUS):
            V.append(fed[k] / seen[k] if seen[k] else 0.0)
            seen[k] += 1
            fed[k] += int(food[o + s.isi])
        else:
            V.append(np.nan)
    return np.array(V)


def pre_us_rows(h, scheds, food, keep=lambda s, i: True):
    """Mean hidden state over the 3 pre-US ticks of every CS trial; with targets and subject ids."""
    X, y, subj = [], [], []
    for b, s in enumerate(scheds):
        V = expectation_targets(s, food[b])
        for i, (o, k) in enumerate(zip(s.trial_onset, s.trial_kind)):
            if k in (CS_PLUS, CS_MINUS) and keep(s, i):
                X.append(h[b, o + s.isi - PRE_US:o + s.isi].mean(0))
                y.append(V[i])
                subj.append(b)
    return np.array(X), np.array(y), np.array(subj)


def ridge(X, y):
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    w = np.linalg.solve(Xc.T @ Xc + RIDGE * np.eye(X.shape[1]), Xc.T @ (y - ym))
    return w, mu, ym


def cr(p, scheds, kind, phase):
    out = []
    for b, s in enumerate(scheds):
        ph = s.phases.index(phase)
        for o, k, tp in zip(s.trial_onset, s.trial_kind, s.trial_phase):
            if k == kind and tp == ph:
                out.append(p[b, o + s.isi - PRE_US:o + s.isi].mean() - p[b, o - BASE:o].mean())
    return float(np.mean(out))


def cs_mask(scheds, T, kind, phase):
    m = np.zeros((len(scheds), T))
    for b, s in enumerate(scheds):
        ph = s.phases.index(phase)
        for o, k, tp in zip(s.trial_onset, s.trial_kind, s.trial_phase):
            if k == kind and tp == ph:
                m[b, o:o + s.isi] = 1.0
    return m


def analyse(run):
    meta = json.load(open(os.path.join(run, "config.json")))
    pc = PhysiologyConfig(**meta["config"]["physiology"])
    nc = EcologyConfig(**meta["config"]["ecology"]).n_channels
    ck = sorted(f for f in os.listdir(os.path.join(run, "checkpoints")) if f.startswith("params_0"))[-1]
    params = jax.tree.map(jnp.asarray, pickle.load(open(os.path.join(run, "checkpoints", ck), "rb")))
    out = {"run": run, "ckpt": ck}

    # ---- pavlov_standard: reconstruct, probe, steer
    scheds, _ = subject_schedules("pavlov_standard", 32, nc, LAB_SEED)
    st = stack_schedules(scheds)
    z = np.load(os.path.join(run, "lab", "pavlov_standard.npz"))
    food, u, p_rec = z["main_food"], z["main_u"], z["main_p"].astype(np.float32)
    h, p = reconstruct(params, pc, st["cues"], food, u)
    valid = np.arange(p.shape[1])[None, :] < st["length"][:, None]
    out["reconstruction_max_abs_err"] = float(np.abs(p - p_rec)[valid].max())

    X, y, subj = pre_us_rows(h, scheds, food)
    tr, te = subj < 16, subj >= 16
    w, mu, ym = ridge(X[tr], y[tr])
    pred = (X[te] - mu) @ w + ym
    out["M1_probe_r2_heldout"] = float(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    d = w / np.linalg.norm(w)
    w_pi = np.asarray(params["pi"]["w"][:, 0])
    out["M2_cos_probe_readout"] = float(d @ w_pi / np.linalg.norm(w_pi))
    sd = float((X @ d).std())
    out["probe_projection_sd"] = sd

    rng = np.random.default_rng(0)
    r = rng.normal(size=d.shape)
    r -= (r @ d) * d
    r /= np.linalg.norm(r)
    mask = cs_mask(scheds, st["cues"].shape[1], CS_PLUS, "naive")
    steer = {}
    for name, vec in (("probe", d), ("random_orthogonal", r)):
        steer[name] = {}
        for a in ALPHAS:
            _, ps = reconstruct(params, pc, st["cues"], food, u, steer=(jnp.asarray(a * sd * vec), mask))
            steer[name][str(a)] = cr(ps, scheds, CS_PLUS, "naive")
    out["M3_naive_cs_plus_cr_by_alpha"] = steer

    # ---- omission vs yoked: expectation along d at pre-US ticks of the last 15 CS+ training trials
    scheds_o, _ = subject_schedules("omission", 32, nc, LAB_SEED)
    st_o = stack_schedules(scheds_o)
    zo = np.load(os.path.join(run, "lab", "omission.npz"))
    proj = {}
    for cond in ("main", "yoked"):
        ho, _ = reconstruct(params, pc, st_o["cues"], zo[f"{cond}_food"], zo[f"{cond}_u"])
        vals = []
        for b, s in enumerate(scheds_o):
            idx = [i for i, (k, tp) in enumerate(zip(s.trial_kind, s.trial_phase))
                   if k == CS_PLUS and tp == s.phases.index("training")][-15:]
            vals += [((ho[b, s.trial_onset[i] + s.isi - PRE_US:s.trial_onset[i] + s.isi].mean(0) - mu) @ w + ym)
                     for i in idx]
        proj[cond] = float(np.mean(vals))
    out["M4_expectation_omission"], out["M4_expectation_yoked"] = proj["main"], proj["yoked"]
    json.dump(out, open(os.path.join(run, "mechanism.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    for run in sys.argv[1:]:
        o = analyse(run)
        print(run, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items()
                    if k.startswith(("recon", "M1", "M2", "M4"))}, flush=True)
