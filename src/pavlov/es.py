"""OpenAI-ES outer loop (Salimans et al. 2017): literal evolution of the agent's weights.

Fitness = mean whole-lifetime return (physiology only, I1). No gradients flow through time, so
there is no truncation: remembering an association across the whole lifetime is selected for
directly. All members of a generation are evaluated on the SAME lifetimes with the SAME
action-sampling keys (common random numbers), so antithetic pairs differ only by their weights.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from pavlov import agent as A
from pavlov.config import Config
from pavlov.ecology import generate_batch
from pavlov.env import obs_dim
from pavlov.rollout import run_lifetimes


@dataclass(frozen=True)
class ESConfig:
    pop: int = 64                 # members per generation (antithetic: pop/2 noise vectors)
    lifetimes: int = 8            # lifetimes per member per generation (shared across members)
    sigma: float = 0.03
    lr: float = 0.02
    weight_decay: float = 0.005
    generations: int = 2000


def centered_ranks(x):
    r = jnp.argsort(jnp.argsort(x))
    return r / (x.shape[0] - 1) - 0.5


def make_generation(cfg: Config, acfg: A.AgentConfig, ecfg: ESConfig, unravel, opt):
    half = ecfg.pop // 2

    def fitness(flat, sched, key):
        params = unravel(flat)
        ret, _ = run_lifetimes(params, sched.cues, sched.food_onset, sched.params.flipped, cfg,
                               acfg.hidden, key)
        return ret.mean()

    @jax.jit
    def generation(theta, opt_state, key):
        k_eps, k_sched, k_act = jax.random.split(key, 3)
        sched = generate_batch(k_sched, cfg.ecology, ecfg.lifetimes)
        eps = jax.random.normal(k_eps, (half, theta.shape[0]))
        members = jnp.concatenate([theta + ecfg.sigma * eps, theta - ecfg.sigma * eps])
        all_flat = jnp.concatenate([members, theta[None]])
        fit = jax.vmap(lambda f: fitness(f, sched, k_act))(all_flat)
        f_members, f_center = fit[:-1], fit[-1]
        w = centered_ranks(f_members)
        g = ((w[:half] - w[half:])[:, None] * eps).sum(0) / (ecfg.pop * ecfg.sigma)
        g = -g + ecfg.weight_decay * theta  # optax minimises
        upd, opt_state = opt.update(g, opt_state, theta)
        return optax.apply_updates(theta, upd), opt_state, f_members.mean(), f_center

    return generation


def _atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def train_es(cfg: Config, acfg: A.AgentConfig, ecfg: ESConfig, seed: int, out_dir: str,
             log_every: int = 10, ckpt_every: int = 100, init_from: str | None = None,
             resume: bool = False):
    """init_from: path to an ancestor's params pickle (common-ancestor design); the Adam state starts
    fresh. resume: continue exactly (theta, Adam state, RNG key, generation) from
    checkpoints/es_state.pkl written every `ckpt_every` generations."""
    os.makedirs(out_dir, exist_ok=True)
    ck = os.path.join(out_dir, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    state_path = os.path.join(ck, "es_state.pkl")
    if not (resume and os.path.exists(state_path)):
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump({"config": cfg.to_dict(), "agent": asdict(acfg), "es": asdict(ecfg), "seed": seed,
                       "outer_loop": "openai_es", "init_from": init_from, "jax": jax.__version__}, f, indent=2)
    key = jax.random.PRNGKey(seed)
    key, kp = jax.random.split(key)
    params = A.init_params(kp, obs_dim(cfg.ecology.n_channels), acfg)
    theta, unravel = ravel_pytree(params)
    if init_from:
        with open(init_from, "rb") as f:
            anc = pickle.load(f)
        theta_anc, _ = ravel_pytree(jax.tree.map(jnp.asarray, anc))
        assert theta_anc.shape == theta.shape, "ancestor architecture mismatch"
        theta = theta_anc
    opt = optax.adam(ecfg.lr)
    opt_state = opt.init(theta)
    start_gen = 0
    if resume and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        theta, key, start_gen = jnp.asarray(st["theta"]), jnp.asarray(st["key"]), st["gen"]
        opt_state = jax.tree.map(jnp.asarray, st["opt_state"])
        print(f"resumed at generation {start_gen}", flush=True)
    generation = make_generation(cfg, acfg, ecfg, unravel, opt)
    steps_per_gen = (ecfg.pop + 1) * ecfg.lifetimes * cfg.ecology.lifetime
    log_path = os.path.join(out_dir, "train_log.jsonl")
    t0 = time.time()
    for gen in range(start_gen, ecfg.generations):
        key, kg = jax.random.split(key)
        theta, opt_state, f_mean, f_center = generation(theta, opt_state, kg)
        if gen % log_every == 0 or gen == ecfg.generations - 1:
            el = max(time.time() - t0, 1e-6)
            rec = {"generation": gen, "env_steps": (gen + 1) * steps_per_gen, "wall_s": round(el, 1),
                   "steps_per_s": round((gen + 1 - start_gen) * steps_per_gen / el),
                   "fitness_members": float(f_mean), "fitness_center": float(f_center),
                   "theta_norm": float(jnp.linalg.norm(theta))}
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{gen:5d}] steps={rec['env_steps']:.2e} fit_mean={rec['fitness_members']:.2f} "
                  f"fit_center={rec['fitness_center']:.2f} |θ|={rec['theta_norm']:.1f} "
                  f"sps={rec['steps_per_s']}", flush=True)
        if (gen + 1) % ckpt_every == 0 or gen == ecfg.generations - 1:
            p = jax.device_get(unravel(theta))
            for name in (f"params_{gen + 1:05d}.pkl", "params_latest.pkl"):
                _atomic_pickle(p, os.path.join(ck, name))
            _atomic_pickle({"theta": np.asarray(theta), "opt_state": jax.device_get(opt_state),
                            "key": np.asarray(key), "gen": gen + 1}, state_path)
    return unravel(theta)
