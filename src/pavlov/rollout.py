"""Closed-loop execution of an agent over whole lifetimes (forward only, no learning).
Shared by ES fitness evaluation and by training-ecology evaluation."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from pavlov import agent as A
from pavlov.config import Config
from pavlov.env import env_step, init_env, observe


def run_lifetimes(params, cues, food_onset, flipped, cfg: Config, hidden: int, key):
    """cues [B,T,Nc], food_onset [B,T], flipped [B] -> (returns [B], p_saliva [B,T])."""
    pc = cfg.physiology
    B = food_onset.shape[0]

    def body(carry, xs):
        es, h, key = carry
        cues_t, food_t = xs
        key, k = jax.random.split(key)
        obs = observe(es, cues_t, food_t, pc)
        h, logit, _ = A.step(params, h, obs)
        p = jax.nn.sigmoid(logit)
        u = (jax.random.uniform(k, p.shape) < p).astype(jnp.float32)
        es, r, _ = env_step(es, food_t, u, flipped, pc)
        return (es, h, key), (r, p)

    xs = (jnp.swapaxes(cues, 0, 1), jnp.swapaxes(food_onset, 0, 1))
    _, (r, p) = jax.lax.scan(body, (init_env(B), jnp.zeros((B, hidden)), key), xs)
    return r.sum(0), p.T
