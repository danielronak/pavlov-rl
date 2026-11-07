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


def _xs(sched):
    sw = lambda a: jnp.swapaxes(a, 0, 1)
    return (sw(sched.cues), sched.food_onset.T, sched.us_pred.T, sched.food_free.T,
            sched.pred_window.T, sched.window_start.T, sched.gate_draw.T)


def run_schedule(params, sched, cfg: Config, hidden: int, key):
    """Closed-loop lifetimes with the declared R-O gate (ro_gate). For ro_family != partial_omission
    or ungated lifetimes this is identical to run_lifetimes. Returns (returns [B], p [B,T])."""
    from pavlov.ro_gate import gated_food, update_responded
    pc = cfg.physiology
    gating = cfg.ecology.ro_family == "partial_omission"
    B = sched.food_onset.shape[0]
    gated = sched.params.gated

    def body(carry, xs):
        es, h, key, resp = carry
        cues_t, food_t, usp_t, free_t, win_t, ws_t, draw_t = xs
        key, k = jax.random.split(key)
        resp = jnp.where(ws_t, False, resp)
        food = gated_food(usp_t, free_t, gated, resp, draw_t, cfg.ecology.gate_factor) if gating else food_t
        obs = observe(es, cues_t, food, pc)
        h, logit, _ = A.step(params, h, obs)
        p = jax.nn.sigmoid(logit)
        u = (jax.random.uniform(k, p.shape) < p).astype(jnp.float32)
        resp = update_responded(resp, ws_t, win_t, u)
        es, r, _ = env_step(es, food, u, sched.params.flipped, pc)
        return (es, h, key, resp), (r, p)

    init = (init_env(B), jnp.zeros((B, hidden)), key, jnp.zeros(B, bool))
    _, (r, p) = jax.lax.scan(body, init, _xs(sched))
    return r.sum(0), p.T


def fixed_action_returns(sched, actions, cfg: Config):
    """Lifetime returns of a FIXED action sequence [B, T] (oracles), with the declared R-O gate."""
    from pavlov.ro_gate import gated_food, update_responded
    pc = cfg.physiology
    gating = cfg.ecology.ro_family == "partial_omission"
    B = sched.food_onset.shape[0]
    gated = sched.params.gated

    def body(carry, xs):
        es, resp = carry
        (cues_t, food_t, usp_t, free_t, win_t, ws_t, draw_t), u = xs
        resp = jnp.where(ws_t, False, resp)
        food = gated_food(usp_t, free_t, gated, resp, draw_t, cfg.ecology.gate_factor) if gating else food_t
        u = u.astype(jnp.float32)
        resp = update_responded(resp, ws_t, win_t, u)
        es, r, _ = env_step(es, food, u, sched.params.flipped, pc)
        return (es, resp), r

    _, r = jax.lax.scan(body, (init_env(B), jnp.zeros(B, bool)), (_xs(sched), actions.T))
    return r.sum(0)
