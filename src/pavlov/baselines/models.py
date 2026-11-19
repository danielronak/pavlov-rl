"""Baseline learners as lab policies (parameters fixed in docs/DECISIONS.md 2026-11-19, never tuned
on lab metrics). Each sees only the agent's observation [cues(Nc), food, saliva, u_prev, r_prev] and
salivates reflexively (p = 1) while food is in the mouth.

  rw  Rescorla-Wagner (trial level): V_c per channel, updated at the end of each stimulus episode.
  td  TD(0) with a complete serial compound (one feature per channel x ticks since its onset).
  q   instrumental tabular Q-learner (state = channel on x ticks since onset), rewarded by physiology.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

K = 32  # time-since-onset units per channel


def _split(obs):
    nc = obs.shape[1] - 4
    return obs[:, :nc] > 0, obs[:, nc] > 0, obs[:, -2], obs[:, -1]


def _rw_step(alpha, carry, obs, t, info):
    V, present, fed = carry
    on, food, _, _ = _split(obs)
    any_on = on.any(1)
    end = present.any(1) & ~any_on
    pred = (V * present).sum(1)
    lam = fed.astype(jnp.float32)
    V = V + jnp.where(end[:, None] & present, alpha * (lam - pred)[:, None], 0.0)
    present = jnp.where(end[:, None], False, present | on)
    fed = jnp.where(end, False, fed | (food & any_on))
    p = jnp.where(food, 1.0, jnp.clip((V * on).sum(1), 0.0, 1.0))
    return (V, present, fed), p


def _since(since, on):
    return jnp.where(on, jnp.where(since >= 0, since + 1, 0), -1)


def _onehot(since):
    return jax.nn.one_hot(jnp.where(since >= 0, jnp.minimum(since, K - 1), -1), K)  # -1 -> all zeros


def _td_step(alpha, gamma, carry, obs, t, info):
    w, since, prev_x, prev_food = carry
    on, food, _, _ = _split(obs)
    since = _since(since, on)
    x = _onehot(since)
    r = (food & ~prev_food).astype(jnp.float32)
    v, v_prev = (w * x).sum((1, 2)), (w * prev_x).sum((1, 2))
    delta = r + gamma * v - v_prev
    w = w + alpha * delta[:, None, None] * prev_x
    v = (w * x).sum((1, 2))
    p = jnp.where(food, 1.0, jnp.clip(v, 0.0, 1.0))
    return (w, since, x, food), p


def _q_state(since, on):
    nc = on.shape[1]
    first = jnp.argmax(on, axis=1)
    s_on = first * K + jnp.minimum(jnp.take_along_axis(since, first[:, None], 1)[:, 0], K - 1)
    return jnp.where(on.any(1), s_on, nc * K)


def _q_step(alpha, gamma, beta, carry, obs, t, info):
    Q, since, prev_s, started = carry
    on, food, u_prev, r_prev = _split(obs)
    since = _since(since, on)
    s = _q_state(since, on)
    b = jnp.arange(Q.shape[0])
    a = u_prev.astype(jnp.int32)
    target = r_prev + gamma * Q[b, s].max(1)
    upd = alpha * (target - Q[b, prev_s, a])
    Q = Q.at[b, prev_s, a].add(jnp.where(started, upd, 0.0))
    p = jax.nn.sigmoid(beta * (Q[b, s, 1] - Q[b, s, 0]))
    return (Q, since, s, jnp.ones_like(started)), jnp.where(food, 1.0, p)


def make_policy(model: str, n_channels: int):
    """Returns make_policy(B) -> (step, carry), as lab.experiment.run_protocol expects."""
    nc = n_channels

    def factory(B):
        if model == "rw":
            return partial(_rw_step, 0.3), (jnp.zeros((B, nc)), jnp.zeros((B, nc), bool), jnp.zeros(B, bool))
        if model == "td":
            return partial(_td_step, 0.1, 0.9), (jnp.zeros((B, nc, K)), -jnp.ones((B, nc), jnp.int32),
                                                  jnp.zeros((B, nc, K)), jnp.zeros(B, bool))
        if model == "q":
            return partial(_q_step, 0.2, 0.9, 5.0), (jnp.zeros((B, nc * K + 1, 2)), -jnp.ones((B, nc), jnp.int32),
                                                      jnp.full(B, nc * K, jnp.int32), jnp.zeros(B, bool))
        raise KeyError(model)
    return factory


MODELS = ("rw", "td", "q")
