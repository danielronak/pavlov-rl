"""Policies that can be run in the lab.

`agent_policy` wraps a frozen trained agent. Its step function only sees the observation; it
gets no schedule information.

The ORACLE policies exist only to validate the measurement pipeline (Phase-3 gate). Some read
privileged information (which channel is the bell) through `info`. They are never compared with
agents as if they were models.

Every policy step has the signature (carry, obs, t, info) -> (carry, p_saliva) and must be a
hashable module-level function or a functools.partial over hashable args (static for jit).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from pavlov import agent as A

FOOD_IDX = -4  # obs layout: [cues(Nc), food, saliva, u_prev, r_prev]


class _Params:
    """Hashable read-only wrapper so frozen params can be a static jit argument."""
    def __init__(self, params):
        self.params = jax.tree.map(jnp.asarray, params)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


def _agent_step(wrapped: _Params, h, obs, t, info):
    h, logit, _ = A.step(wrapped.params, h, obs)
    return h, jax.nn.sigmoid(logit)


def agent_policy(params, hidden: int, B: int):
    return partial(_agent_step, _Params(params)), jnp.zeros((B, hidden))


def _never(carry, obs, t, info):
    return carry, jnp.zeros(obs.shape[0])


def _bell_reader(carry, obs, t, info):
    """Pavlovian BY CONSTRUCTION: fixed reflex to the bell channel, plus reflex to food.
    Expected: CR present already in the naive phase, no extinction, OSI ≈ 0."""
    bell_on = jnp.take_along_axis(obs[:, :-4], info["bell"][:, None], axis=1)[:, 0] > 0
    food = obs[:, FOOD_IDX] > 0
    return carry, jnp.where(bell_on, 0.9, jnp.where(food, 1.0, 0.0))


def _withholder(carry, obs, t, info):
    """Instrumental BY CONSTRUCTION (privileged: told whether the omission contingency is on):
    salivates to the bell after the naive phase, except under omission, where it withholds.
    Its yoked partner (no contingency) responds. Expected OSI = 1. Validates the OSI arithmetic."""
    bell_on = jnp.take_along_axis(obs[:, :-4], info["bell"][:, None], axis=1)[:, 0] > 0
    food = obs[:, FOOD_IDX] > 0
    active = (t >= info["naive_end"]) & ~info["omission"]
    return carry, jnp.where(bell_on & active, 0.9, jnp.where(food, 1.0, 0.0))


def oracle_policy(name: str, B: int):
    if name == "never":
        return _never, jnp.zeros(B)
    if name == "bell_reader":
        return _bell_reader, jnp.zeros(B)
    if name == "withholder":
        return _withholder, jnp.zeros(B)
    raise KeyError(name)


ORACLE_POLICIES = ("never", "bell_reader", "withholder")
