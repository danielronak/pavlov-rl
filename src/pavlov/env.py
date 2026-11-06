"""Batched closed-loop environment over pre-generated lifetime schedules.

Timing within tick t (must match physiology.phys_step):
  obs_t  = [cues_t, food_in_mouth_t, saliva_t, u_{t-1}, r_{t-1}]
  agent picks u_t from obs_t (and its recurrent state)
  r_t    = energy_t(saliva_t) - c*u_t ; saliva_{t+1} depends on u_t
So salivation can only help food that arrives at t+1 or later -> anticipation is required.
The agent observes nothing else: no trial flags, no time index, no cue identity labels.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pavlov.config import PhysiologyConfig
from pavlov.physiology import PhysState, phys_step


class EnvState(NamedTuple):
    t: jnp.ndarray          # scalar int (lifetimes in a batch are synchronised)
    phys: PhysState         # batched [B]
    prev_u: jnp.ndarray     # [B]
    prev_r: jnp.ndarray     # [B]


def obs_dim(n_channels: int) -> int:
    return n_channels + 4


def init_env(B: int) -> EnvState:
    return EnvState(
        t=jnp.int32(0),
        phys=PhysState(saliva=jnp.zeros(B), food_timer=jnp.zeros(B, jnp.int32)),
        prev_u=jnp.zeros(B),
        prev_r=jnp.zeros(B),
    )


def observe(state: EnvState, cues_t, food_onset_t, cfg: PhysiologyConfig):
    food_in_mouth = jnp.where(food_onset_t, cfg.food_ticks, state.phys.food_timer) > 0
    return jnp.concatenate(
        [
            cues_t.astype(jnp.float32),
            food_in_mouth[:, None].astype(jnp.float32),
            state.phys.saliva[:, None],
            state.prev_u[:, None],
            state.prev_r[:, None],
        ],
        axis=-1,
    )


def env_step(state: EnvState, food_onset_t, u, flipped, cfg: PhysiologyConfig):
    step = jax.vmap(lambda s, f, a, fl: phys_step(s, f, a, fl, cfg))
    phys, r, info = step(state.phys, food_onset_t, u, flipped)
    new = EnvState(t=state.t + 1, phys=phys, prev_u=u.astype(jnp.float32), prev_r=r)
    return new, r, info
