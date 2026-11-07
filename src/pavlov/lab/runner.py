"""Frozen-agent lab execution.

Invariant I5: this module never imports or calls an optimizer, and policies receive parameters
read-only. All learning in the lab must happen in the policy's recurrent state.
Invariant I7: the schedule is fixed before the run; the agent's actions influence what happens
only via the declared contingency:
  none     -> food = planned US
  omission -> a planned CS+ US is cancelled if the agent salivated in that trial's response window
  positive -> a planned CS+ US is delivered only if the agent salivated in that window
  replay   -> food = an exact recorded food sequence (yoked partner)
"""
from __future__ import annotations

from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from pavlov.config import PhysiologyConfig
from pavlov.env import env_step, init_env, observe

CONTINGENCIES = ("none", "omission", "positive", "replay")


@partial(jax.jit, static_argnames=("policy_step", "contingency", "pc", "n_trials"))
def _run(policy_step: Callable, pol_carry, sched, replay_food, info, key, *, contingency, pc,
         n_trials):
    B, T = sched["us_planned"].shape
    bidx = jnp.arange(B)

    def body(carry, xs):
        es, pc_carry, key, flags = carry
        t, cues_t, us_t, ust_t, win_t, rep_t = xs
        key, k = jax.random.split(key)
        hit = flags[bidx, jnp.maximum(ust_t, 0)]
        if contingency == "none":
            food = us_t
        elif contingency == "omission":
            food = us_t & ~((ust_t >= 0) & hit)
        elif contingency == "positive":
            food = us_t & ((ust_t < 0) | hit)
        else:  # replay
            food = rep_t
        obs = observe(es, cues_t, food, pc)
        pc_carry, p = policy_step(pc_carry, obs, t, info)
        u = (jax.random.uniform(k, p.shape) < p).astype(jnp.float32)
        responded = (win_t >= 0) & (u > 0)
        w = jnp.maximum(win_t, 0)
        flags = flags.at[bidx, w].set(flags[bidx, w] | responded)
        es, r, _ = env_step(es, food, u, jnp.zeros(B, bool), pc)
        return (es, pc_carry, key, flags), (p, u, food, r)

    xs = (jnp.arange(T), jnp.swapaxes(sched["cues"], 0, 1), sched["us_planned"].T,
          sched["us_trial"].T, sched["window_trial"].T, replay_food.T)
    init = (init_env(B), pol_carry, key, jnp.zeros((B, n_trials), bool))
    (_, _, _, flags), (p, u, food, r) = jax.lax.scan(body, init, xs)
    return p.T, u.T, food.T, r.T, flags


def run_lab(policy_step, pol_carry, sched: dict, contingency: str, pc: PhysiologyConfig, key,
            info: dict | None = None, replay_food=None):
    """sched: output of scheduler.stack_schedules. Returns numpy arrays p, u, food, reward [B, T]
    and per-trial response flags [B, n_trials]."""
    assert contingency in CONTINGENCIES, contingency
    if contingency == "replay":
        assert replay_food is not None and replay_food.shape == sched["us_planned"].shape
    else:
        replay_food = np.zeros_like(sched["us_planned"])
    js = {k: jnp.asarray(sched[k]) for k in ("cues", "us_planned", "us_trial", "window_trial")}
    p, u, food, r, flags = _run(policy_step, pol_carry, js, jnp.asarray(replay_food), info or {},
                                key, contingency=contingency, pc=pc, n_trials=int(sched["n_trials"]))
    return {"p": np.asarray(p), "u": np.asarray(u), "food": np.asarray(food), "reward": np.asarray(r),
            "flags": np.asarray(flags)}
