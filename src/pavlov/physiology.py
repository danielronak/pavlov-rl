"""Saliva + digestion physiology.

This module is the ONLY place reward is computed (invariant I1). Its inputs are deliberately
restricted to: food onset, the salivation command, and the lifetime's digestion sign. It never
receives cue information, so reward cannot reference the CS.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pavlov.config import PhysiologyConfig


class PhysState(NamedTuple):
    saliva: jnp.ndarray      # S_t in [0, 1)
    food_timer: jnp.ndarray  # ticks of food left in the mouth (int)


def init_phys() -> PhysState:
    return PhysState(saliva=jnp.float32(0.0), food_timer=jnp.int32(0))


def efficiency(saliva, flipped, cfg: PhysiologyConfig):
    """Digestion efficiency. Normal: saliva helps. Flipped (R-O variability arm): saliva hurts."""
    normal = cfg.b0 + (1.0 - cfg.b0) * saliva
    hurt = 1.0 - (1.0 - cfg.b0) * saliva
    return jnp.where(flipped, hurt, normal)


def phys_step(state: PhysState, food_onset, u, flipped, cfg: PhysiologyConfig):
    """One tick.

    Order of events within tick t:
      1. a US at t puts food in the mouth (food_timer = D);
      2. if food is present, energy is extracted using the CURRENT saliva S_t;
      3. the salivation command u_t raises saliva for t+1 (so anticipation must precede food).
    Returns (new_state, reward, info).
    """
    u = u.astype(jnp.float32)
    food_timer = jnp.where(food_onset, cfg.food_ticks, state.food_timer)
    food_present = food_timer > 0
    energy = jnp.where(
        food_present, (cfg.energy / cfg.food_ticks) * efficiency(state.saliva, flipped, cfg), 0.0
    )
    reward = energy - cfg.cost * u
    saliva = state.saliva + cfg.k_up * u * (1.0 - state.saliva) - cfg.k_down * state.saliva
    new_state = PhysState(saliva=saliva, food_timer=jnp.maximum(food_timer - 1, 0))
    info = {"energy": energy, "food_present": food_present, "saliva_used": state.saliva}
    return new_state, reward, info


def run_open_loop(food_onset, actions, flipped, cfg: PhysiologyConfig):
    """Run physiology over a lifetime with a fixed action sequence (valid because the world
    schedule does not depend on actions in the primary ecology — invariant I2)."""

    def body(state, xs):
        f, u = xs
        state, r, info = phys_step(state, f, u, flipped, cfg)
        return state, (r, info["energy"], info["food_present"], info["saliva_used"])

    _, (r, energy, food_present, saliva) = jax.lax.scan(body, init_phys(), (food_onset, actions))
    return {"reward": r, "energy": energy, "food_present": food_present, "saliva": saliva}
