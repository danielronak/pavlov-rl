"""Training ecology ("the wild"): per-lifetime world schedule generator.

Design property that enforces invariant I2 structurally: the entire world schedule (cues and food
onsets) is generated BEFORE the agent acts, from the lifetime's random key alone. Actions cannot
affect it. Lab contingencies (omission, etc.) live in pavlov.lab, never here (invariant I6).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pavlov.config import EcologyConfig


class LifetimeParams(NamedTuple):
    perm: jnp.ndarray        # [N_c] logical cue L is shown on physical channel perm[L]
    is_pred: jnp.ndarray     # [N_c] bool, logical
    rate: jnp.ndarray        # [N_c] onset rate per idle tick, logical
    dur: jnp.ndarray         # [N_c] cue duration, logical
    isi: jnp.ndarray         # [N_c] CS-onset -> US interval (predictive cues only)
    trace: jnp.ndarray       # [N_c] bool, trace cue
    p0: jnp.ndarray          # [N_c] original contingency
    food_rate: jnp.ndarray   # scalar unsignalled-food rate
    flipped: jnp.ndarray     # scalar bool: saliva hurts digestion this lifetime (R-O arm)
    resampled: jnp.ndarray   # scalar bool: cue->channel mapping was resampled (S-O arm)
    event_rate: jnp.ndarray  # scalar world-level stimulus rate (exclusive_events mode)
    gated: jnp.ndarray       # scalar bool: declared R-O gate active this lifetime (ro_family partial_omission)


class Schedule(NamedTuple):
    cues: jnp.ndarray         # [T, N_c] bool, PHYSICAL channels (what the agent sees)
    food_onset: jnp.ndarray   # [T] bool
    expected_us: jnp.ndarray  # [T] bool: tick a predictive-cue onset predicts a US (analysis/oracles only)
    pred_on: jnp.ndarray      # [T] bool: some predictive cue is on (analysis/oracles only)
    params: LifetimeParams
    pred_window: jnp.ndarray  # [T] bool: a predictive cue's response window (CS onset .. tick before US)
    window_start: jnp.ndarray # [T] bool: first tick of a predictive window
    us_pred: jnp.ndarray      # [T] bool: planned US caused by a predictive cue
    food_free: jnp.ndarray    # [T] bool: unsignalled food (never gated)
    gate_draw: jnp.ndarray    # [T] float: pre-drawn uniform used by the declared gate


def sample_params(key, cfg: EcologyConfig) -> LifetimeParams:
    n = cfg.n_channels
    ks = jax.random.split(key, 13)
    two = jax.random.uniform(ks[0]) < cfg.p_two_predictive
    idx = jnp.arange(n)
    is_pred = (idx == 0) | ((idx == 1) & two)

    # EVERY channel (predictive or distractor) draws its temporal structure from the SAME
    # distributions, so a cue's type is identifiable only from its relation to food (I3).
    isi = jax.random.randint(ks[1], (n,), cfg.isi_min, cfg.isi_max + 1)
    trace = jax.random.uniform(ks[2], (n,)) < cfg.p_trace
    gap = jax.random.randint(ks[3], (n,), cfg.trace_gap_min, cfg.trace_gap_max + 1)
    gap = jnp.minimum(gap, isi - 1)
    dur = jnp.where(trace, isi - gap, isi + 1)  # delay cue stays on through the (would-be) US tick
    rate = jax.random.uniform(ks[5], (n,), minval=cfg.cue_rate_min, maxval=cfg.cue_rate_max)
    p0 = jax.random.uniform(ks[7], (n,), minval=cfg.contingency_min, maxval=cfg.contingency_max)

    resampled = jax.random.uniform(ks[8]) < cfg.q_so_variability
    perm = jnp.where(resampled, jax.random.permutation(ks[9], n), idx)
    altered = jax.random.uniform(ks[10]) < cfg.v_ro_variability
    if cfg.ro_family == "digestion_flip":
        flipped, gated = altered, jnp.bool_(False)
    elif cfg.ro_family == "partial_omission":
        flipped, gated = jnp.bool_(False), altered
    else:
        raise ValueError(cfg.ro_family)
    food_rate = jax.random.uniform(ks[11], minval=0.0, maxval=cfg.unsignalled_food_max)
    event_rate = jax.random.uniform(ks[12], minval=cfg.event_rate_min, maxval=cfg.event_rate_max)
    return LifetimeParams(perm, is_pred, rate, dur, isi, trace, p0, food_rate, flipped, resampled,
                          event_rate, gated)


def generate_schedule(key, cfg: EcologyConfig) -> Schedule:
    kp, ks = jax.random.split(key)
    P = sample_params(kp, cfg)
    n = cfg.n_channels
    neg = -jnp.ones(n, jnp.int32)

    def tick(carry, k):
        timer, countdown, pending, active, p_cur = carry
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(k, 7)
        # non-stationarity: predictive cues toggle between active and silent
        toggle = P.is_pred & (jax.random.uniform(k1, (n,)) < cfg.hazard)
        reactivate = toggle & ~active
        revert = jax.random.uniform(k2, (n,)) < cfg.reversion
        p_new = jax.random.uniform(k3, (n,), minval=cfg.contingency_min, maxval=cfg.contingency_max)
        p_cur = jnp.where(reactivate, jnp.where(revert, P.p0, p_new), p_cur)
        active = jnp.where(toggle, ~active, active)
        # cue onsets: every channel waits until its previous "trial" window (isi) has elapsed,
        # predictive or not, so refractory structure is identical across cue types
        idle = (timer == 0) & (pending < 0)
        if cfg.exclusive_events:
            # one stimulus at a time: a world-level event presents ONE uniformly chosen channel,
            # and only when every channel is idle (identical statistics across channels, I3)
            start = jnp.all(idle) & (jax.random.uniform(k4) < P.event_rate)
            ch = jax.random.randint(k7, (), 0, n)
            onset = start & (jnp.arange(n) == ch)
        else:
            onset = idle & (jax.random.uniform(k4, (n,)) < P.rate)
        timer = jnp.where(onset, P.dur, timer)
        on = timer > 0
        # US scheduling
        success = jax.random.uniform(k5, (n,)) < p_cur * active
        countdown = jnp.where(onset & P.is_pred & success, P.isi, countdown)
        pending = jnp.where(onset, P.isi, pending)
        us = jnp.any(countdown == 0)
        exp_us = jnp.any((pending == 0) & P.is_pred)
        window = jnp.any((pending > 0) & P.is_pred)
        free = jax.random.uniform(k6) < P.food_rate
        food = us | free
        # advance timers
        timer = jnp.maximum(timer - 1, 0)
        countdown = jnp.where(countdown > 0, countdown - 1, -1)
        pending = jnp.where(pending > 0, pending - 1, -1)
        out = (on, food, exp_us, jnp.any(on & P.is_pred), window, us, free)
        return (timer, countdown, pending, active, p_cur), out

    init = (jnp.zeros(n, jnp.int32), neg, neg, jnp.ones(n, bool), P.p0)
    _, (on_logical, food, exp_us, pred_on, window, us, free) = jax.lax.scan(
        tick, init, jax.random.split(ks, cfg.lifetime)
    )
    cues = jnp.zeros_like(on_logical).at[:, P.perm].set(on_logical)
    start = window & ~jnp.concatenate([jnp.zeros(1, bool), window[:-1]])
    # separate key stream so existing schedules stay bit-identical
    gate_draw = jax.random.uniform(jax.random.fold_in(key, 7919), (cfg.lifetime,))
    return Schedule(cues=cues, food_onset=food, expected_us=exp_us, pred_on=pred_on, params=P,
                    pred_window=window, window_start=start, us_pred=us, food_free=free,
                    gate_draw=gate_draw)


def generate_batch(key, cfg: EcologyConfig, n: int) -> Schedule:
    return jax.vmap(lambda k: generate_schedule(k, cfg))(jax.random.split(key, n))
