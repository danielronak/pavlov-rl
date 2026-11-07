"""Hand-written reference policies (open-loop action arrays computed from the schedule).

Used ONLY for (a) the Phase-1 physiology gate and (b) validating the lab measurement pipeline.
They are never used in training. Some oracles use privileged information (which cue is
predictive, the true ISI); that is the point: they bound what a learner could achieve.
"""
from __future__ import annotations

import jax.numpy as jnp

from pavlov.config import PhysiologyConfig
from pavlov.ecology import Schedule


def _lead_window(events, k):
    """x[t] = any(events[t+1 .. t+k]): the k ticks immediately BEFORE each event."""
    T = events.shape[-1]
    out = jnp.zeros_like(events)
    for j in range(1, k + 1):
        shifted = jnp.concatenate([events[..., j:], jnp.zeros(events.shape[:-1] + (j,), bool)], -1)
        out = out | shifted[..., :T]
    return out


def food_present(food_onset, D):
    T = food_onset.shape[-1]
    out = jnp.zeros_like(food_onset)
    for d in range(D):
        shifted = jnp.concatenate([jnp.zeros(food_onset.shape[:-1] + (d,), bool), food_onset[..., : T - d]], -1)
        out = out | shifted
    return out


def never(s: Schedule, cfg: PhysiologyConfig):
    return jnp.zeros_like(s.food_onset)


def always(s: Schedule, cfg: PhysiologyConfig):
    return jnp.ones_like(s.food_onset)


def reflex(s: Schedule, cfg: PhysiologyConfig):
    """Unconditioned response only: salivate while food is in the mouth."""
    return food_present(s.food_onset, cfg.food_ticks)


def any_cue(s: Schedule, cfg: PhysiologyConfig):
    """Cue-blind: salivate whenever ANY stimulus is on (no knowledge of which predicts food)."""
    return jnp.any(s.cues, -1) | reflex(s, cfg)


def pred_cue_on(s: Schedule, cfg: PhysiologyConfig):
    """Knows which cue is predictive but not its timing: salivate while it is on."""
    return s.pred_on | reflex(s, cfg)


def cue_oracle(s: Schedule, cfg: PhysiologyConfig, k: int = 4):
    """Knows predictive cue + ISI (not the trial outcome): salivate k ticks before each EXPECTED US."""
    return _lead_window(s.expected_us, k) | reflex(s, cfg)


def prophet(s: Schedule, cfg: PhysiologyConfig, k: int = 4):
    """Upper bound: knows every actual food onset in advance (including unsignalled food)."""
    return _lead_window(s.food_onset, k) | reflex(s, cfg)


def ro_aware(actions, s: Schedule):
    """R-O-aware version of any oracle: in lifetimes where saliva hurts digestion (flipped), never
    salivate. Used for the instrumental-ecology arm (v > 0) only."""
    return actions & ~s.params.flipped[:, None]


def probe_then_aware(s: Schedule, cfg: PhysiologyConfig, k: int = 4, n_probe: int = 2):
    """Learnability oracle (G10): reflex-salivate during the first n_probe meals of every lifetime
    (the only way to discover the lifetime type under A4), then behave R-O-aware."""
    nth = jnp.cumsum(s.food_onset, -1)
    probe = food_present(s.food_onset, cfg.food_ticks) & (nth <= n_probe)
    return ro_aware(cue_oracle(s, cfg, k), s) | probe


ORACLES = {
    "never": never,
    "always": always,
    "reflex": reflex,
    "any_cue": any_cue,
    "pred_cue_on": pred_cue_on,
    "cue_oracle": cue_oracle,
    "prophet": prophet,
}
