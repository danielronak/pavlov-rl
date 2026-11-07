"""Declared response->outcome gate for the "partial_omission" R-O family (DECISIONS 2026-11-07).

This is the ONLY place in training where the agent's actions can affect food delivery
(invariant I2's declared exception). In a gated lifetime, if the agent salivated at any tick of a
predictive cue's response window, that trial's planned US is delivered only with probability
`factor` (on top of the cue's contingency, which the schedule already applied). Unsignalled food
is never gated. In ungated lifetimes, and for every other ro_family, delivered food == planned.
"""
from __future__ import annotations

import jax.numpy as jnp


def gated_food(us_pred_t, food_free_t, gated, responded, draw_t, factor):
    keep = ~(gated & responded) | (draw_t < factor)
    return food_free_t | (us_pred_t & keep)


def update_responded(responded, window_start_t, window_t, u_t):
    """Reset at the start of a predictive window; set if the agent salivates inside the window."""
    responded = jnp.where(window_start_t, False, responded)
    return responded | (window_t & (u_t > 0))
