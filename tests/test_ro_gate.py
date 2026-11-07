"""Declared R-O gate (ro_family partial_omission) and its no-op guarantees."""
import jax
import jax.numpy as jnp
import numpy as np

from pavlov import agent as A
from pavlov.config import Config, EcologyConfig
from pavlov.ecology import generate_batch
from pavlov.env import obs_dim
from pavlov.physiology import run_open_loop
from pavlov.rollout import fixed_action_returns, run_lifetimes, run_schedule


def _cfg(**kw):
    return Config(ecology=EcologyConfig(lifetime=1500, **kw))


def test_schedule_decomposition():
    s = generate_batch(jax.random.PRNGKey(0), EcologyConfig(lifetime=1500), 16)
    assert np.array_equal(np.asarray(s.food_onset), np.asarray(s.us_pred | s.food_free))
    # every predictive window ends right before the expected-US tick
    ws, win, exp = map(np.asarray, (s.window_start, s.pred_window, s.expected_us))
    d = ws.sum(1) - exp.sum(1)  # a window opening near the lifetime end may lose its US tick
    assert np.all((d == 0) | (d == 1))
    assert not np.any(win & exp)


def test_gate_is_noop_without_partial_omission():
    cfg = _cfg(v_ro_variability=0.5)  # digestion_flip family: the gate must never act
    s = generate_batch(jax.random.PRNGKey(1), cfg.ecology, 16)
    act = jnp.ones_like(s.food_onset)
    a = fixed_action_returns(s, act, cfg)
    b = jax.vmap(lambda fo, x, fl: run_open_loop(fo, x, fl, cfg.physiology)["reward"].sum())(s.food_onset, act, s.params.flipped)
    assert np.allclose(np.asarray(a), np.asarray(b), atol=1e-4)


def test_gate_halves_us_only_when_responding_in_window():
    cfg = _cfg(v_ro_variability=1.0, ro_family="partial_omission", unsignalled_food_max=0.0)
    s = generate_batch(jax.random.PRNGKey(2), cfg.ecology, 64)
    assert bool(np.all(np.asarray(s.params.gated)))
    pc = cfg.physiology
    respond = s.pred_window  # salivate through every predictive window (and nowhere else)
    never = jnp.zeros_like(s.food_onset)
    r_resp = np.asarray(fixed_action_returns(s, respond, cfg))
    r_never = np.asarray(fixed_action_returns(s, never, cfg))
    open_never = np.asarray(jax.vmap(lambda fo, x, fl: run_open_loop(fo, x, fl, pc)["reward"].sum())(s.food_onset, never, s.params.flipped))
    assert np.allclose(r_never, open_never, atol=1e-4)            # no response -> full food
    assert r_resp.mean() < r_never.mean()                          # responding in-window costs food
    # salivating OUTSIDE windows never triggers the gate
    outside = jnp.ones_like(s.food_onset) & ~s.pred_window
    r_out = np.asarray(fixed_action_returns(s, outside, cfg))
    open_out = np.asarray(jax.vmap(lambda fo, x, fl: run_open_loop(fo, x, fl, pc)["reward"].sum())(s.food_onset, outside, s.params.flipped))
    assert np.allclose(r_out, open_out, atol=1e-4)


def test_run_schedule_matches_run_lifetimes_for_other_families():
    cfg = _cfg(v_ro_variability=0.5)
    s = generate_batch(jax.random.PRNGKey(3), cfg.ecology, 8)
    acfg = A.AgentConfig(hidden=16, embed=8)
    params = A.init_params(jax.random.PRNGKey(4), obs_dim(cfg.ecology.n_channels), acfg)
    a, pa = run_schedule(params, s, cfg, 16, jax.random.PRNGKey(5))
    b, pb = run_lifetimes(params, s.cues, s.food_onset, s.params.flipped, cfg, 16, jax.random.PRNGKey(5))
    assert np.array_equal(np.asarray(a), np.asarray(b)) and np.array_equal(np.asarray(pa), np.asarray(pb))
