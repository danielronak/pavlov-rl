import inspect

import jax
import jax.numpy as jnp
import numpy as np

from pavlov import agent as A
from pavlov import ppo
from pavlov.config import Config, EcologyConfig, PhysiologyConfig
from pavlov.env import env_step, init_env, obs_dim, observe

PC = PhysiologyConfig()


def test_obs_contains_no_privileged_info():
    """The agent sees only cues, food in mouth, saliva, previous action and previous reward."""
    assert obs_dim(6) == 10
    params = list(inspect.signature(observe).parameters)
    assert params == ["state", "cues_t", "food_onset_t", "cfg"]


def test_food_visible_on_onset_tick_but_saliva_lags():
    es = init_env(1)
    o = observe(es, jnp.zeros((1, 6), bool), jnp.array([True]), PC)
    assert o[0, 6] == 1.0  # food in mouth visible
    es2, r, _ = env_step(es, jnp.array([True]), jnp.array([1.0]), jnp.array([False]), PC)
    # energy on the onset tick uses saliva BEFORE this tick's command
    assert np.isclose(float(r[0]), PC.energy / PC.food_ticks * PC.b0 - PC.cost)
    assert float(es2.phys.saliva[0]) > 0


def test_loss_terms_only_policy_value_entropy():
    """I4: the PPO loss contains no auxiliary terms."""
    src = inspect.getsource(ppo.make_fns)
    assert "total = pg + pcfg.vf_coef * vl - pcfg.ent_coef * ent" in src
    params = A.init_params(jax.random.PRNGKey(0), 10, A.AgentConfig(hidden=8, embed=8))
    assert set(params) == {"embed", "gru_x", "gru_h", "pi", "v"}


def test_one_training_segment_runs():
    cfg = Config(ecology=EcologyConfig(lifetime=200))
    acfg = A.AgentConfig(hidden=16, embed=8)
    pcfg = ppo.PPOConfig(n_envs=8, seg_len=50, minibatches=2, epochs=1)
    opt, rollout, update = ppo.make_fns(cfg, acfg, pcfg)
    params = A.init_params(jax.random.PRNGKey(0), 10, acfg)
    from pavlov.ecology import generate_batch
    s = generate_batch(jax.random.PRNGKey(1), cfg.ecology, 8)
    es, h = init_env(8), jnp.zeros((8, 16))
    pre = jnp.zeros_like(s.food_onset)
    es, h, key, h0, traj, lv = rollout(params, es, h, s.cues, s.food_onset, s.params.flipped, pre,
                                       jax.random.PRNGKey(2))
    assert traj["obs"].shape == (50, 8, 10) and int(es.t) == 50
    params2, _, aux = update(params, opt.init(params), h0, traj, lv, jax.random.PRNGKey(3))
    assert np.isfinite(float(aux["pg"]))
