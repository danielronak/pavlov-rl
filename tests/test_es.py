import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from pavlov import agent as A
from pavlov.config import Config, EcologyConfig
from pavlov.es import ESConfig, centered_ranks, make_generation


def test_centered_ranks():
    r = np.asarray(centered_ranks(jnp.array([3.0, -1.0, 10.0, 0.0])))
    assert np.allclose(sorted(r), [-0.5, -1 / 6, 1 / 6, 0.5]) and r[2] == 0.5


def test_generation_runs_and_moves_theta():
    cfg = Config(ecology=EcologyConfig(lifetime=100))
    acfg = A.AgentConfig(hidden=8, embed=8)
    e = ESConfig(pop=8, lifetimes=2)
    theta, unravel = ravel_pytree(A.init_params(jax.random.PRNGKey(0), 10, acfg))
    opt = optax.adam(e.lr)
    gen = make_generation(cfg, acfg, e, unravel, opt)
    th2, _, fm, fc = gen(theta, opt.init(theta), jax.random.PRNGKey(1))
    assert np.isfinite(float(fm)) and np.isfinite(float(fc))
    assert float(jnp.abs(th2 - theta).max()) > 0
