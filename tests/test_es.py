import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.flatten_util import ravel_pytree

from pavlov import agent as A
from pavlov.env import obs_dim
from pavlov.config import Config, EcologyConfig
from pavlov.es import ESConfig, centered_ranks, make_generation


def test_centered_ranks():
    r = np.asarray(centered_ranks(jnp.array([3.0, -1.0, 10.0, 0.0])))
    assert np.allclose(sorted(r), [-0.5, -1 / 6, 1 / 6, 0.5]) and r[2] == 0.5


def test_generation_runs_and_moves_theta():
    cfg = Config(ecology=EcologyConfig(lifetime=100))
    acfg = A.AgentConfig(hidden=8, embed=8)
    e = ESConfig(pop=8, lifetimes=2)
    theta, unravel = ravel_pytree(A.init_params(jax.random.PRNGKey(0), obs_dim(cfg.ecology.n_channels), acfg))
    opt = optax.adam(e.lr)
    gen = make_generation(cfg, acfg, e, unravel, opt)
    th2, _, fm, fc = gen(theta, opt.init(theta), jax.random.PRNGKey(1))
    assert np.isfinite(float(fm)) and np.isfinite(float(fc))
    assert float(jnp.abs(th2 - theta).max()) > 0


def test_resume_is_exact(tmp_path):
    """Run 4 generations straight vs 2 + resume + 2: identical parameters."""
    import pickle
    from pavlov.es import train_es
    cfg = Config(ecology=EcologyConfig(lifetime=100))
    acfg = A.AgentConfig(hidden=8, embed=8)
    e4 = ESConfig(pop=4, lifetimes=2, generations=4)
    e2 = ESConfig(pop=4, lifetimes=2, generations=2)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    train_es(cfg, acfg, e4, 0, a, ckpt_every=1)
    train_es(cfg, acfg, e2, 0, b, ckpt_every=1)
    train_es(cfg, acfg, e4, 0, b, ckpt_every=1, resume=True)
    pa = pickle.load(open(f"{a}/checkpoints/params_latest.pkl", "rb"))
    pb = pickle.load(open(f"{b}/checkpoints/params_latest.pkl", "rb"))
    assert all(np.array_equal(x, y) for x, y in zip(jax.tree.leaves(pa), jax.tree.leaves(pb)))


def test_init_from_ancestor(tmp_path):
    import pickle
    from pavlov.es import train_es
    cfg = Config(ecology=EcologyConfig(lifetime=100))
    acfg = A.AgentConfig(hidden=8, embed=8)
    anc = A.init_params(jax.random.PRNGKey(42), obs_dim(4), acfg)
    path = str(tmp_path / "anc.pkl")
    pickle.dump(jax.device_get(anc), open(path, "wb"))
    out = str(tmp_path / "c")
    train_es(cfg, acfg, ESConfig(pop=4, lifetimes=2, generations=1, lr=0.0, weight_decay=0.0), 0, out,
             init_from=path, ckpt_every=1)
    p = pickle.load(open(f"{out}/checkpoints/params_latest.pkl", "rb"))
    assert all(np.allclose(x, y) for x, y in zip(jax.tree.leaves(anc), jax.tree.leaves(p)))
