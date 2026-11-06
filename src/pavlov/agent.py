"""GRU actor-critic in pure JAX. Heads: salivation logit (Bernoulli) and value. Nothing else
(invariant I4: no auxiliary heads)."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class AgentConfig:
    hidden: int = 64
    embed: int = 64


def _dense(key, n_in, n_out, scale=1.0):
    w = jax.nn.initializers.orthogonal(scale)(key, (n_in, n_out))
    return {"w": w, "b": jnp.zeros(n_out)}


def init_params(key, obs_dim: int, cfg: AgentConfig):
    ks = jax.random.split(key, 8)
    H, E = cfg.hidden, cfg.embed
    return {
        "embed": _dense(ks[0], obs_dim, E, jnp.sqrt(2.0)),
        "gru_x": _dense(ks[1], E, 3 * H),
        "gru_h": {"w": jnp.concatenate([jax.nn.initializers.orthogonal()(k, (H, H)) for k in jax.random.split(ks[2], 3)], 1),
                  "b": jnp.zeros(3 * H)},
        "pi": _dense(ks[3], H, 1, 0.01),
        "v": _dense(ks[4], H, 1, 1.0),
    }


def gru_cell(p, h, x):
    H = h.shape[-1]
    gx = x @ p["gru_x"]["w"] + p["gru_x"]["b"]
    gh = h @ p["gru_h"]["w"] + p["gru_h"]["b"]
    z = jax.nn.sigmoid(gx[..., :H] + gh[..., :H])
    r = jax.nn.sigmoid(gx[..., H:2 * H] + gh[..., H:2 * H])
    n = jnp.tanh(gx[..., 2 * H:] + r * gh[..., 2 * H:])
    return (1.0 - z) * n + z * h


def step(params, h, obs):
    """One tick. Returns (h_new, logit, value)."""
    x = jnp.tanh(obs @ params["embed"]["w"] + params["embed"]["b"])
    h = gru_cell(params, h, x)
    logit = (h @ params["pi"]["w"] + params["pi"]["b"])[..., 0]
    value = (h @ params["v"]["w"] + params["v"]["b"])[..., 0]
    return h, logit, value


def unroll(params, h0, obs_seq):
    """obs_seq [L, B, D] -> logits [L, B], values [L, B], h_final."""
    def body(h, o):
        h, lg, v = step(params, h, o)
        return h, (lg, v)

    h, (logits, values) = jax.lax.scan(body, h0, obs_seq)
    return logits, values, h


def bernoulli_logp(logit, u):
    return u * jax.nn.log_sigmoid(logit) + (1.0 - u) * jax.nn.log_sigmoid(-logit)


def bernoulli_entropy(logit):
    p = jax.nn.sigmoid(logit)
    return -(p * jax.nn.log_sigmoid(logit) + (1.0 - p) * jax.nn.log_sigmoid(-logit))
