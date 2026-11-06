"""Recurrent PPO outer loop (meta-training ≈ evolution).

Loss = clipped policy loss + value loss - entropy bonus. Nothing else (invariant I4).
The recurrent state is carried across segments within a lifetime and reset between lifetimes,
so any within-lifetime learning must live in the GRU state.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pavlov import agent as A
from pavlov.config import Config
from pavlov.ecology import generate_batch
from pavlov.env import env_step, init_env, obs_dim, observe
from pavlov.oracles import _lead_window


@dataclass(frozen=True)
class PPOConfig:
    n_envs: int = 256
    seg_len: int = 100            # truncated-BPTT window; must divide lifetime
    n_lifetime_batches: int = 130 # total env steps = n_lifetime_batches * n_envs * lifetime
    gamma: float = 0.99
    lam: float = 0.95
    lr: float = 3e-4
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.003
    epochs: int = 4
    minibatches: int = 4
    max_grad_norm: float = 0.5


def make_fns(cfg: Config, acfg: A.AgentConfig, pcfg: PPOConfig):
    pc = cfg.physiology
    T = cfg.ecology.lifetime
    L = pcfg.seg_len
    assert T % L == 0, "seg_len must divide lifetime"
    opt = optax.chain(optax.clip_by_global_norm(pcfg.max_grad_norm), optax.adam(pcfg.lr))

    @jax.jit
    def rollout(params, es, h, cues, food, flipped, pre_us, key):
        def body(carry, _):
            es, h, key = carry
            key, k = jax.random.split(key)
            t = es.t
            cues_t = jax.lax.dynamic_index_in_dim(cues, t, axis=1, keepdims=False)
            food_t = jax.lax.dynamic_index_in_dim(food, t, axis=1, keepdims=False)
            pre_t = jax.lax.dynamic_index_in_dim(pre_us, t, axis=1, keepdims=False)
            obs = observe(es, cues_t, food_t, pc)
            h_new, logit, value = A.step(params, h, obs)
            u = (jax.random.uniform(k, logit.shape) < jax.nn.sigmoid(logit)).astype(jnp.float32)
            logp = A.bernoulli_logp(logit, u)
            es, r, info = env_step(es, food_t, u, flipped, pc)
            out = dict(obs=obs, u=u, logp=logp, value=value, r=r, energy=info["energy"],
                       p=jax.nn.sigmoid(logit), pre=pre_t)
            return (es, h_new, key), out

        h0 = h
        (es, h, key), traj = jax.lax.scan(body, (es, h, key), None, length=L)
        # bootstrap
        t_end = es.t
        done = t_end >= T
        ti = jnp.minimum(t_end, T - 1)
        cues_t = jax.lax.dynamic_index_in_dim(cues, ti, axis=1, keepdims=False)
        food_t = jax.lax.dynamic_index_in_dim(food, ti, axis=1, keepdims=False)
        _, _, last_v = A.step(params, h, observe(es, cues_t, food_t, pc))
        last_v = jnp.where(done, 0.0, last_v)
        return es, h, key, h0, traj, last_v

    def gae(rewards, values, last_v):
        def body(carry, xs):
            adv_next, v_next = carry
            r, v = xs
            delta = r + pcfg.gamma * v_next - v
            adv = delta + pcfg.gamma * pcfg.lam * adv_next
            return (adv, v), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), (rewards, values), reverse=True)
        return adv, adv + values

    def loss_fn(params, h0, obs, u, logp_old, adv, ret):
        logits, values, _ = A.unroll(params, h0, obs)
        logp = A.bernoulli_logp(logits, u)
        ratio = jnp.exp(logp - logp_old)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - pcfg.clip, 1 + pcfg.clip) * adv_n).mean()
        vl = 0.5 * ((values - ret) ** 2).mean()
        ent = A.bernoulli_entropy(logits).mean()
        total = pg + pcfg.vf_coef * vl - pcfg.ent_coef * ent
        return total, dict(pg=pg, vl=vl, ent=ent, kl=((ratio - 1) - jnp.log(ratio)).mean())

    @jax.jit
    def update(params, opt_state, h0, traj, last_v, key):
        adv, ret = gae(traj["r"], traj["value"], last_v)
        B = h0.shape[0]
        mb = B // pcfg.minibatches

        def epoch(carry, k):
            params, opt_state = carry
            perm = jax.random.permutation(k, B).reshape(pcfg.minibatches, mb)

            def minibatch(carry, idx):
                params, opt_state = carry
                (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(
                    params, h0[idx], traj["obs"][:, idx], traj["u"][:, idx],
                    traj["logp"][:, idx], adv[:, idx], ret[:, idx])
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), perm)
            return (params, opt_state), aux

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state),
                                                jax.random.split(key, pcfg.epochs))
        return params, opt_state, jax.tree.map(jnp.mean, aux)

    return opt, rollout, update


def train(cfg: Config, acfg: A.AgentConfig, pcfg: PPOConfig, seed: int, out_dir: str,
          log_every: int = 1, ckpt_every: int = 10):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"config": cfg.to_dict(), "agent": asdict(acfg), "ppo": asdict(pcfg), "seed": seed,
                   "jax": jax.__version__}, f, indent=2)
    key = jax.random.PRNGKey(seed)
    key, kp = jax.random.split(key)
    params = A.init_params(kp, obs_dim(cfg.ecology.n_channels), acfg)
    opt, rollout, update = make_fns(cfg, acfg, pcfg)
    opt_state = opt.init(params)
    gen = jax.jit(lambda k: generate_batch(k, cfg.ecology, pcfg.n_envs))
    B, T, L = pcfg.n_envs, cfg.ecology.lifetime, pcfg.seg_len
    log_path = os.path.join(out_dir, "train_log.jsonl")
    t_start = time.time()

    for lb in range(pcfg.n_lifetime_batches):
        key, kg = jax.random.split(key)
        sched = gen(kg)
        pre_us = _lead_window(sched.expected_us, 2)  # training DIAGNOSTIC only (not used for any decision)
        es = init_env(B)
        h = jnp.zeros((B, acfg.hidden))
        ret_sum = jnp.zeros(B)
        energy_sum = jnp.zeros(B)
        sal_sum = 0.0
        p_pre, n_pre, p_other, n_other = 0.0, 0.0, 0.0, 0.0
        auxs = []
        for _ in range(T // L):
            es, h, key, h0, traj, last_v = rollout(params, es, h, sched.cues, sched.food_onset,
                                                   sched.params.flipped, pre_us, key)
            key, ku = jax.random.split(key)
            params, opt_state, aux = update(params, opt_state, h0, traj, last_v, ku)
            ret_sum += traj["r"].sum(0)
            energy_sum += traj["energy"].sum(0)
            sal_sum += float(traj["u"].mean())
            pre = traj["pre"]
            p_pre += float((traj["p"] * pre).sum()); n_pre += float(pre.sum())
            p_other += float((traj["p"] * ~pre).sum()); n_other += float((~pre).sum())
            auxs.append(aux)
        steps = (lb + 1) * B * T
        rec = {
            "lifetime_batch": lb, "env_steps": steps, "wall_s": round(time.time() - t_start, 1),
            "steps_per_s": round(steps / (time.time() - t_start)),
            "lifetime_return": float(ret_sum.mean()), "lifetime_energy": float(energy_sum.mean()),
            "saliva_rate": sal_sum / (T // L),
            "diag_p_pre_us": p_pre / max(n_pre, 1), "diag_p_other": p_other / max(n_other, 1),
            **{k: float(np.mean([a[k] for a in auxs])) for k in auxs[0]},
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if lb % log_every == 0:
            print(f"[{lb:4d}] steps={steps:.2e} ret={rec['lifetime_return']:.2f} "
                  f"sal={rec['saliva_rate']:.3f} p_pre={rec['diag_p_pre_us']:.3f} "
                  f"p_oth={rec['diag_p_other']:.3f} ent={rec['ent']:.3f} kl={rec['kl']:.4f} "
                  f"sps={rec['steps_per_s']}", flush=True)
        if (lb + 1) % ckpt_every == 0 or lb == pcfg.n_lifetime_batches - 1:
            ck = os.path.join(out_dir, "checkpoints")
            os.makedirs(ck, exist_ok=True)
            with open(os.path.join(ck, f"params_{lb + 1:05d}.pkl"), "wb") as f:
                pickle.dump(jax.device_get(params), f)
            with open(os.path.join(ck, "params_latest.pkl"), "wb") as f:
                pickle.dump(jax.device_get(params), f)
    return params
