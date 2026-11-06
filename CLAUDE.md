# CLAUDE.md — Emergent Pavlovian Conditioning in Evolved Agents

Working title: **"Pavlov's dog as an evolved learner: ecological statistics determine when
conditioning, and its insensitivity to outcomes, emerge without being built in."**

Read this file fully before touching code. The scientific value of this project depends entirely
on the **spontaneity invariants** below. A clean null result is publishable; a positive result
produced by a leaked shortcut is worthless.

---

## 1. Research question

In Pavlov's experiment, salivation is (a) elicited by food without training (the unconditioned
response, UR), (b) acquired by an arbitrary bell that predicts food (the conditioned response, CR),
and (c) **not instrumental within the dog's life**: food arrives whether or not the dog
salivates, and in omission tests (Sheffield, 1965) dogs keep salivating even when salivating
cancels the food.

We ask:

- **Q1 (emergence).** If an agent's only effector is salivation, and salivation's only consequence
  is physiological (saliva improves digestion and costs energy), does optimizing that physiology
  over many lifetimes produce an agent that, frozen and placed in Pavlov's lab, acquires a CR to a
  **novel** bell, with no reward, loss or wiring that links cue and response?
- **Q2 (Pavlovian vs instrumental).** Is the emergent CR *Pavlovian* (insensitive to an omission
  contingency) or *instrumental* (it disappears under omission)? We predict the answer is set by
  ecology: **variability in stimulus→outcome (S-O) relations across lifetimes selects for
  conditioning; invariance of response→outcome (R-O) relations selects for outcome-insensitive
  (Pavlovian) responding** (cf. Stephens 1991; Dayan et al. 2006). We map this as a phase diagram.
- **Q3 (phenomena).** Which of Pavlov's phenomena and later ones (extinction, spontaneous recovery,
  differential conditioning, inhibition of delay, blocking, latent inhibition, partial-reinforcement
  extinction effect) emerge, and which fail, compared with Rescorla-Wagner/TD and with animal data?
- **Q4 (mechanism).** What do the agent's internal activations look like: an expectation signal,
  a TD-error-like signal, and is salivation a fixed readout of the expectation (Pavlovian) or gated
  by recent outcomes (instrumental)? Settle it with causal interventions, not only decoding.

The two-loop structure is the whole point:
**outer loop** (meta-training ≈ evolution) is instrumental with respect to fitness;
**inner loop** (one lifetime, weights frozen, learning happens only in recurrent activations) is
where Pavlovian or instrumental behavior appears. Nothing in the outer loop ever sees the lab.

---

## 2. Spontaneity invariants (NON-NEGOTIABLE)

Any change that touches one of these needs an entry in `docs/DECISIONS.md` *before* the change,
and confirmatory results produced under a violated invariant must be discarded.

| # | Invariant | Enforced by |
|---|-----------|-------------|
| I1 | **Reward is physiology only:** `r_t = energy_t − c·u_t`. Nothing in the reward references cues, time since cue, trials, or the lab. | `tests/test_invariants.py::test_reward_is_physiology_only` |
| I2 | **Salivation's only causal path is the saliva level** (and, only in R-O variability arms, the declared `ROContingency` family). Salivation never changes cue or food timing in the primary ecology. | `test_salivation_does_not_affect_world` |
| I3 | **No cue has a fixed meaning across lifetimes** when S-O variability `q = 1`. Cue→channel assignment is resampled each lifetime. The lab "bell" channel has no innate meaning. | `test_cue_permutation` |
| I4 | **No auxiliary losses or heads.** No food-prediction loss, no saliva target, no supervised shaping. Policy loss + value loss + entropy only. | code review + `test_loss_terms` |
| I5 | **Weights are frozen in the lab.** All learning during lab protocols happens in recurrent state. The lab never calls an optimizer. | `lab/runner.py` asserts; `test_lab_frozen` |
| I6 | **The lab is held out.** The omission contingency, Pavlov's acquisition→extinction→rest protocol, and the yoked design never appear in the primary training ecology. `docs/OOD_TABLE.md` lists every lab test and which features are in or out of distribution. | `test_ecology_has_no_omission` |
| I7 | **Experimenter blindness.** Lab schedulers (scripted or human) cannot read agent outputs. The only exception is a declared `Contingency` object (e.g. omission), which is logged. | `Scheduler` API exposes no agent state |
| I8 | **Naive baseline is always measured.** CR to the bell *before* any pairing is reported for every protocol; pre-exposure and truly-random controls are always run. | `lab/protocols.py` |
| I9 | **No tuning on target metrics.** Hyperparameters and ecology defaults are chosen only from (a) oracle-gap analysis of the physiology and (b) training return. Never from CR, omission or phenomena metrics. | `docs/DECISIONS.md` audit trail |
| I10 | **Seed discipline.** Pilot/dev seeds = 100–199. Confirmatory seeds = 0–19, used only after `docs/PREREGISTRATION.md` is frozen (git tag `prereg-v1`). | `configs/` + `scripts/check_seeds.py` |

What counts as "spontaneous" in the paper:
> The CR is spontaneous if it emerges from optimizing a physiology-only objective over an ecology
> in which the test cue has no fixed meaning, the test protocol and contingencies are absent,
> and no component of the agent or objective references the cue→response relation.

Things that are **not** violations but must be stated openly in the paper:
- Saliva improves digestion. This is the functional premise (Domjan 2005; Hollis 1997). It is the
  analogue of evolution, not of the experiment.
- The agent observes food in its mouth, its own saliva level (interoception), its previous action
  and its previous energy gain. This is standard for meta-RL (Wang et al. 2018).

---

## 3. System overview

```
Ecology generator (per lifetime)            Physiology
  cues: N_c binary channels                   saliva S_{t+1} = S_t + k_up·u_t·(1−S_t) − k_down·S_t
  predictive cue(s): random channel,          food in mouth for D ticks after US
    ISI, contingency, delay/trace             energy_t = (E/D)·[b0 + (1−b0)·S_t] while food in mouth
  distractor cues, unsignalled food           r_t = energy_t − c·u_t
  S-O variability q, R-O variability v,
  contingency hazard h, reversion ρ

Agent: GRU actor-critic (meta-RL). Input = [cues, food, S_t, u_{t−1}, r_{t−1}]
       Output = Bernoulli salivation prob p_t (the "drops" readout), value V_t
Outer loop: recurrent PPO across lifetimes (replication: OpenAI-ES)

Lab (frozen agent): Scheduler (scripted | replay | human) + optional Contingency
  protocols: naive, acquisition (CS+ bell / CS− metronome), extinction, rest, recovery,
             omission + yoked, truly random, CS-alone/US-alone, phenomena battery
```

The CR readout is the salivation probability `p_t` (lower variance than sampled actions).
The environment uses sampled `u_t`.

---

## 4. Repo layout

```
pavlov-rl/
  CLAUDE.md
  docs/
    IMPLEMENTATION_PLAN.md   phases, gates, compute
    PREREGISTRATION.md       hypotheses, metrics, tests (freeze → tag prereg-v1)
    DECISIONS.md             dated log of every design change + reason (append-only)
    LITERATURE.md            prior art, novelty positioning, citations TO VERIFY
    OOD_TABLE.md             per lab test: what is in- vs out-of-distribution
  configs/                   yaml: ecology/*.yaml, agent/*.yaml, sweep/*.yaml
  src/pavlov/
    physiology.py            saliva + digestion (pure functions, JAX)
    ecology.py               lifetime sampler + step function (JAX, vmappable)
    oracles.py               hand-written reference policies (never, reflex, oracle-anticipatory, always)
    agent.py                 GRU actor-critic
    ppo.py                   recurrent PPO (outer loop)
    es.py                    evolution-strategies outer loop (replication)
    lab/
      scheduler.py           Scheduler + Contingency API (blind by construction)
      protocols.py           Pavlov protocols + controls + phenomena battery
      runner.py              frozen-agent execution, logging
    analysis/
      metrics.py             CR index, OSI, discrimination, recovery ratio, etc.
      stats.py               Holm-Bonferroni, bootstrap CIs, trend tests
      probes.py              linear probes, interventions on hidden state
    baselines/
      rescorla_wagner.py, td_csc.py, instrumental_ac.py, pavlovian_control.py
  scripts/                   train.py, run_lab.py, sweep.py, make_figures.py, check_seeds.py
  tests/
  results/<exp_id>/<seed>/   config.yaml, git_hash, metrics.json, checkpoints/ (gitignored large files)
  labbench/                  browser lab bench (human-experimenter mode, blind toggle, logging)
```

---

## 5. Commands

```bash
source .venv/bin/activate
pytest -q                                   # always green before any commit
python scripts/oracle_table.py              # physiology sanity check: anticipation must pay
python scripts/train.py --config configs/exp/pilot.yaml --seed 100
python scripts/run_lab.py --run results/pilot/100 --protocol pavlov_standard
python scripts/check_seeds.py               # refuses confirmatory seeds before tag prereg-v1
```

---

## 6. Conventions

- **Stack:** Python 3.13, JAX (CPU unless a GPU is available), optax, numpy/scipy/pandas, matplotlib.
  No PyTorch. Environments are pure JAX functions so they can be `jit`/`vmap`-ed over lifetimes.
- **Determinism:** every run takes an explicit integer seed. Every result directory stores the
  resolved config, the git hash and the JAX version.
- **Statistics (carried over from the earlier CompPsy-RL paper):** the unit of analysis is the
  independent meta-training seed (N=20 confirmatory). Within a seed, average over lab "subjects"
  (test lifetimes). Use two-tailed tests with Holm-Bonferroni for preregistered families, and
  report effect sizes with 95% bootstrap CIs. For near-deterministic quantities, report magnitudes
  and CIs rather than inflated p-values. Report null and failed phenomena as boundary conditions.
- **Never** edit a confirmatory result, re-run a confirmatory seed after looking at it, or delete
  a failed run. Failed runs are logged in `DECISIONS.md`.
- **Claims language:** "CR-like", "Pavlovian-like signature". No claims of subjective experience.
- Figures are regenerated only from `results/` by `scripts/make_figures.py`.
- Citations in `docs/LITERATURE.md` marked `[VERIFY]` must be checked against the source before
  they go into the paper.
