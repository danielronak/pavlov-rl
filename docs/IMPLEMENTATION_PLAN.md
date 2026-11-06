# Implementation Plan

Each phase ends with a **gate**: a concrete check that must pass before the next phase starts.
Gates exist so that we never tune toward the result we want (invariant I9).

Target venues, in order of fit:
- **Journal:** PLoS Computational Biology or eLife (computational account plus animal benchmarks),
  or Psychological Review if the theory section (the phase diagram) is strong.
- **Conference:** CCN or CogSci for an early version. NeurIPS/ICLR is possible if the
  meta-learning analysis (phase diagram + mechanism) stands on its own as an ML contribution.

---

## Headline experiment design

### Training ecology ("the wild")

> **Current primary ecology = amendment A3** (see `docs/DECISIONS.md`): 4 channels, delay cues
> only, one stimulus at a time, world event rate U(1/40, 1/20), metabolic cost 0.01 (A1). The
> table below is the original draft; where it conflicts, A3/A1 and `src/pavlov/config.py` win.
> Trace conditioning, compound cues (blocking, overshadowing) and overlapping stimuli are now
> out-of-sample lab tests.

Each lifetime lasts T ticks (default 3000; must be ≥ the longest lab protocol).

| Element | Default (to be fixed in Phase 1 by oracle-gap analysis, not lab results) |
|---|---|
| Cue channels | N_c = 6 binary channels |
| Predictive cues per lifetime | 1 or 2, on channels drawn at random (prob q = S-O variability; with prob 1−q, channel 0 is the predictive one) |
| CS→US interval (ISI) | U{3,…,12} ticks, per cue per lifetime |
| Delay vs trace | 50/50 per cue (trace gap U{1,…,4}) |
| Contingency P(US \| CS) | U(0.5, 1.0) |
| Distractor cues | remaining channels, onset rate U(0.005, 0.03), uncorrelated with food |
| Unsignalled food | rate U(0, 0.005) per tick |
| Inter-event interval | geometric, mean ≈ 30 ticks |
| Non-stationarity | hazard h per tick that a predictive cue goes "silent" (contingency → 0); when it changes again, it reverts to its original contingency with prob ρ, otherwise resamples |
| R-O variability v | prob per lifetime that saliva's digestion effect is altered (primary family: efficiency slope flipped, so saliva hurts digestion). No omission contingency in any family used for the primary test (I6). |

### Physiology

- Saliva: `S_{t+1} = S_t + k_up·u_t·(1−S_t) − k_down·S_t`, with k_up ≈ 0.35, k_down ≈ 0.15.
- Food stays in the mouth for D = 3 ticks after the US, and is eaten automatically.
- Energy per food tick: `(E/D)·[b0 + (1−b0)·S_t]`, with E = 1 and b0 = 0.3.
- Metabolic cost per salivation tick: c ≈ 0.02.
- Saliva lags behind the command, so **anticipatory** salivation pays and constant salivation
  does not. Phase 1 checks this with oracles before any learning happens.

### Agent

- GRU with 64–128 units. Input: `[cues(N_c), food, S_t, u_{t−1}, r_{t−1}]`.
- Heads: Bernoulli salivation logit and value.
- Outer loop: recurrent PPO with hidden state carried across truncated-BPTT windows of 128 ticks.
  Replication: OpenAI-ES (literal "evolution").
- Weights are frozen in the lab (I5).

### Lab (Pavlov's protocol, held out)

1. **Naive:** bell (CS+) and metronome (CS−) presented alone, 5 trials each. Report CR.
2. **Acquisition:** 40 trials, CS+ → food (ISI 6 delay), CS− unpaired, random ITI 20–60.
3. **Extinction:** 30 CS+ trials without food.
4. **Rest:** 600 ticks with no cues and no food.
5. **Recovery test:** 5 CS+ trials without food (spontaneous recovery).

Controls, each run with the same frozen agents:
- truly random (Rescorla 1967);
- CS-alone;
- US-alone (sensitization);
- pre-exposure to the bell (latent inhibition).

**Omission test (decisive).** If salivation happens at any tick during CS+ on a trial, that
trial's food is cancelled. Each omission subject gets a **yoked** partner that receives exactly
the same food schedule with no contingency (Sheffield 1965; Williams & Williams 1969).

**Positive-contingency test (dissociation control).** Food is delivered only if the agent
salivates during the CS. This checks that the agent can follow response contingencies at all.

### Metrics

- **CR index:** mean p(saliva) over the last 3 ticks of the CS before the US, minus the pre-CS
  baseline (10 ticks).
- **Discrimination:** CR(CS+) − CR(CS−).
- **Omission Sensitivity Index:** OSI = 1 − CR_omission / CR_yoked over the last omission
  block, clipped to [0, 1]. 0 = fully Pavlovian, 1 = fully instrumental.
- **Recovery ratio:** CR(first recovery trial) / CR(end of acquisition).
- **Inhibition of delay:** slope of p(saliva) over CS time in long-CS trials.
- **Food cost of the CR under omission:** food lost relative to the yoked partner (the analogue
  of "paying for the bias" in CompPsy-RL Exp 3).

---

## Phases

### Phase 0 — Scaffolding (½ day)
- venv, deps, repo layout, pytest, config loader, seed checker, `results/` conventions.
- **Gate:** `pytest` green; `check_seeds.py` refuses seeds 0–19 without the `prereg-v1` tag.

### Phase 1 — Physiology + ecology + oracles (1–2 days)
- `physiology.py` and `ecology.py` as pure JAX functions (vmap over lifetimes).
- `oracles.py` with these policies:
  - *never* (never salivates);
  - *always* (salivates every tick);
  - *reflex* (salivates only while food is in the mouth);
  - *oracle-anticipatory* (knows the predictive cue and its ISI, salivates k ticks before the US);
  - *cue-blind clock* (salivates at the average ISI after **any** cue).
- `scripts/oracle_table.py` reports mean lifetime energy per oracle across the ecology.
- Gate criteria, defined **before** any agent exists (I9), implemented in `scripts/oracle_table.py`:
  - G1: reflex > never (the UR is worth having);
  - G2: cue-oracle > reflex (anticipation pays);
  - G3: cue-oracle > any-cue (knowing *which* cue predicts food pays, so association is needed);
  - G4: always < reflex (constant salivation is no shortcut).
  - G2 and G3 must exceed 10% of the cue-oracle − reflex gap, with the paired 95% CI excluding 0.
  - Log the final values in `DECISIONS.md`.
- Invariant tests I1–I3 and I6.
- **Gate:** the ordering holds with margins, and the invariant tests pass.

### Phase 2 — Agent + outer loop (2–4 days)
- GRU actor-critic and recurrent PPO; throughput benchmark on 4 CPUs.
- Pilot training on seed 100 at default ecology (q = 1, v = 0).
- **Gate (training only, no lab metrics):** mean lifetime energy exceeds *reflex* and
  approaches *oracle-anticipatory* (≥ 50% of the gap). If not: raise the budget or network size,
  or fix bugs. Never touch the reward (I1) or add heads (I4).

### Phase 3 — Lab (2 days)
- `Scheduler` API with no access to agent state (I7) and a `Contingency` API for omission and
  positive contingencies; all protocols and controls above; `runner.py` with frozen weights (I5).
- Unit tests: scheduler blindness, yoked replay is exact, and the lab never calls an optimizer.
- `docs/OOD_TABLE.md` filled in.
- **Gate:** the oracle policies give the expected lab signatures (e.g. *never* has CR = 0 and
  the hand-wired Pavlovian baseline has OSI ≈ 0). This validates the measurement pipeline
  independently of the agent.

### Phase 4 — Pilot lab results (1 day) — GO/NO-GO
- Run the lab on pilot seeds 100–102.
- Outcomes:
  - **CR emerges:** proceed.
  - **No CR, but training succeeded:** this is a real finding. Agents may have found a
    non-associative strategy. Diagnose with oracles before changing anything.
  - **Training failed:** go back to Phase 2.
- Any change made after seeing pilot lab results is logged in `DECISIONS.md` with a reason,
  and will be disclosed in the paper.

### Phase 5 — Baselines (2 days)
- **B1:** instrumental actor-critic trained from scratch *within* each lifetime (no meta-learning).
- **B2:** hand-wired Pavlovian control (Modayil & Sutton 2014): a TD prediction of food plus a
  fixed threshold reflex. This is the "architectural" reference.
- **B3:** Rescorla-Wagner and TD with a complete serial compound (CSC) representation, for the
  phenomena battery.
- All baselines go through the identical lab pipeline.

### Phase 6 — Preregistration freeze
- Finalize `PREREGISTRATION.md` (hypotheses, metrics, tests, exclusion rules, N), freeze the
  analysis code, `git tag prereg-v1`. Optionally post on OSF.

### Phase 7 — Confirmatory headline (N = 20 seeds 0–19)
- Two arms: **Pavlovian ecology** (q = 1, v = 0) and **instrumental ecology** (q = 1, v = 0.5).
- Tests: H1–H3 from the preregistration.

### Phase 8 — Phase diagram
- Grid over q ∈ {0, 0.25, 0.5, 0.75, 1} × v ∈ {0, 0.1, 0.25, 0.5}, N = 5 per cell (N = 10 if
  compute allows).
- Plotted per cell: CR to the novel bell (does conditioning exist?) and OSI (is it Pavlovian?).
- Predicted regions:
  - low q → an innate response to channel 0 only, and no conditioning to the bell;
  - high q, low v → Pavlovian;
  - high q, high v → instrumental.

### Phase 9 — Phenomena battery
- Extinction, spontaneous recovery (scaled by ρ), differential conditioning, inhibition of
  delay, blocking, overshadowing, latent inhibition, partial-reinforcement extinction effect,
  second-order conditioning.
- Each phenomenon is scored against a directional criterion fixed in the preregistration and
  compared with RW/TD and animal data. Failures are reported.
- Secondary ecology sweep over ρ (spontaneous recovery should scale with ρ).

### Phase 10 — Mechanism
- Linear probes on GRU state: time-to-US, P(US), a TD-error-like signal.
- Interventions: clamp or patch the expectation direction and measure the salivation change;
  compare how strongly salivation depends on expectation vs. on recent reward (the
  Pavlovian-vs-instrumental readout). Across seeds, correlate the size of the reward-gating
  component with OSI.

### Phase 11 — Robustness
- OpenAI-ES outer loop; LSTM vs GRU; hidden size 32/64/128; physiology parameters ±25%;
  lifetime length.

### Phase 12 — Human-experimenter lab bench
- Browser page: a frozen agent exported to JS (GRU forward pass only), with bell, metronome and
  food buttons.
- Blind mode hides salivation (I7). Every press is logged.
- `scripts/replay_human.py` replays logged schedules to all 20 seeds, with yoked and
  shuffled-bell controls.
- In the paper this is a robustness result (agents condition under irregular human schedules)
  and a public demo.

### Phase 13 — Paper
- Figures: (1) paradigm schematic; (2) acquisition/extinction/recovery curves vs Pavlov 1927;
  (3) omission vs yoked; (4) phase diagram; (5) phenomena scorecard vs RW/TD/animals;
  (6) mechanism.

---

## Compute

The container has 4 CPU cores, 15 GB RAM and no GPU. Phase 2 measures throughput first.

Rough expectation: ~10⁸ environment steps per seed.
- At ≥ 100k steps/s on CPU, that is ~20 min/seed. The headline (2 arms × 20 seeds) is ~13 h,
  and the phase diagram (20 cells × 5 seeds) is ~33 h.
- If throughput is far lower, options are: rent a single GPU for Phases 7–8, cut the phase
  diagram to N = 5 on a coarser grid, or shorten lifetimes (only if T stays ≥ the longest lab
  protocol).

---

## Risks and what we do about them

| Risk | Response |
|---|---|
| Anticipation barely pays, so agents never salivate early | Phase 1 oracle margins, fixed before agents exist |
| CR to every cue (no discrimination) | CS− and distractor controls; report as generalization if it happens |
| "Omission insensitivity" is just slow inner learning | Dissociation controls: the same agent must show fast extinction (S-O learning) and follow the positive contingency |
| Reviewer: "meta-learning makes this inevitable" | The phase diagram shows it is conditional on ecology; low-q cells must show *no* conditioning |
| Reviewer: "you built the phenomena into the ecology" | Ecology defaults fixed a priori from oracle gaps; phenomena not represented in the ecology (blocking, latent inhibition, PREE, inhibition of delay) serve as out-of-sample tests |
| Prior art appears | `LITERATURE.md` is a living document; re-run the search before Phase 6 |
| Compute | See above |
