# The Pavlov project, explained from the ground up

*Companion to `paper/draft.md` and `paper/ieee/main.pdf`. Written 2026-11-19; robustness check added 2026-11-25. Everything here is
traceable to a file in this repository; the file is named where it matters.*

**How to use this document.** Read Part A to understand the science you need. Part B is the
design and *why* each piece is the way it is. Part C is the history, including what failed. Part D
is every result, figure and number. Part E is my critical review. Part F tells you how to rewrite
the paper from this document. Part G is a glossary, a file map and a cheat sheet of numbers.
If you understand Parts A–E you can defend every sentence in the paper.

---

## Contents

- Part A — Background
  - A1 Pavlov's experiment and its vocabulary
  - A2 The phenomena that came after Pavlov
  - A3 Pavlovian vs instrumental: the omission test
  - A4 Models of conditioning: Rescorla–Wagner and TD
  - A5 Reinforcement learning, recurrent networks and meta-learning
  - A6 Evolution strategies
  - A7 Why the evolution of learning is a question at all
- Part B — The design and the reasons for it
  - B1 The research question and the two-loop idea
  - B2 The spontaneity invariants
  - B3 The physiology (formulas explained)
  - B4 The ecology (the simulated world)
  - B5 The agent
  - B6 The outer loop and the training criterion (Phase-2 gate)
  - B7 The laboratory
  - B8 Measures: CR, discrimination, OSI, M_ro
- Part C — The history of the project
- Part D — Results
  - D1 The pre-registration and the statistics
  - D2 The confirmatory run
  - D3 Confirmatory results, figure by figure
  - D4 Exploratory 1: fixed vs variable cue meanings
  - D5 Exploratory 2: inside the network
  - D6 Exploratory 3: the phenomena battery
  - D7 Exploratory 4: baseline models
  - D8 Exploratory 5: the compound-cue follow-up (does blocking evolve?)
  - D9 Exploratory 6: robustness to network size and physiology
- Part E — Critical review
- Part F — How to rewrite the paper from this document
- Part G — Glossary, file map, numbers cheat sheet, reproduction

---

# Part A — Background

## A1 Pavlov's experiment and its vocabulary

Ivan Pavlov (Nobel Prize 1904, *Conditioned Reflexes* 1927) studied digestion in dogs. He noticed
that dogs began to salivate before food reached them — at the sight of the attendant, the sound of
footsteps. He turned this into an experiment:

1. Food in the mouth makes a dog salivate with no training. Food is the **unconditioned stimulus
   (US)**; salivation to food is the **unconditioned response (UR)**.
2. A neutral signal — a bell, a metronome — makes no salivation at first.
3. If the signal is repeatedly followed by food, the dog starts salivating to the signal alone. The
   signal is now a **conditioned stimulus (CS)** and the salivation it evokes is the **conditioned
   response (CR)**.

Words used throughout this project:

- **Pairing / trial.** One presentation of the CS, possibly followed by the US.
- **Acquisition.** The phase in which CS and US are paired and the CR grows.
- **CS+ / CS−.** In *differential conditioning*, one signal (CS+, our "bell") is always followed by
  food, another (CS−, our "metronome") never is. The animal learns to respond to CS+ only. The
  difference CR(CS+) − CR(CS−) is **discrimination**; it rules out the possibility that the animal
  is simply more aroused by any sound.
- **Delay conditioning.** The CS stays on until the US arrives (what we use). In *trace*
  conditioning there is a gap between CS offset and US.
- **ISI (inter-stimulus interval).** Time from CS onset to US.
- **ITI (inter-trial interval).** Time between trials.
- **Extinction.** After acquisition, the CS is presented without food; the CR declines.
- **Spontaneous recovery.** After extinction and a rest, the CR partly comes back. Pavlov took this
  as evidence that extinction suppresses rather than erases the learning.

## A2 The phenomena that came after Pavlov

These are the "classic phenomena" tested in the battery (Part D6). Each tells something about *how*
an animal learns.

- **Contingency (Rescorla 1967).** Pairing alone is not enough; the CS must *predict* the US. In
  the **truly-random control**, the CS and the US occur equally often but independently. Animals
  then show little CR, even though CS and US sometimes coincide. We use this as a control.
- **Sensitisation / pseudo-conditioning.** Food alone can make an animal respond more to anything.
  The **US-alone control** gives food without the CS; if the CS then evokes a response, it is not
  learning about the CS.
- **Blocking (Kamin 1969).** Train A → food. Then train the compound AX → food. Test X alone: the
  animal responds little to X, because A already predicted the food and X added no information.
  Compared with an **overshadowing control** (same AX training without prior A training), X is
  "blocked". Blocking is the signature of *error-driven* learning: you only learn when you are
  surprised.
- **Latent inhibition (Lubow & Moore 1959).** Presenting a CS many times without consequence
  before pairing slows later learning about it. The animal has learned to ignore it.
- **Partial-reinforcement extinction effect (PREE).** If the CS was followed by food only 50% of
  the time during acquisition, the CR extinguishes *more slowly* than after 100% reinforcement.
  Error-correcting models predict the opposite, which makes PREE a famous model failure.
- **Inhibition of delay (Pavlov).** With a long CS, salivation is low early in the CS and rises as
  the US approaches. It shows that the animal learns *when* food comes, not just *that* it comes.
- **Second-order conditioning.** After A → food, pair B → A (no food). B now evokes a CR, learned
  from a signal, not from food.

## A3 Pavlovian vs instrumental: the omission test

There are two broad ways a response can be controlled.

- **Instrumental (operant) control.** The response is performed *because of its consequences*: a
  rat presses a lever because pressing produces food. The relevant relation is
  **response → outcome (R-O)**. Change the consequences and the response changes.
- **Pavlovian control.** The response is triggered by a *prediction* of the outcome, whatever the
  consequences of responding. The relevant relation is **stimulus → outcome (S-O)**.

Salivation in Pavlov's setup could be either: the dog might salivate because salivating "earns" the
food. The **omission test** separates them. Arrange things so that salivating *cancels* the food on
that trial. An instrumental responder learns to stop salivating, because that is what gets food. A
Pavlovian responder keeps salivating, because the bell still predicts food (on the trials where it
did not salivate) and the response is tied to the prediction. Sheffield (1965) did this with dogs,
and Williams & Williams (1969) with pigeon key-pecking ("negative automaintenance"). The animals
kept responding although it cost them the reward. This is the puzzling, "irrational" property of
Pavlovian responses that motivates this project.

To interpret an omission test you need a **yoked control**. The omission subject's responding
changes how many USs it gets, and fewer USs alone could reduce responding. So each omission subject
is paired with a yoked partner that receives *exactly the same* sequence of CSs and USs, with no
contingency. If the omission subject responds less than its yoked partner, the difference is due to
the contingency (instrumental sensitivity); if it responds the same, it is insensitive (Pavlovian).

## A4 Models of conditioning: Rescorla–Wagner and TD

**Rescorla–Wagner (RW, 1972).** Each CS *c* has an associative strength *V_c*. On each trial, the
prediction is the sum of the strengths of the CSs present, and every present CS is updated by the
prediction error:

  V_c ← V_c + α · (λ − Σ_{present} V)

where λ = 1 if the US occurred and 0 if not, and α is a learning rate. Read it as: "learn in
proportion to how surprised you were". Because the error is shared among all present cues, RW
predicts **blocking** (once A predicts λ, the error on AX trials is ~0, so X learns nothing). RW has
no notion of time within a trial and does not predict spontaneous recovery, latent inhibition or
PREE.

**Temporal-difference learning (TD; Sutton & Barto 1990).** TD learns a value V_t = expected
discounted future US from moment *t*:

  δ_t = r_t + γ·V_t − V_{t−1},   w ← w + α·δ_t·x_{t−1}

with δ the TD error, r the US, γ the discount (how much the future counts), and x the features of the
moment. With a **complete serial compound (CSC)** representation, there is one feature for every
(cue, time since cue onset), so the model learns a value that *ramps up* across the CS toward the US
(a version of inhibition of delay). TD also predicts blocking and second-order conditioning. Its
error δ resembles the firing of dopamine neurons (Schultz, Dayan & Montague 1997).

Both are **prediction** models. To make them behave, one attaches a response to the prediction —
e.g. "salivate in proportion to V". That link is put in by hand. Our agents have no such link
designed in; whether and how it appears is the question.

## A5 Reinforcement learning, recurrent networks and meta-learning

**Reinforcement learning (RL).** An *agent* observes the world, takes *actions*, and receives
*rewards*. A *policy* maps observations to action probabilities. The goal is to maximise the
*return*, the sum of rewards over time.

**Recurrent neural network (RNN) / GRU.** A network with a hidden state h_t that is updated at
every tick from the previous state and the new input, so it can remember. We use a **gated
recurrent unit (GRU)**:

  z = σ(W_z x + U_z h + b_z),  r = σ(W_r x + U_r h + b_r)
  n = tanh(W_n x + r ⊙ (U_n h) + b_n),  h_new = (1 − z) ⊙ n + z ⊙ h

z decides how much of the old state to keep, r how much of it to use when computing the candidate n.
The 64 numbers in h are the agent's entire short-term memory.

**Meta-learning ("learning to learn").** If you train an RNN over *many different tasks*, where each
task needs to be figured out on the fly, the optimiser cannot store the answer to any one task in
the weights. Instead it builds weights that make the hidden state *learn* the current task from
experience. After training, the weights are fixed, yet the agent still adapts within an episode —
the learning happens in the activations. This is **meta-reinforcement learning (meta-RL)** (Wang et
al. 2018). Here, one "task" is one **lifetime** in which a randomly chosen cue predicts food, and
the outer optimiser plays the role of **evolution**: it shapes, over many lifetimes, a brain that
learns within a lifetime.

## A6 Evolution strategies

**OpenAI evolution strategies (ES; Salimans et al. 2017 — reference still to be verified and added to the paper).** A gradient-free optimiser. At each
**generation**:

1. Draw random perturbations ε_1 … ε_n of the weights θ (Gaussian noise).
2. Evaluate each perturbed network θ + σ·ε_i over several lifetimes; its **fitness** F_i is the mean
   lifetime return.
3. Move θ in the direction of the perturbations that did well:
   ĝ = (1 / (n·σ)) · Σ_i F̃_i · ε_i, then update θ with the Adam optimiser.

Details we use: **antithetic sampling** (each ε is also evaluated as −ε, which reduces noise),
**centred ranks** (F̃_i are ranks rescaled to [−0.5, 0.5], so outliers do not dominate), **common
random numbers** (the paired ±ε members see the same lifetimes and action-sampling noise), and
**weight decay** (shrinks weights slightly each step). Population 64 (32 noise vectors × 2 signs),
8 lifetimes per member, σ = 0.03, learning rate 0.02, weight decay 0.005.

Why ES rather than a gradient method (PPO)? Recurrent PPO was tried first and failed the training
criterion repeatedly (Part C). ES optimises whole-lifetime return directly, has no problem with long
credit assignment across thousands of ticks (no backpropagation through time), and is a natural
analogue of evolution. PPO remains a "replication arm" for later.

## A7 Why the evolution of learning is a question at all

Learning is costly (time, errors while learning). An inborn response is free but rigid. Theory
(Stephens 1991; Katsnelson et al. 2012) says learning should evolve when the world **changes between
generations** (so no fixed response is right for everyone) but is **predictable within a lifetime**
(so what you learn early stays useful). In our ecology this maps onto two knobs:
- **S-O variability, q**: how often the cue → food relation is reshuffled between lifetimes.
- **R-O variability, v**: how often the consequences of responding differ between lifetimes.

The original hypothesis was: q makes conditioning evolve; v makes it instrumental. The first part
held (and an exploratory test supports it). The second part did not (Part C, Part D).

**Who has evolved learners before (and how we differ).** Burgos (1997) evolved neural networks for
Pavlovian conditioning, but they learned with a designed learning rule. Izquierdo, Harvey and Beer
(2007, 2008) evolved recurrent networks with fixed weights that learn an association in their
dynamics, exactly our "learning in activations", but evolution rewarded doing the association task.
Kozielska and Weissing (2023, 2024) evolved networks with a prediction-error learning rule and found,
like us, that learning does not evolve in fairly constant worlds. What is new here: the objective
is physiology only (no fitness term for association), there is no learning rule at all, and the
evolved learner is tested in a held-out lab including the omission test. Details and the full
reference check: `docs/LITERATURE_CHECK.md`.

---

# Part B — The design and the reasons for it

## B1 The research question and the two-loop idea

The question: *if salivation is the only thing an agent can do, and salivation's only consequence is
physiological, does optimising physiology over many lifetimes produce an agent that, frozen and put
in Pavlov's lab, acquires a CR to a novel bell — and is that CR Pavlovian?*

The design has **two loops**:

- **Outer loop (evolution).** ES changes the weights to maximise digestion energy over many simulated
  lifetimes. It is "instrumental with respect to fitness": everything is optimised for its
  consequences on energy. It never sees the lab.
- **Inner loop (a lifetime).** Weights are fixed. The agent experiences cues and food; anything it
  learns, it learns in its hidden state. The lab is one such lifetime with an experimenter's
  schedule.

The point of the design: any Pavlovian (or instrumental) behaviour in the lab is not programmed; it
is what evolution found useful in the ecology, expressed in a test the agent has never seen.

## B2 The spontaneity invariants

These ten rules (CLAUDE.md §2) are what make "spontaneous" a defensible word. Each has an automated
test or a code-level guarantee.

| # | Rule | Why it matters |
|---|---|---|
| I1 | Reward is physiology only: r_t = energy_t − c·u_t | If reward mentioned cues, we would be training the CR directly. |
| I2 | Salivation only changes the saliva level (except the declared R-O families in sister arms) | If salivating changed food timing, the CR could be instrumental by design. |
| I3 | No cue has a fixed meaning when q = 1 (cue→channel mapping resampled every lifetime) | Otherwise the lab bell could be innately meaningful and the "CR" would be inborn. |
| I4 | No auxiliary losses (no "predict food" loss) | An auxiliary prediction loss would build the prediction in. |
| I5 | Weights frozen in the lab | Lab learning must be the agent's own within-lifetime learning, not more training. |
| I6 | The lab is held out (omission, Pavlov's protocol, yoking never occur in training) | Otherwise the agent could have been selected for the test itself. |
| I7 | Experimenter blindness: the lab scheduler cannot read the agent | Prevents the schedule from adapting to the agent (a hidden feedback loop). |
| I8 | Naive baseline always measured | Proves the CR was not there before pairing. |
| I9 | No tuning on target metrics; parameters chosen from oracle analysis and training return only | Prevents fishing for the result. |
| I10 | Seed discipline: pilot seeds 100–199; confirmatory 0–99 only after the pre-registration freeze | Prevents "try seeds until it works". |

What *is* built in, and must be said openly: saliva helps digestion (this is the functional premise
— it plays the role of evolution's raw material, not of the experiment), and the agent observes food
in its mouth, its saliva level, its previous action and its previous reward (standard meta-RL inputs).

## B3 The physiology (formulas explained)

Every tick, the agent chooses u_t ∈ {0, 1} (salivate or not), sampled from its probability p_t.

**Saliva level** S ∈ [0, 1):

  S_{t+1} = S_t + k_up · u_t · (1 − S_t) − k_down · S_t,   k_up = 0.35, k_down = 0.15

Each salivation tick fills 35% of the remaining room; saliva decays by 15% per tick. Salivating
for 2–3 ticks before food arrives raises S to about 0.5–0.6 at food time (0 → 0.35 → 0.53 → 0.61).

**Food and digestion.** Food stays in the mouth for D = 3 ticks after it arrives. Energy per tick
with food in the mouth:

  energy_t = (E / D) · [ b0 + (1 − b0) · S_t ],   E = 1, b0 = 0.3

With no saliva, digestion efficiency is 30%; with S near 1 it is ~100%. Because saliva rises over
several ticks, saliva produced *before* food arrives makes the first ticks of digestion more
efficient. That is the entire functional advantage of anticipation.

**Reward:** r_t = energy_t − c · u_t, c = 0.01 (salivating costs a little).

**Why these numbers?** They were chosen in Phase 1 by **oracle analysis** before any agent existed
(`scripts/oracle_table.py`, `docs/DECISIONS.md` 2026-11-06). Oracles are hand-written policies:

- *never*: never salivate;
- *reflex*: salivate only while food is in the mouth (the UR alone);
- *any_cue*: salivate whenever any cue is on (cue-blind);
- *pred_cue_on*: knows which cue is predictive, salivates while it is on;
- *cue_oracle*: knows the predictive cue and its ISI, salivates k ticks before each expected US;
- *always*: always salivate.

The **Phase-1 gate** demanded, with confidence intervals: reflex > never (G1); cue_oracle > reflex
(anticipation pays, G2); cue_oracle > any_cue (knowing *which* cue matters, G3); always < reflex
(salivating blindly is wasteful, G4); and, after amendment A1, pred_cue_on > reflex (G5, a
"learnability" step so that evolution can climb gradually). Amendment A1 lowered the cost c from
0.02 to 0.01 — the smallest single change that passed G5 — because the first agents failed to learn
anticipation (Part C). Later gates G6–G8 (A3) check one stimulus at a time, at least 30 pairings per
lifetime, and that associating the right cue is worth ≥ 25% of the anticipation gap.

**Typical returns in the final ecology** (held-out lifetimes, 20 ancestors): never 7.14,
reflex 11.61, pred_cue_on 13.50, cue_oracle 14.63; evolved agents 13.33.

## B4 The ecology (the simulated world)

A **lifetime** = 3000 ticks. Parameters (`src/pavlov/config.py`, final values after amendments):

- **4 cue channels** (think of 4 lights/sounds); amendment A3 reduced this from 6.
- **One stimulus at a time** (A3): the world produces *events* at a per-tick rate drawn per lifetime
  from U(1/40, 1/20). Each event presents one channel; no new event until the current one (and its
  ISI window) is over. This mirrors Pavlov's lab, where one signal is presented at a time.
- **Predictive cues**: one per lifetime, two with probability 0.5. A predictive cue is followed by
  food at a fixed ISI (drawn per cue from 3–12 ticks) with probability = its **contingency**, drawn
  from U(0.5, 1). Distractor channels are never followed by food.
- **Identical statistics** (A2): predictive and distractor channels have the same onset rates and
  durations, so the *only* way to tell them apart is their relation to food. (Earlier versions leaked
  cue type through timing statistics; this was a validity bug, found and fixed.)
- **Delay cues only** (A3): the cue stays on through the US tick.
- **S-O variability q** = 1: every lifetime, the roles are randomly permuted across channels (I3).
- **Non-stationarity**: a predictive cue can go silent and come back (hazard 1/1500 per tick); when
  it comes back it keeps its old contingency with probability ρ = 0.5. This gives a little reason to
  track changes within a life (relevant for extinction and recovery).
- **Unsignalled food**: occasional food without any cue (rate up to 0.005 per tick).
- **R-O variability v** = 0 for ancestors. Sister arms use v = 0.5 with one of two families:
  - *digestion flip* (amendment A4): in flipped lifetimes, efficiency = b0·(1 − S): saliva *hurts*
    digestion. At S = 0 both lifetime types give b0, so the type is revealed only by salivating —
    the agent must learn the R-O relation within the lifetime;
  - *partial omission*: in "gated" lifetimes, salivating at any tick inside a predictive cue's window
    makes that trial's food arrive with probability 0.5·p instead of p. This is a mild, probabilistic
    version of the lab's omission test, present during evolution.

## B5 The agent

- **Input** each tick: [4 cue bits, food-in-mouth bit, saliva level S_t, previous action u_{t−1},
  previous reward r_{t−1}]. Why each: cues and food are the sensory world; S is interoception (you
  can feel your mouth); u and r let the network relate its own actions to outcomes (essential for any
  learning within a life).
- **Network**: input → 64-unit tanh embedding → GRU with 64 units → two linear heads: a salivation
  logit (p_t = sigmoid(logit)) and a value estimate (only used by PPO; ignored by ES).
- **No auxiliary heads or losses** (I4).
- **The CR readout** in the lab is p_t, the salivation probability (less noisy than the sampled
  0/1 action). The environment uses the sampled action.

## B6 The outer loop and the training criterion (Phase-2 gate)

ES as in A6. Ancestors: 4000 generations from random weights (≈ 6.5 billion simulated ticks).
Training is checkpointed every 100 generations with the full optimiser state, so an interrupted run
resumes *exactly* (bit-for-bit) — important because the cloud machine restarted seven times during
the confirmatory run.

**Phase-2 gate** (`scripts/eval_training.py`): did training actually work? On 256 held-out
lifetimes, compute the fraction of the anticipation gap closed:

  gap_fraction = (R_agent − R_reflex) / (R_cue_oracle − R_reflex)

The gate passes if gap_fraction ≥ 0.5. Only training-ecology returns are used (I9), never lab
results. For v > 0 arms, "aware" oracles are used (oracles that also know the lifetime type), because
the blind oracles would be beaten trivially.

**Important**: in the confirmatory study, passing the gate is *not* an inclusion criterion (intention
to treat). All seeds were analysed; gate pass rates are reported.

## B7 The laboratory

`src/pavlov/lab/`. The agent is **frozen** (the lab code never imports an optimiser; tested). A lab
"subject" = one run of the frozen agent from a zero hidden state. Each protocol runs **32 subjects**.
Schedules are generated from a fixed **lab seed (2026)**, so every agent meets *exactly* the same 32
schedules. The bell (CS+) and metronome (CS−) are assigned to two random channels per subject (so the
result does not depend on a particular channel). CS lasts 6 ticks plus the US tick (ISI 6); ITI is
uniform 15–35 ticks.

**Blindness (I7).** The scheduler builds the whole schedule before the run from the protocol, the
RNG and the number of channels — it receives no agent output. The only way behaviour can change what
happens is a **declared contingency** (omission or positive), applied by the runner and logged.

**Protocols** (each begins with a **naive phase**: 4 bell + 4 metronome trials without food, I8):

| Protocol | What happens | What it tests |
|---|---|---|
| pavlov_standard | 25 bell+food and 25 metronome trials (shuffled), then 20 bell-alone (extinction), 300 empty ticks (rest), 5 bell-alone (recovery) | acquisition, discrimination, extinction, spontaneous recovery |
| truly_random | 25 bell + 25 metronome, 25 food deliveries at random times; then 5 bell tests | contingency vs mere pairing |
| cs_alone | bell and metronome, no food; 5 bell tests | exposure alone |
| us_alone | 25 food deliveries without the bell + 25 metronome; 5 bell tests | sensitisation |
| omission | 40 bell+food trials where salivating during the bell cancels that food, 20 metronome trials; plus a yoked partner | Pavlovian vs instrumental |
| positive | 40 bell trials where food comes *only if* the agent salivates during the bell; plus yoked partner | sensitivity to a positive R-O relation |

**Phase-3 gate (validation of the measurement pipeline).** Before any agent entered the lab, the
whole pipeline was run on hand-built policies with known answers (`scripts/validate_lab.py`, 19/19
checks passed): *never* (all CRs 0, OSI undefined), *bell_reader* (Pavlovian by construction:
salivates to the bell from the start — the naive check catches it; OSI 0), *withholder*
(instrumental by construction: stops salivating under omission — OSI 1). This shows the measures can
detect each kind of behaviour.

## B8 Measures

**CR on one trial** (`src/pavlov/analysis/metrics.py`):

  CR = mean p over the 3 ticks before the US (or where it would have been)
     − mean p over the 10 ticks before CS onset

i.e. how much salivation rises in anticipation, relative to the agent's own baseline. It ranges
roughly from −1 to 1. A CR of 0.6 means salivation probability rises by 0.6 before food.

**Per-run measures** (means over the 32 subjects):

| Name | Definition | Meaning |
|---|---|---|
| D_naive | CR(bell) − CR(metronome), 4+4 naive trials | Response before learning (should be ≈ 0) |
| D_acq | same, last 10 acquisition trials of each | Learned discrimination |
| CR_acq_cs_plus | bell CR, last 10 acquisition trials | Acquisition level |
| CR_test_paired | bell CR, first 5 extinction trials | Test after pairing (5 unreinforced trials, matched to the controls) |
| CR_test_truly_random | bell CR in the 5 test trials after truly-random training | Control |
| CR_test_us_alone | same after US-alone | Control |
| CR_ext_last5 | bell CR, extinction trials 16–20 | End of extinction |
| CR_recovery_first | bell CR, recovery trials | After rest |
| OSI | clip(1 − CR_omission / CR_yoked, 0, 1), last 15 of 40 bell trials | 0 = ignores omission (Pavlovian), 1 = stops completely (instrumental). Undefined if CR_yoked < 0.05 (no CR to test). |
| positive_index | CR_positive − CR_yoked | Effect of a positive contingency |
| US delivered under omission | fraction of planned bell foods actually received | Cost of responding |
| M_ro | p(salivate \| food in mouth) in normal lifetimes − same in flipped lifetimes, ticks 800–2999 of 256 held-out digestion-variable lifetimes | Did the agent learn, within a life, that saliva helps or hurts digestion? |
| gap_fraction, gate_pass | Phase-2 gate | Training quality |

Why OSI is clipped at 0: omission CR can exceed yoked CR (it did, slightly); the index is meant to
measure *reduction*, so negative values are set to 0. Note this biases the mean OSI slightly
*upward*, which makes "OSI < 0.2" a conservative claim.

---

# Part C — The history of the project (what we did, in order, and why)

**Phase 0–1 (Nov 6): scaffolding, physiology, oracles.** Repo, JAX environment, invariant tests,
oracle gate G1–G4 passed with the first physiology.

**Phase 2, rounds 1–2: recurrent PPO fails.** Agents learned the reflex but closed only 4–28% of
the anticipation gap (gate needs 50%). Diagnosis: no gradual path from reflex to anticipation
(knowing *which* cue predicts was worth nothing without knowing *when*), and the policy became
deterministic too early. Amendment A1 lowered the salivation cost (0.02 → 0.01) so that a
"learnability" step exists (G5). Still failed. A diagnostic showed agents salivated to *any*
stimulus, without learning which one predicts food — a non-associative strategy.

**Validity bug and fix (A2).** We found that predictive and distractor cues had different timing
statistics, so cue type leaked without any food pairing. Fixed: all channels share identical
statistics.

**Round 3: ES vs PPO.** ES (new outer loop) and PPO with longer BPTT both plateaued at ~35% of the
gap. The ecology (6 channels, overlapping cues, trace cues) was too hard to discover anticipation in.

**Amendment A3: a Pavlov-like world.** 4 channels, delay cues only, one stimulus at a time, event
rate chosen by a pre-stated rule (first rate passing oracle gates G1–G8). Trace conditioning became a
lab-only test.

**Round 4: ES passes** (gap 0.63 and 0.66, seeds 109, 110); PPO still fails. ES became primary.
Training-ecology diagnostic: within a lifetime, salivation to the predictive cue rose while
salivation to distractors fell — genuine within-lifetime association.

**Phase 3: the lab** was built, parameters fixed before any agent entered, and validated with
oracles (19/19).

**Phase 4: pilot lab results (seeds 109, 110)** — the first time anyone looked at lab results: clear
acquisition (+0.81, +0.72), naive ≈ 0, one-trial acquisition, partial extinction, weak/no spontaneous
recovery, controls ≈ 0, OSI 0.00 (food delivered 12%). From here every decision is flagged as made
having seen lab data.

**The H2 saga (Nov 7): trying to evolve an instrumental CR.**
1. *v = 0.5, digestion flip, first family (seeds 112, 113).* One seed got stuck at "never
   salivate"; the other partly learned. **Design flaw found**: in flipped lifetimes, efficiency was
   1 − 0.7·S, i.e. 1.0 at S = 0 vs 0.3 normally — the first meal revealed the lifetime type through
   food energy alone, with no response-outcome learning. The arm was selecting the wrong thing.
2. *Amendment A4*: flipped efficiency = b0·(1 − S), equal to normal at S = 0, so only salivating
   reveals the type. New oracle gate G10 checks that probing then adapting pays. *Seeds 114, 115*
   stuck at "never" (the network's output saturated at "off"; ES perturbations could not escape) and
   were killed by a machine restart; the ES code then lacked exact resume — fixed afterwards.
3. *Common-ancestor design.* Instead of evolving from scratch, take ancestors that already condition
   (109, 110) and split each into a stable and a variable sister lineage (2000 more generations).
   Variable sisters evolved genuine within-lifetime R-O learning about digestion (they stopped
   salivating in flipped lifetimes after a few meals) — but their bell CR was still OSI 0. Extending
   to 4000 generations: same.
4. *Partial-omission family.* Make salivating during a predictive cue partly cancel food during
   evolution (seeds 120, 121). Evolution **abandoned the CR**: no anticipatory salivation at all,
   reflex only. OSI undefined. The instrumental solution ("salivate unless it costs food") would
   have paid more (oracle 12.02 vs evolved 11.66) but was not found.
5. *Search-limit check* (seeds 122, 123): 4× population and 4× lifetimes per member. Identical
   result (gate +0.10, CR abandoned). The "maybe the search was too weak" loophole stays open but is
   narrowed.

**Pilot conclusion.** H2 (R-O variability → omission-sensitive CR) was not supported anywhere. Three
evolutionary outcomes: stable world → Pavlovian CR; digestion variability → separate R-O learning +
weaker Pavlovian CR; US-delivery variability → no CR. This became **H2′**: *ecology decides whether a
CR exists, not whether it is Pavlovian.* It is pilot-derived and is disclosed as such.

**A metric fix (disclosed).** OSI is undefined when the yoked CR < 0.05 (previously a 0/0 case gave a
bogus 0). Only affected no-CR cases.

**Pre-registration frozen (Nov 16)** — git commit `b4d07cf`, tag `prereg-v1` (the tag must be pushed
from the PI's machine; the cloud session cannot push tags).

**Confirmatory run (Nov 16–18)** — 80 runs, ~200,000 ES generations, 7 machine restarts (all resumed
exactly), 320 jobs, 0 failures, no exclusions. Analysis run unchanged.

**After the confirmatory run (Nov 18–19)**, all exploratory and planned before running: figures;
the q-control; mechanism v1 and v2; phenomena battery; baseline models; paper draft; IEEE version.

**Compound-cue follow-up (Nov 19).** The battery found no blocking, and the obvious excuse was that
the ancestors' world never showed two cues at once. To test it, a `p_compound` option was added to
the world (off = the old world, bit for bit; checked by re-running an old evaluation and by a new
test). Ten compound sisters were evolved from ancestors 0–9; predictions C1/C2 and the analysis
script were committed before launch. Result: no blocking (C1 failed), still Pavlovian (C2 held).
Because "maybe compounds didn't matter to evolution" was the next obvious excuse, a cross-evaluation
was planned (after seeing C1, labelled post hoc) and run: compounds did matter, and the compound
sisters adapted, just not by blocking (D8).

Every one of these steps has a dated entry in `docs/DECISIONS.md` with the reason, and whether it was
made after seeing lab results (y/n).

---

# Part D — Results

## D1 The pre-registration and the statistics

**What a pre-registration is.** A document, frozen *before* the data exist, that fixes the
hypotheses, the exact measures, the tests, the exclusion rules and the sample size. It prevents
"HARKing" (hypothesising after results are known) and p-hacking. Ours is `docs/PREREGISTRATION.md`,
frozen at `b4d07cf`. Later changes may only be appended as dated amendments (one so far: a date
erratum).

**Design.** 20 ancestors (seeds 0–19), each evolved 4000 generations; each split into three sisters
(seeds 25+a, 50+a, 75+a), 2000 generations each. Replacements 20–24 if an ancestor is excluded.

**Unit of analysis = the seed (an independent evolutionary history), N = 20.** Lab subjects within a
seed are averaged (they are not independent replications of evolution).

**The 11 tests** (one family):

| Test | Claim | Test used |
|---|---|---|
| H1a | D_acq > 0 | one-sample t, two-tailed |
| H1b | D_acq − D_naive > 0 | paired t |
| H1c | CR_test_paired − CR_test_truly_random > 0 | paired t |
| H1d | CR_test_paired − CR_test_us_alone > 0 | paired t |
| H3a | 0.5·CR_acq − CR_ext_last5 > 0 (extinction to below half) | one-sample t |
| H3b | ancestor OSI < 0.2 | one-sided t against 0.2 |
| H3c | positive_index within ±0.1 | TOST equivalence |
| H2′a | digestion-sister OSI < 0.2 | one-sided t |
| H2′b | M_ro(digestion sister) − M_ro(ancestor) > 0 | paired t (manipulation check) |
| H2′c | D_acq(partial-omission) − D_acq(stable) < 0 | paired t |
| H2′d | D_acq(partial-omission) < 0.1 | one-sided t |

H1 is supported iff H1a–d all are; H3 iff H3a–c; H2′ iff H2′a–d.

**Statistics, in plain words.**
- *t-test.* Asks whether a mean differs from a value, given the spread across seeds.
  t = mean / (SD / √N). A **paired** t-test does this on per-seed differences.
- *Two-tailed, direction required.* A two-tailed test detects differences in either direction; we
  count it as support only if the effect is in the predicted direction (stricter than one-tailed).
- *One-sided bound test.* "Mean < 0.2": a one-tailed t-test against 0.2.
- *TOST (two one-sided tests)* for equivalence: to claim the mean lies in (−0.1, +0.1), show it is
  above −0.1 *and* below +0.1; the p-value is the larger of the two.
- *Holm–Bonferroni.* With 11 tests, some would pass by chance. Sort the p-values; compare the
  smallest with 0.05/11, the next with 0.05/10, and so on; stop at the first failure. Adjusted
  p-values ("p (Holm)") are reported.
- *Bootstrap 95% CI.* Resample the 20 seeds with replacement 10,000 times, recompute the mean each
  time; the middle 95% of those means is the interval. It shows the plausible range of the effect.
- *Cohen's d_z* = mean difference / SD of differences: effect size in units of between-seed spread.
  0.8 is conventionally "large"; ours are often 3–7.
- *Degenerate rule.* If all values are identical (SD = 0), a t-test is undefined; the test is decided
  by the value (p = 0 on the predicted side, else 1) and flagged.
- *Minimum for OSI tests.* OSI tests need ≥ 10 seeds with a defined OSI, else "untestable".
- *Why such tiny p-values?* Between-seed spread is small because every seed is a large average; with
  N = 20 and d_z ≈ 3–7, p-values like 10⁻¹² are expected. CLAUDE.md asks us to emphasise magnitudes
  and CIs for such near-deterministic quantities.

**Exclusion (only rule).** Exclude an ancestor if training diverged (NaN) or its final return is
below the reflex oracle. None was excluded.

## D2 The confirmatory run

`scripts/run_confirmatory.py` (restart-safe driver: it skips finished steps and resumes runs from
checkpoints) ran: train → Phase-2 gate → R-O learning check → all lab protocols, for all 80 runs.
`scripts/analyze_confirmatory.py` (frozen at the tag) produced `results/confirm/analysis.md`.

## D3 Confirmatory results, figure by figure

**Training.** Ancestors: return 13.33 vs reflex 11.61, cue oracle 14.63 → gap fraction 0.57; 17/20
passed the gate (stable sisters 19/20, digestion sisters 0/20 with gap 0.08, partial-omission 0/20).
In the training world, p(salivate) was 0.43 in the ticks before an expected US vs 0.05 elsewhere.

**All 11 tests supported** (`results/confirm/analysis.md`):

| Test | Mean [95% CI] | d_z | p (Holm) |
|---|---|---|---|
| H1a acquisition | +0.599 [0.534, 0.665] | 3.9 | 2.6e−12 |
| H1b acquired − naive | +0.602 [0.535, 0.673] | 3.7 | 4.9e−12 |
| H1c paired − truly random | +0.358 [0.312, 0.406] | 3.2 | 4.3e−11 |
| H1d paired − US alone | +0.401 [0.349, 0.457] | 3.1 | 5.5e−11 |
| H3a extinction | +0.122 [0.080, 0.163] | 1.3 | 3.8e−05 |
| H3b ancestor OSI | 0.010 [0.001, 0.023] | — | 2.6e−17 |
| H3c positive index | −0.003 [−0.009, 0.002] | — | 2.8e−18 |
| H2′a digestion OSI (n = 18) | 0.083 [0.020, 0.165] | — | 0.0042 |
| H2′b R-O learning evolved | +0.676 [0.580, 0.754] | 3.3 | 3.5e−11 |
| H2′c partial-omission − stable D_acq | −0.727 [−0.771, −0.684] | −7.1 | 5.3e−17 |
| H2′d partial-omission D_acq | 0.000 [0.000, 0.000] | — | (degenerate-like; report magnitude) |

**Figure 1 — acquisition/extinction curves** (`figures/fig1_acquisition_extinction.png`).
- x-axis: trial number within each phase (acquisition 1–25, extinction 1–20, after rest 1–5);
  y-axis: CR (rise in p(salivate) before food).
- Blue = bell (CS+), orange = metronome (CS−). Line = mean over 20 seeds; band = 95% bootstrap CI.
- Read it: on trial 1 the bell evokes almost nothing (≈ 0.07, a small response to any new stimulus,
  equal for the metronome). By trial 2 the bell CR is ~0.6 and stays there: **one-trial acquisition**.
  The metronome stays near 0.05: **discrimination**. In extinction the CR decays over ~10 trials to
  ~0.2 and stays: **partial extinction**. After the rest it is still ~0.2: **no spontaneous recovery**.
- Left: ancestors; right: stable sisters (higher plateau, ~0.77, same shape).

**Figure 2 — controls** (`fig2_controls.png`). Left: discrimination before pairing (grey dots ≈ 0) vs
after 25 pairings (blue, ≈ 0.6). Right: the bell's CR at test after pairing (≈ 0.47), after
truly-random exposure (≈ 0.12) and after food alone (≈ 0.08). Dots are seeds; the large marker with
a vertical bar is the mean and its 95% CI. Message: the response is learning about the bell's
predictive relation to food, not exposure or sensitisation.

**Figure 3 — omission vs yoked** (`fig3_omission.png`). Four panels (ancestor, stable,
digestion-variable, partial-omission sisters). Each thin line is one seed: left end = yoked partner's
CR, right end = omission CR. Large markers: means with CIs. Text: fraction of the available food the
omission subject actually got. Message: lines are flat or rising — responding does not drop under
omission, although ancestors receive only 16% of the food (stable 19%, digestion 46%). The
partial-omission sisters have no CR at all (both 0; they receive 100% of the food because they never
salivate early).

**Figure 4 — the four evolutionary histories** (`fig4_ecology_arms.png`). Three panels, each with one
column of dots per arm:
- *Does a CR exist?* D_acq: ancestors 0.60, stable 0.73, digestion 0.10, partial-omission 0.00.
- *Did R-O learning evolve?* M_ro: ≈ −0.02 in ancestors and stable sisters, 0.65 in digestion sisters,
  0 in partial-omission sisters.
- *Is the response instrumental?* OSI: ≈ 0 for ancestors and stable sisters, 0.08 for digestion
  sisters (two seeds near 0.5; two seeds undefined), undefined for partial-omission sisters. The dashed
  line is the pre-registered margin 0.2.
Message: R-O learning can evolve (digestion arm), but it does not make the bell response
instrumental; and when responding costs food, evolution switches the response off instead.

## D4 Exploratory 1: fixed vs variable cue meanings (the q-control)

**Why.** All confirmatory agents had q = 1. A reviewer could ask whether the variability is what makes
learning evolve. **Design** (`scripts/run_svar.py`; predictions logged first): 5 ancestors at q = 0
(roles never permuted: channel 0 always predictive, channel 1 the second predictive cue in half of
lifetimes, channels 2–3 never predictive) and 5 at q = 0.5. The lab bell lands on channel 0 for 10 of
the 32 subjects, on channel 1 for 11, on channels 2–3 for 11.

**Pre-written analysis** (`results/svar/analysis.md`): naive CR to a bell on channel 0: q = 0 0.85,
q = 0.5 0.28, q = 1 0.11; learned discrimination on channels 1–3: 0.37, 0.46, 0.59. Prediction P1
("no learning on channels 1–3 at q = 0") **failed** in its stated form; P2 and P3 held.

**Post-hoc split** (`scripts/svar_channel_split.py`, flagged as post hoc): channel 1 learned at
q = 0 (0.80) and q = 0.5 (0.92); channels 2–3 not learned at q = 0 (−0.05) or q = 0.5 (−0.005),
learned at q = 1 (0.63).

**Figure 5** (`fig5_so_variability.png`): left = response before pairing ("inborn"), right = after
pairing ("learned"); x = q; colours = which channel the bell was on. Read: blue (channel 0) starts
high on the left at q = 0 and falls as q increases (inborn response disappears); green (channels 2–3)
is flat at 0 until q = 1 on the right (no learning until those channels can matter often); orange
(channel 1) is learned at every q.

**Interpretation.** Fixed meaning → inborn response; variable and often-relevant meaning → learning;
rarely relevant (≈ 1 lifetime in 5 at q = 0.5) → ignored. This resembles **preparedness / selective
associability** in animals (e.g. rats associate nausea with tastes more easily than with lights).
Small N and post hoc; needs a dedicated pre-registered test.

## D5 Exploratory 2: inside the network

**Reconstruction.** We re-ran each frozen network on the exact recorded lab inputs to recover its
hidden states; the recomputed p matches the recorded p to 2.4·10⁻⁴ (float16 precision).

**v1 — expectation probe** (`scripts/mechanism.py`). A *linear probe* is a ridge regression from the
64 hidden units to a target; if it predicts held-out data, the information is "linearly decodable".
Target: V = fraction of earlier presentations of the current cue that were followed by food.
- Held-out R² = 0.59 (59% of variance explained): an expectation *is* represented.
- *Cosine* between the probe direction and the salivation readout weights: 0.07 — nearly orthogonal.
- *Steering*: add α × (probe direction) to the state during naive trials. If it were the causal
  expectation, salivation should rise with α. It did not — a random direction had the same (U-shaped)
  effect. So the probe found a *correlate*, not the cause. (Reported as a negative result.)
- Under omission, the decoded expectation was not lower than in yoked partners (+0.017): consistent
  with Pavlovian behaviour.

**v2 — activation patching** (`scripts/mechanism_patch.py`). Take the hidden state just before a bell
trial late in acquisition (the "donor"), and copy its k most-changed units into the same subject's
state before its first *naive* bell trial (the "recipient"), then let the network run. If the CR now
appears on a trial that comes before any learning, those units carry the learned association.
- k = 8 units: 76% [61, 90] of the learned CR transfers; k = 16: all of it (>100%, because the
  recipient also has a fresher baseline).
- Donor from *another subject trained with the bell on the same channel*: transfers equally (81% at
  k = 8). Donor trained with the bell on a *different* channel: nothing transfers, responding drops
  (−26% at k = 16).
- Conclusion: the memory is compact (~8–16 of 64 units), **channel-specific** ("channel X predicts
  food"), and coded the same way across individuals of one evolved network.
- Methods honesty: the planned control (a CS− donor) was found to be conceptually flawed on a dev run
  before the real run and was replaced; logged.

## D6 Exploratory 3: the phenomena battery

`src/pavlov/lab/phenomena.py`, `scripts/run_phenomena.py`, `scripts/analyze_phenomena.py`. Designs
and criteria fixed in advance. Between-subject groups of 32 within each seed; Holm over 5 per arm.

| Phenomenon | Design | Ancestors (mean [CI]) | Shown? |
|---|---|---|---|
| Blocking | A+ ×20, AX+ ×10, test X; control: B+ pretraining | −0.005 [−0.033, 0.023] | No |
| Latent inhibition | A− ×20 then A+ ×10 vs B− pre-exposure | −0.015 [−0.037, 0.008] | No |
| PREE | 100% vs 50% (equal US count), then 20 extinction | +0.054 [0.009, 0.096] (Holm p 0.15; artefact) | No |
| Inhibition of delay | 24-tick CS: slope of p across CS | −0.001 [−0.003, 0.001] | No (12-tick CS: small positive slope) |
| Second-order | A+, then B→A, test B vs unpaired | +0.012 [−0.002, 0.026] | No (stable sisters: tiny yes) |

Why "artefact" for PREE: the 50% group ends acquisition lower (0.56 vs 0.66); the absolute
extinction curves coincide from trial 3. The ratio differs only because of the denominator.

**Figure 6** (`fig6_phenomena.png`): one panel per phenomenon, each on its own scale; dots = seeds;
shaded half = the direction animals show. Every mean sits on or near 0.

**Interpretation.** The evolved learner is **fast (one-shot), channel-specific and non-competitive**.
It does not share prediction error among cues (no blocking), does not lower attention to familiar
cues (no LI), and does not track reinforcement history (no PREE). The first plausible reason, that
the ecology never presents compound cues, was tested directly and is *not* enough (D8). For LI and
PREE, the untested candidate reason is that contingencies rarely change within a life.

## D7 Exploratory 4: baseline models

`src/pavlov/baselines/models.py` — the models are lab policies seeing the same observations, run
through the identical lab: RW (α = 0.3), TD-CSC (α = 0.1, γ = 0.9), an instrumental tabular
Q-learner trained from scratch within the session (α = 0.2, γ = 0.9, softmax β = 5). All have the food
reflex. Parameters: textbook defaults, fixed before running.

Key comparisons (`results/baselines/scorecard.md`):
- Core Pavlovian results: agents match RW and TD (acquisition, contingency, omission insensitivity).
- **Blocking**: RW −0.28, TD −0.16 (strong); agents −0.005 (none). This is the clearest
  mechanistic difference.
- Extinction: RW goes to 0, TD to 60%, agents to 33%.
- Inhibition of delay: only TD shows a ramp.
- The instrumental learner never learns to anticipate within a session (lab-length trial and error is
  not enough; evolution provides what it lacks).
- None shows spontaneous recovery or latent inhibition.

## D8 Exploratory 5: the compound-cue follow-up (does blocking evolve?)

**Why.** Blocking is the result that made Rescorla–Wagner famous, and both RW and TD show it in our
lab. Our agents did not. If the reason is simply that their world never showed two cues together,
then adding such moments to the world should make blocking evolve. If it does not, the absence is a
real property of the evolved learner.

**What changed in the world** (`src/pavlov/ecology.py`, `configs/exp/compound.yaml`). With
probability `p_compound` = 0.3, an event shows a second channel at the same time as the first. The
second channel is drawn at random from the other three, fresh at every event. Everything else is the
ancestors' world. The extra randomness uses its own random stream, so with `p_compound` = 0 the world
is exactly the old one.

**Design.** Common ancestor, like the sisters in the main experiment: copies of ancestors 0–9 evolved
for 2000 more generations in the compound world (seeds 140–149, development range). Each is compared
with the stable sister of the *same* ancestor, which evolved 2000 generations in the old world.
Pairing by ancestor removes the ancestor-to-ancestor differences. N = 10, exploratory.

**Predictions, written before the run** (`docs/DECISIONS.md`, 2026-11-19):
- C1: compound sisters show blocking (CR(X) in the blocking group minus the control < 0), and more
  than their stable sisters.
- C2: they still condition (D_acq > 0.3) and stay omission-insensitive (OSI < 0.2).
- If C1 fails, missing compounds alone do not explain missing blocking.

**How the blocking test works** (same as D6). Blocking group: A+ ×20, then AX+ ×10, then X alone.
Control ("overshadowing") group: B+ ×20, then AX+ ×10, then X alone. In the blocking group, A already
predicts the food when X is added, so a competitive learner learns little about X. Effect = CR(X)
blocking − CR(X) control; animals and RW/TD give a negative number.

**Results** (`results/compound/analysis.md`):

| Measure | Compound sisters | Stable sisters |
|---|---|---|
| Acquired discrimination | 0.67 [0.60, 0.75] | 0.75 [0.71, 0.80] |
| OSI | 0.03 [0.00, 0.07] | 0.00 [0.00, 0.00] |
| Gap closed (gate ≥ 0.5) | 0.60 [0.58, 0.63] (10/10 pass) | 0.65 [0.62, 0.67] |
| Blocking | +0.012 [−0.018, 0.044] | +0.013 [−0.020, 0.050] |

- C1 **not supported**. CR to X was 0.24 in the blocking group and 0.23 in the control: X was learned
  equally either way. Compound − stable: −0.002 [−0.065, 0.059], p = 0.96.
- C2 **supported**.
- Other battery effects: unchanged in kind. One small difference: a flatter response across the
  24-tick CS (−0.004, p = 0.04, uncorrected; not something to build on).

**The post-hoc cross-evaluation** (`scripts/compound_crosseval.py`, `results/compound/crosseval.md`).
The next excuse would be "compounds were too rare or too cheap to matter, so evolution ignored them."
To check, every compound and stable sister played the *same* 256 held-out lifetimes of *both* worlds;
the score is the training return (the quantity evolution acts on).

| Tested in | Compound sisters | Stable sisters | Difference (paired) |
|---|---|---|---|
| Compound world | 16.25 [16.17, 16.33] | 15.86 [15.80, 15.92] | +0.39 [0.30, 0.49], p = 3·10⁻⁵ |
| Ancestral world | 13.45 [13.38, 13.54] | 13.56 [13.49, 13.62] | −0.10 [−0.21, −0.01], p = 0.09 |

Reference oracles in the compound world: never 8.51, reflex 13.94, pred_cue_on 16.43, cue oracle 17.76.

**What it means.** Compounds mattered: the compound sisters became specialists, better in their world
and slightly worse in the old one. So evolution did respond to compounds, but the solution it found is
not RW-style cue competition. Blocking is not a by-product of being a good conditioner, even in a
world with compounds.

**Why might blocking still not pay? (my reasoning, untested.)** In this world the second cue is a
fresh random channel each time, so there is never one fixed redundant cue in a lifetime, which is
what the lab test uses. Wrongly salivating to a cue that was sometimes paired with food costs only
c = 0.01 per tick. A world where the same redundant cue recurs throughout a lifetime, or where false
alarms cost more, would be the stronger test (about 3 hours of compute).

**What the adaptation is** was not investigated. Candidates: responding differently to two-cue
events, or weighing within-lifetime evidence about distractor channels differently.

## D9 Exploratory 6: robustness to network size and physiology

**The question.** Two settings were fixed early and never varied: the size of the network (64 units,
chosen for speed) and the physiology (k_up 0.35, b0 0.3, chosen by the oracle analysis). Would the
core result survive if either had been set differently?

**The design** (planned and committed before any run; `docs/DECISIONS.md` 2026-11-19). Four
variants of the confirmatory ancestor, each evolved from scratch for 4000 generations with the same
ES settings, three development seeds each:

| Variant | Change | Seeds |
|---|---|---|
| GRU 128 | twice as many recurrent units | 153–155 |
| GRU 32 | half as many | 150–152 |
| Weak physiology | k_up and the saliva benefit (1 − b0) both × 0.75: k_up 0.2625, b0 0.475 | 156–158 |
| Strong physiology | both × 1.25: k_up 0.4375, b0 0.125 | 159–161 |

The ±25% was set a priori. The only screen allowed (rule I9) was the physiology-only oracle check,
which both physiology variants pass: anticipation still pays (cue oracle − reflex: weak +1.60,
default +2.73, strong +3.73).

**The criteria** (on variant means; N = 3 is too small for tests, so no p-values):
- **R1** conditioning to a novel bell: acquired discrimination > 0.3 and |naive discrimination| < 0.1;
- **R2** contingency, not mere exposure: paired − truly random CR > 0.1;
- **R3** omission insensitivity: OSI < 0.2.

A variant "replicates" if all three hold. Every seed counts, including those that fail the gate.

**The result** (`scripts/analyze_robust.py`, `results/robust/analysis.md`):

| Measure | Ancestors (N=20) | GRU 128 | GRU 32 | Weak | Strong |
|---|---|---|---|---|---|
| Gap closed | 0.57 | 0.49 | 0.52 | 0.37 | 0.54 |
| Seeds passing the gate | 17/20 | 1/3 | 2/3 | 0/3 | 2/3 |
| Naive discrimination | 0.00 | 0.03 | −0.02 | 0.00 | −0.01 |
| Acquired discrimination (R1) | 0.60 | 0.48 | 0.53 | 0.39 | **0.24** |
| Paired − truly random (R2) | 0.36 | 0.31 | 0.37 | 0.23 | 0.14 |
| Extinction drop | 0.45 | 0.29 | 0.48 | 0.38 | 0.16 |
| OSI (R3) | 0.01 | 0.00 | 0.00 | 0.02 | 0.01 |
| Food delivered under omission | 0.16 | 0.11 | 0.12 | 0.75 | 0.03 |
| Replicates? | — | yes | yes | yes | **no (R1)** |

**How to read it.**
- *Network size does not matter* for the core result: half and double size both replicate.
- *Weak physiology replicates, with a caveat.* None of its three seeds passed the training gate (gap
  closed 0.24–0.44), so these agents are less well trained. Their response is about half the size of
  the ancestors' (CR under omission 0.21 vs 0.45), which is why omission still let 75% of the food
  through: a smaller response cancels less food. They still ignore omission (OSI 0.02).
- *Strong physiology does not replicate R1.* Two lineages conditioned normally (acquired
  discrimination 0.32 and 0.44, both passed the gate). The third (seed 161) evolved no response to a
  novel bell at all (−0.04), and that pulled the mean to 0.24. It is not because anticipation paid
  less: with stronger saliva, the payoff of anticipating is *larger*. With three seeds there is
  nothing more to say safely; one guess (untested) is that seed 161 got stuck in a different local
  optimum. Report it as a boundary condition: conditioning to a novel cue usually, not always, evolves.
- *What held in all 12 lineages:* no response to the bell before pairing, and omission insensitivity
  (every OSI ≤ 0.05). This is the most robust result in the project.

**Why it matters for the paper.** A reviewer can no longer say the result depends on one network
size or one hand-picked physiology. It also narrows the search-limit worry a little: a network twice
as large still evolved an omission-insensitive response. It does not close it (same optimiser, same
compute).

---

# Part E — Critical review (my honest assessment)

**What is strong.**
1. Clean design: two loops, invariants enforced by tests, blind lab, yoked controls, naive baselines.
2. Pre-registration with a frozen analysis, intention-to-treat, no exclusions, exact resumption after
   restarts, every decision logged with whether results had been seen.
3. Large, consistent effects across 20 independent evolutionary histories, and across network sizes
   and physiologies (D9; omission insensitivity held in all 12 robustness lineages).
4. The H2 failure is turned into a sharper, interesting claim (H2′), and the disclosure is complete.
5. Mechanistic evidence uses causal interventions (patching), not only decoding.

**What a sceptical reviewer will say, and what I think.**
1. *"The CR is expected — you optimised anticipation."* Partly true. Evolution was rewarded for
   anticipating food in a world where cue meanings vary; within-lifetime conditioning is the natural
   solution. The non-obvious parts are: (a) the agent generalises to a lab protocol it never saw (but
   the lab bell is, to the agent, much like another lifetime's cue — the lab is mostly *in*
   distribution for acquisition); (b) the omission insensitivity; (c) what happens under R-O
   variability (H2′). The paper should not oversell Q1 as surprising; the contribution is the full
   package and H2′.
2. *"Omission insensitivity is trivially optimal in your ecology."* Yes — in the ancestors' world
   salivating never affects food delivery, so ignoring consequences costs nothing. That is the point
   of the functional argument, but it makes H3b close to a prediction-by-construction. The sister arms
   are what make it informative: even when R-O variability is present, the CR did not become
   instrumental.
3. *"The instrumental solution exists but evolution did not find it — maybe a search limit."* The 4×
   search check narrows this but cannot close it; so does the robustness check, where a network
   twice as large still stayed Pavlovian (D9). A different optimiser or longer evolution might find
   it. State as a limitation (we do).
4. *"H2′ is post hoc."* It is pilot-derived, then tested on fresh seeds. That is legitimate but must
   be labelled (it is).
5. *"The digestion sisters failed the training gate (0/20)."* True: they closed only 8% of the
   aware gap, lower than in pilots (25–33%). Their weak CR (0.10) makes their OSI noisier (two seeds
   ≈ 0.5). Their M_ro shows R-O learning did evolve, so the manipulation worked, but the arm is less
   well trained. Intention-to-treat analysis was pre-registered, so it stays in.
6. *"One-shot acquisition is not like animals."* Dogs need several pairings; our agents learn in one.
   The agents' world had ~30 pairings per lifetime and strong contingencies, so fast learning paid.
   Report as a difference.
7. *"No spontaneous recovery, no blocking, no LI, no PREE."* These are real boundary conditions. They
   are also informative (non-competitive learner). The compound-cue follow-up removed the easiest
   excuse for blocking: agents evolved with compounds adapted to them and still did not block (D8).
   A reviewer may say our compound world was the wrong one (random second cue, cheap false alarms);
   that is fair, and the stronger version is listed below.
8. *"The lab differs from the ecology in places."* Yes (`docs/OOD_TABLE.md`): only 2 of 4 channels
   used, no unsignalled food, a long rest, compounds, long CS. Reported.
9. *"The CR uses p, not actions."* p is the policy's probability; using it reduces noise. The sampled
   actions tell the same story (US delivery under omission is computed from sampled actions).
10. *"Statistics: p-values are absurdly small."* Because effects are huge relative to between-seed
    spread. We report CIs and d_z; that is the right emphasis.
11. *"Patching the most-changed units of course restores the donor."* Ranking units by donor −
    recipient difference does bias toward success; the cross-channel control (same ranking procedure,
    mismatched donor gives nothing) is what makes the specificity claim valid. The *number* of units
    needed (8–16) is the informative part.
12. *"Baselines are untuned."* Deliberately (I9). RW and TD with other parameters would still block;
    the instrumental learner's failure might change with other parameters — stated as a caveat.
13. *Process issues.* Several date errata, seven restarts (plus one in the robustness check), one
    control redesigned. All logged; none affected results.
14. *Citations and prior work.* Checked on 2026-11-19 (`docs/LITERATURE_CHECK.md`): three
    references corrected (Frost Hollow first author and title, Domjan DOI, Binz year), titles
    added, one detail still † (Hollis page numbers). Five closely related papers that were missing
    are now cited (A7). A reviewer from artificial life will know Izquierdo et al.; the answer is
    the physiology-only objective, no learning rule, and the held-out omission test. The two
    closest papers were checked at abstract level only and should be read in full.
15. *"Does it depend on your settings?"* Mostly not (D9): 3 of 4 variants replicate. With a
    stronger saliva benefit, 1 of 3 lineages evolved no conditioning. Report it; with N = 3 per
    variant we cannot say how often this happens.

**What I would do to strengthen the paper (priority order).**
1. Stronger blocking test: the same redundant cue throughout a lifetime and/or costlier false
   alarms; and find out what the compound sisters adapted (D8). (The first compound test is done.)
2. A dedicated, pre-registered test of the q-pattern (selective associability), more seeds.
3. Robustness: GRU size 32/128 and physiology ±25% are done (D9). Still open: LSTM, PPO
   replication, and more seeds for the strong-physiology variant.
4. More R-O families and a stronger search for the instrumental solution.
5. The human-experimenter lab bench (Phase 12) as a demonstration.
6. Read the five closest papers in full (`docs/LITERATURE_CHECK.md` §2–3); post the
   pre-registration publicly (e.g. OSF).

---

# Part F — How to rewrite the paper from this document

| Paper section | Source in this document |
|---|---|
| Title / abstract | Part B1, D3 headline numbers, D4–D7 one sentence each |
| Introduction | A1–A7 (background), B1 (question), B2 (spontaneity), Part C's H2 story briefly |
| Methods: physiology | B3 formulas and parameters |
| Methods: ecology | B4 list |
| Methods: agent / outer loop | B5, A6, B6 |
| Methods: lab and measures | B7, B8 |
| Methods: pre-registration & stats | D1, Part C (deviations) |
| Results 1–3 (confirmatory) | D3 table and figure explanations |
| Results 4–8 (exploratory) | D4, D5, D6, D7, D8, D9 |
| Discussion | D3 message lines, D6 interpretation, Part E points 1–8 |
| Limitations | Part E |

**Rules for writing it.** Say "CR-like", "Pavlovian-like"; never claim experience or equivalence with
animals. Label exploratory results. Report magnitudes and CIs first, p-values second. Report failures
(spontaneous recovery, blocking, P1) as boundary conditions.

---

# Part G — Glossary, file map, numbers, reproduction

## Glossary

- **Ancestor / sister lineage**: an evolved agent (4000 generations); copies evolved further under
  different ecologies (2000 generations).
- **Arm**: one experimental condition (ancestor, stable, digestion-variable, partial-omission).
- **Contingency**: P(US | CS); in the lab also a rule linking behaviour to food (omission/positive).
- **CR / UR / CS / US**: conditioned/unconditioned response/stimulus (A1).
- **d_z**: effect size for paired data (D1).
- **Gap fraction**: share of the reflex→oracle gap closed (B6).
- **Generation**: one ES update.
- **Lab subject**: one run of the frozen agent from a zero state.
- **Meta-RL**: learning, by an outer loop, a network that learns within an episode (A5).
- **M_ro**: within-lifetime response–outcome learning score (B8).
- **OOD**: out of distribution (not seen during evolution).
- **Oracle**: hand-written reference policy (B3).
- **OSI**: omission sensitivity index (B8).
- **q / v**: S-O / R-O variability across lifetimes (A7, B4).
- **Seed**: integer that fixes all randomness of a run; one seed = one independent evolutionary
  history.
- **Tick**: one time step of the simulation.
- **Yoked partner**: a control receiving identical cues and food without the contingency (A3).

## File map

| Path | What it is |
|---|---|
| `CLAUDE.md` | Project rules and invariants |
| `docs/PREREGISTRATION.md` | Frozen hypotheses and tests |
| `docs/DECISIONS.md` | Dated log of every decision |
| `docs/OOD_TABLE.md` | What in the lab is in/out of distribution |
| `docs/LITERATURE.md` | Prior work and citations to verify |
| `docs/PILOT_RESULTS.md` | Pilot lab write-up |
| `src/pavlov/physiology.py`, `ecology.py`, `agent.py`, `es.py` | The simulation, the agent, the outer loop |
| `src/pavlov/lab/` | Scheduler, protocols, runner, phenomena |
| `src/pavlov/analysis/` | Metrics and statistics |
| `src/pavlov/baselines/models.py` | RW, TD, Q-learner |
| `scripts/run_confirmatory.py`, `analyze_confirmatory.py` | Confirmatory pipeline and analysis |
| `scripts/run_svar.py`, `analyze_svar.py`, `svar_channel_split.py` | q-control |
| `scripts/mechanism.py`, `mechanism_patch.py` | Mechanism v1, v2 |
| `scripts/run_phenomena.py`, `analyze_phenomena.py` | Phenomena battery |
| `scripts/run_baselines.py`, `compare_baselines.py` | Baselines and scorecard |
| `scripts/run_compound.py`, `analyze_compound.py`, `compound_crosseval.py` | Compound-cue follow-up |
| `scripts/run_robust.py`, `analyze_robust.py` | Robustness check |
| `scripts/make_figures.py` | All figures (`--paper` for title-free versions) |
| `results/confirm/` | Confirmatory runs, analysis, mechanism, phenomena |
| `results/svar/`, `results/baselines/`, `results/compound/`, `results/robust/` | Exploratory results |
| `paper/draft.md`, `paper/ieee/main.tex`, `main.pdf` | The paper |

## Numbers cheat sheet

- Physiology: k_up 0.35, k_down 0.15, D 3, E 1, b0 0.3, c 0.01.
- Ecology: 3000 ticks, 4 channels, event rate U(1/40, 1/20), ISI 3–12, contingency U(0.5, 1),
  P(two predictive) 0.5, hazard 1/1500, ρ 0.5, unsignalled food ≤ 0.005, q = 1, v = 0.
- ES: pop 64, 8 lifetimes, σ 0.03, lr 0.02, wd 0.005, GRU 64; ancestors 4000 gens, sisters 2000.
- Training (ancestors): agent 13.33, reflex 11.61, cue oracle 14.63, never 7.14; gap 0.57; gate 17/20.
- Lab: 32 subjects, lab seed 2026, ISI 6, ITI 15–35.
- H1: D_acq 0.60 [0.53, 0.67]; D_naive −0.004; paired − truly random 0.36; paired − US-alone 0.40.
- H3: extinction to 0.20 (≈ 33%); OSI 0.010 [0.001, 0.023]; food under omission 16%; positive index
  −0.003.
- H2′: stable D_acq 0.73; digestion D_acq 0.10, M_ro +0.68 vs ancestor, OSI 0.08 (n = 18);
  partial-omission D_acq 0.000.
- q-control: inborn ch0 0.85 / 0.28 / 0.11 (q = 0 / 0.5 / 1); learned ch2–3 −0.05 / −0.005 / 0.63.
- Mechanism: probe R² 0.59; patching 76% at 8 units, 100%+ at 16; mismatched −26%.
- Phenomena (ancestors): blocking −0.005, LI −0.015, PREE +0.054 (artefact), IoD −0.001, SOC +0.012.
- Baselines blocking: RW −0.28, TD −0.16.
- Compound follow-up (10 vs 10, paired): blocking +0.012 (C1 fails); D_acq 0.67, OSI 0.03 (C2 holds);
  return in compound world +0.39 [0.30, 0.49] over stable sisters, −0.10 in the ancestral world.
- Robustness (3 seeds each): D_acq GRU128 0.48, GRU32 0.53, weak 0.39, strong 0.24 (seed 161 −0.04);
  OSI ≤ 0.05 in all 12; 3 of 4 variants meet R1–R3.

## Reproduction

```bash
source .venv/bin/activate
pytest -q
python scripts/analyze_confirmatory.py
python scripts/analyze_svar.py && python scripts/svar_channel_split.py
python scripts/summarize_mechanism.py
python scripts/analyze_phenomena.py
python scripts/run_baselines.py && python scripts/compare_baselines.py
python scripts/analyze_compound.py && python scripts/compound_crosseval.py
python scripts/analyze_robust.py
python scripts/make_figures.py && python scripts/make_figures.py --paper --out figures/paper
cd paper/ieee && pdflatex main.tex && pdflatex main.tex
```

Re-running training uses `scripts/run_confirmatory.py` (about 40 hours on 4 CPU cores).
