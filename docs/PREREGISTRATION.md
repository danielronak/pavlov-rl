# Preregistration (DRAFT — not frozen)

Status: **draft**. Freeze = `git tag prereg-v1` at the end of Phase 6. After that, this file is
changed only by appending dated amendments.

## Confirmatory hypotheses

**H1 — Emergent conditioning.** In the Pavlovian ecology (q = 1, v = 0), agents frozen after
meta-training show discrimination CR(CS+) − CR(CS−) > 0 over acquisition trials 31–40. The naive
CR before pairing is ≈ 0, meaning the 95% CI for the naive discrimination includes 0, or its
upper bound is < 25% of the acquired discrimination.
- Test: one-sample t-test across seeds (N = 20) on acquired discrimination, plus a paired test
  of acquired vs naive. Effect size: Cohen's d with bootstrap CI.

**H2 — Pavlovian signature depends on R-O invariance.** OSI (omission vs yoked) is lower in the
Pavlovian ecology (v = 0) than in the instrumental ecology (v = 0.5).
- Test: Welch t-test on per-seed OSI, N = 20 per arm.

**H3 — Dissociation.** In the Pavlovian ecology, S-O learning is intact while R-O sensitivity is
low:
- (a) extinction lowers the CR to < 50% of its acquired value within 30 trials;
- (b) OSI < 0.5;
- (c) the agent follows the positive contingency less than the instrumental-ecology agents do.

Holm-Bonferroni correction across the family {H1, H2, H3a, H3b, H3c}.

## Exploratory (labelled as such in the paper)
- Phase-diagram structure (Q2), with a predicted ordering of regions.
- Phenomena battery: directional criteria per phenomenon, fixed before running:
  - spontaneous recovery: recovery ratio > extinction-end ratio;
  - blocking: CR to the added cue (after pretraining A) < CR in the overshadowing control;
  - latent inhibition: acquisition after pre-exposure slower than without;
  - partial-reinforcement extinction effect: slower extinction after 50% reinforcement than
    after 100%;
  - inhibition of delay: positive slope of p(saliva) over a long CS;
  - second-order conditioning: CR to CS2 paired only with CS1 > CR in an unpaired CS2 control.
- Mechanism analyses.

## Exclusion rules
A seed is excluded only if meta-training diverges (NaN) or final training energy is below the
*reflex* oracle. Every exclusion is reported, and an excluded seed is replaced by the next unused
confirmatory seed (20, 21, …).

## Fixed analysis parameters
Per-seed lab subjects: 32 test lifetimes (different bell/metronome channel assignments and RNG).
Bootstrap: 10,000 resamples. α = 0.05 (two-tailed unless stated).
