# Phase 4 — Pilot lab results (GO / NO-GO)

Agents: ES pilot seeds 109 and 110 (A3 ecology, q = 1, v = 0), checkpoint gen 4000, frozen.
Lab: 32 subjects per seed, lab seed 2026, protocols and metrics as fixed in `DECISIONS.md`
on 2026-11-07. Values are mean ± 95% CI across subjects within a seed. N = 2 seeds, so this is
a pilot: no seed-level statistics.

**Verdict: GO.** A CR-like response to a novel bell emerges, discriminates from the metronome,
extinguishes, and does **not** disappear under omission (OSI = 0 in both seeds).

## Pavlov standard protocol

| Measure | Seed 109 | Seed 110 |
|---|---|---|
| Naive CR, bell (before any pairing) | +0.170 ± 0.052 | +0.201 ± 0.067 |
| Naive CR, metronome | +0.166 ± 0.054 | +0.184 ± 0.058 |
| Naive discrimination | +0.004 ± 0.086 | +0.018 ± 0.101 |
| Acquisition CR, bell (last 10) | +0.827 ± 0.050 | +0.724 ± 0.090 |
| Acquisition CR, metronome (last 10) | +0.017 ± 0.013 | +0.008 ± 0.006 |
| **Acquired discrimination** | **+0.810 ± 0.053** | **+0.716 ± 0.090** |
| Extinction CR (last 5 of 20) | +0.192 ± 0.081 | +0.160 ± 0.049 |
| Recovery CR (first trial after 300-tick rest) | +0.206 ± 0.079 | +0.042 ± 0.037 |

Curves (subject means):
- **Acquisition is essentially one-trial.** Bell CR goes 0.03 → 0.79 (109) and 0.02 → 0.67 (110)
  after a single pairing, then stays flat. The metronome CR decays from about 0.08 to about 0.01.
- **Extinction is gradual:** 0.82 → 0.15 (109) and 0.72 → 0.10 (110) over 20 trials.
- **Spontaneous recovery is absent or weak:** 109 recovers slightly (0.15 at the last extinction
  trial → 0.21); 110 does not (0.10 → 0.04).
- Every one of the 64 subjects discriminates (minimum +0.33), and every bell channel shows it.
  Channel 1 is weaker in both seeds (0.64 and 0.40).

## Controls

| Protocol | Test CR to bell, 109 | Test CR to bell, 110 |
|---|---|---|
| Paired (acquisition, last 10) | +0.827 | +0.724 |
| Truly random (US independent of CS) | +0.115 ± 0.067 | +0.091 ± 0.070 |
| CS alone | +0.033 ± 0.032 | +0.006 ± 0.008 |
| US alone (sensitization) | +0.006 ± 0.006 | +0.016 ± 0.018 |

The response depends on the CS–US contingency, not on exposure to the CS or the US.

## Omission (Sheffield 1965 / Williams & Williams 1969) and positive contingency

| Measure | Seed 109 | Seed 110 |
|---|---|---|
| CR under omission (last 15 CS+) | +0.535 ± 0.065 | +0.528 ± 0.048 |
| CR of yoked partner | +0.512 ± 0.071 | +0.427 ± 0.057 |
| **OSI** | **0.00** | **0.00** |
| US actually delivered under omission | 12.5% | 11.8% |
| CR under positive contingency | +0.335 ± 0.130 | +0.183 ± 0.106 |
| CR of yoked partner | +0.350 ± 0.134 | +0.198 ± 0.112 |

Under omission the agents keep salivating and lose about 88% of the food. Their CR equals that
of yoked partners who receive the same food with no contingency. They are equally insensitive
to a positive contingency. S-O learning is intact (fast acquisition, discrimination, extinction);
R-O sensitivity is absent. That is the Pavlovian dissociation predicted for the v = 0 ecology.

## What this does and does not show yet

- **Not yet shown: H2 (ecology dependence).** OSI = 0 is predicted for v = 0, but alone it can't
  separate "Pavlovian because of ecology" from "this architecture can never learn R-O relations".
  We need the instrumental ecology (v > 0) agents to show OSI > 0.
- **Deviations from animal data, to be reported:** acquisition is one-trial rather than
  negatively accelerated; spontaneous recovery is absent or weak; there is a non-specific naive
  response (≈ 0.17–0.20) to any novel stimulus, equal for bell and metronome.
- In the positive-contingency protocol the agent rarely salivates early, so food seldom arrives;
  conditioning there bootstraps slowly from chance responses (0.06 → 0.34 over 40 trials).
