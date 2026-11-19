# In- vs out-of-distribution features of each lab test

Updated 2026-11-07 for the A3 primary ecology. "IN" means the feature occurs in the training
ecology (q = 1, v = 0, 4 channels, delay cues, one stimulus at a time, ISI ∈ [3, 12],
contingency ∈ [0.5, 1], event rate U(1/40, 1/20), lifetime 3000 ticks).

| Lab test | Feature | Status | Note |
|---|---|---|---|
| All | Bell/metronome channel meaning | IN (no fixed meaning) | channels drawn per subject |
| All | Delay CS, ISI 6, CS on through US tick | IN | |
| All | ITI U{15..35} | IN (approx.) | ecology idle wait ≈ geometric, mean 20–40 |
| All | Only 2 of 4 channels used | partially OOD | training uses all 4 channels |
| All | Session length ≤ 3100 ticks | ≈ IN | ≤ 3% beyond the training lifetime |
| Acquisition | CS+ contingency 1.0, CS− never paired | IN | edge of the training range |
| Acquisition | No unsignalled food | partially OOD | training has unsignalled food U(0, 0.005)/tick |
| Extinction | Contingency drops to 0 | IN mechanism | training hazard toggles cues silent (h = 1/1500) |
| Extinction | Block of 20 unpaired CS+ in a row | partially OOD | |
| Rest → recovery | 300 empty ticks | OOD | longest training gap is far shorter |
| Truly random | US independent of CS | IN | like unsignalled food plus silent cues |
| US alone | US without any CS | IN | unsignalled food |
| **Omission** | **Salivation cancels the US** | **OOD by design (I6)** | never in training |
| Positive contingency | Salivation required for the US | OOD | never in training |
| Yoked | Food replay of a partner | IN at the agent's level | the agent sees an ordinary food schedule |
| Trace / compound / pre-exposure (Phase 9) | trace gaps, overlapping cues, long CS-alone runs | OOD | out-of-sample phenomena |

## Sister lineages (confirmatory H2′, PREREGISTRATION.md)

| Arm | Feature | Status | Note |
|---|---|---|---|
| Ancestor, stable sister | Omission / positive contingency | OOD (I6) | as above |
| Digestion-variable sister | Salivation hurts digestion in half the lifetimes | IN for that arm | R-O relation about digestion, never about US delivery |
| Digestion-variable sister | Omission (salivation cancels the US) | OOD (I6) | |
| Partial-omission sister | Salivation in the CS window halves US probability in half the lifetimes | IN for that arm | declared `ro_gate.py` family (I2) |
| Partial-omission sister | Full omission in the lab | NEAR-distribution | stronger and deterministic version of the training gate (DECISIONS 2026-11-07) |

## Phenomena battery (Phase 9, DECISIONS 2026-11-19)

| Test | Feature | Status | Note |
|---|---|---|---|
| Blocking | Two cues on together (AX compound) | OOD | training has one stimulus at a time (A3) |
| Latent inhibition | 20 unreinforced presentations of one cue in a row | partially OOD | training contingency ≥ 0.5 |
| PREE | 50% reinforcement during acquisition | IN (edge) | training contingency ∈ [0.5, 1] |
| Inhibition of delay | 24-tick CS | OOD | training ISI ≤ 12; the 12-tick version is IN (edge) |
| Second-order | B then A, no US (serial) | IN | one stimulus at a time; unreinforced A is partially OOD |

## Compound-cue follow-up (DECISIONS 2026-11-19)

For the compound sisters (`configs/exp/compound.yaml`, p_compound = 0.3), two cues starting together is
**IN** distribution. The lab's fixed redundant cue X (the same X on all 10 AX+ trials, after 20 A+
trials) is **partially OOD**: in that ecology the second cue of a compound is a fresh random channel
at each event. All other lab tests have the same status as for the ancestors.
