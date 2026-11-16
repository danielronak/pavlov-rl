# Preregistration — prereg-v1 (FROZEN)

Status: **frozen 2026-11-17** at git tag `prereg-v1`. From now on this file changes only by
appending dated amendments at the end, each also logged in `docs/DECISIONS.md`. Confirmatory seeds
(0–99) are refused by `src/pavlov/seeds.py` until the tag exists.

Everything below was fixed after the pilot programme (seeds 100–199, `docs/DECISIONS.md`
2026-11-05 … 2026-11-17) and **before any confirmatory seed was run**. Pilot seeds are never
pooled with confirmatory seeds.

## 0. What changed from the draft, and why (disclosed in the paper)

- **H2 is replaced by H2′.** The draft H2 predicted that R-O variability during evolution makes
  the CR omission-sensitive (higher OSI in v = 0.5 than in v = 0). It was not supported in any
  pilot lineage or family:
  - digestion-domain variability evolved genuine within-lifetime R-O learning, but OSI stayed 0;
  - response→US variability (partial omission) made evolution abandon the CR altogether;
  - a 4× larger search did not change this.

  H2′ states the pattern the pilots showed: *ecology decides whether a CR exists, not whether it
  is Pavlovian. Where a CR exists, it ignores omission.* H2′ is derived from pilot data. It is
  tested here on new, independent seeds and is labelled pilot-derived in the paper.
- Trial windows are aligned with the implemented lab protocols (25 acquisition, 20 extinction,
  40 omission/positive trials). The draft referred to windows that do not exist.
- Equivalence/bound tests replace "≈ 0" wording, and every hypothesis has an exact metric.
- The draft H3c compared against an instrumental-ecology arm that no longer exists. It is now an
  equivalence test against the yoked control.

## 1. Design

**Training (ES outer loop, primary).** Common-ancestor experimental evolution, one ancestor per
confirmatory seed. All settings are as in the pilots. ES uses pop 64, 8 lifetimes per member,
σ 0.03, lr 0.02, weight decay 0.005, with a GRU of 64 units.

| run | seed | config | generations | start |
|---|---|---|---|---|
| ancestor `anc_<a>` | a ∈ 0–19 | `configs/confirm/ancestor.yaml` (A3 ecology, v = 0) | 4000 | fresh init |
| stable sister | 25 + a | `configs/confirm/stable.yaml` (v = 0) | 2000 | ancestor gen 4000, fresh Adam |
| digestion-variable sister | 50 + a | `configs/confirm/digestion.yaml` (A4 flip, v = 0.5) | 2000 | same |
| partial-omission sister | 75 + a | `configs/confirm/partial_omission.yaml` (gate 0.5, v = 0.5) | 2000 | same |

The whole pipeline is run by `scripts/run_confirmatory.py`: training, then for every run the
Phase-2 gate, the R-O learning check and all lab protocols. Results go to
`results/confirm/<run>/`. The analysis is `scripts/analyze_confirmatory.py`. Both scripts are
committed at the tag, and the analysis runs unchanged.

**Lab (frozen weights, I5).** All six protocols (`src/pavlov/lab/protocols.py`) run on the final
checkpoint: ancestor gen 4000, sisters gen 2000. Each run gets 32 subjects, lab seed 2026 and
`PRNGKey(1000 + protocol index)`, so every agent meets the identical set of lab schedules.

**Unit of analysis.** The ancestor seed (N = 20). Within a seed, a lab metric is the mean over
its 32 subjects. Sister arms are paired with their ancestor and with each other.

## 2. Measures (per run; names as in `analyze_confirmatory.py`)

A CR on a trial is the mean p(saliva) over the 3 ticks before the (would-be) US, minus the mean
over the 10 ticks before CS onset (`src/pavlov/analysis/metrics.py`).

- `D_naive`: CS+ CR − CS− CR over the 4 + 4 naive trials before any pairing (pavlov_standard).
- `D_acq`: CS+ CR − CS− CR over the last 10 acquisition trials of each kind (CS+ trials 16–25).
- `CR_acq_cs_plus`: CS+ CR over the last 10 acquisition trials.
- `CR_test_paired`: CS+ CR over the first 5 extinction trials (no US). Matched in trial count and
  US absence to the 5 test trials of the control protocols.
- `CR_test_truly_random`: test CS+ CR after truly-random training (Rescorla 1967).
- `CR_test_us_alone`: test CS+ CR after US-alone exposure (sensitization/pseudoconditioning).
- `CR_ext_last5`: CS+ CR over extinction trials 16–20.
- `OSI` = clip(1 − CR_omission / CR_yoked, 0, 1), using CS+ CRs over the last 15 of 40 training
  trials. It is undefined (NaN) when CR_yoked < 0.05, because then there is no CR to test.
- `positive_index` = CR_positive − CR_yoked over the same window.
- `M_ro` (R-O learning check, `scripts/ro_learning_check.py`): p(saliva | food in mouth) in normal
  lifetimes minus the same in flipped lifetimes. It is measured on ticks 800–2999 of 256 held-out
  lifetimes (EVAL_KEY) of the digestion-variable ecology, whatever ecology the run was trained in.
- `gap_fraction`, `gate_pass`: the Phase-2 gate. It is aware and closed loop for v > 0 arms.

## 3. Confirmatory hypotheses (one Holm-Bonferroni family of 11 tests, α = 0.05)

Two-tailed tests count as support only if the effect is in the predicted direction.

**H1 — Emergent conditioning (ancestors).** Supported iff H1a–H1d are all supported.
- H1a: `D_acq` > 0. One-sample t-test, two-tailed.
- H1b: `D_acq` − `D_naive` > 0. Paired t-test, two-tailed. Descriptive companion: the 95% CI
  of `D_naive`.
- H1c: `CR_test_paired` − `CR_test_truly_random` > 0. Paired t-test, two-tailed (contingency,
  not mere CS/US exposure).
- H1d: `CR_test_paired` − `CR_test_us_alone` > 0. Paired t-test, two-tailed (not sensitization).

**H3 — S-O learning intact, R-O control absent (ancestors).** Supported iff H3a–H3c are all
supported.
- H3a (extinction): 0.5·`CR_acq_cs_plus` − `CR_ext_last5` > 0. One-sample t-test, two-tailed.
- H3b (omission insensitivity): mean `OSI` < 0.2. One-sided t-test against 0.2, using seeds
  with a defined OSI.
- H3c (no positive-contingency control): `positive_index` within (−0.1, +0.1). TOST.

**H2′ — Ecology decides whether a CR exists, not its control mode (sisters).** Supported iff
H2′a–H2′d are all supported. H2′b is a manipulation check: if it fails, H2′a is uninterpretable
and is reported as such.
- H2′a: mean `OSI` of digestion-variable sisters < 0.2. One-sided t-test, seeds with a defined
  OSI.
- H2′b: `M_ro`(digestion sister) − `M_ro`(its ancestor) > 0. Paired t-test, two-tailed. This
  checks that within-lifetime R-O learning actually evolved.
- H2′c: `D_acq`(partial-omission sister) − `D_acq`(stable sister) < 0. Paired t-test, two-tailed.
- H2′d: mean `D_acq`(partial-omission sister) < 0.1. One-sided t-test.

**Fixed rules.**
- Margins (OSI 0.2, positive ±0.1, no-CR 0.1) were fixed by the pilot readings in DECISIONS
  2026-11-07, before any confirmatory run.
- An OSI test needs ≥ 10 seeds with a defined OSI. Otherwise it is "untestable": not supported,
  but kept in the family (its p counts as NaN).
- If all values are identical (SD = 0), the test is decided by the value: p = 0 on the predicted
  side, else 1. It is reported as a magnitude and flagged.
- Effect sizes are means with 95% percentile bootstrap CIs (10,000 resamples over seeds, seed 0)
  and Cohen's d_z.

## 4. Inclusion, exclusion, stopping

- **Ancestor exclusion (only rule).** An ancestor is excluded if meta-training diverged (NaN) or
  its final training return on the held-out evaluation lifetimes is below the reflex oracle.
  An excluded ancestor's sisters are not run, and it is replaced by the next seed (20, 21, …,
  up to 24). More than 5 exclusions → stop and report to the PI.
- **Passing the Phase-2 gate is NOT an inclusion criterion.** All included seeds are analysed
  (intention to treat), and gate pass rates per arm are reported.
- Sister runs are never excluded. A sister that crashes is resumed exactly from its checkpoint.
  If a step fails 3 times, the driver stops, and the failure is logged and reported.
- Confirmatory runs are never re-run after being looked at, and their results are never edited.

## 5. Exploratory (labelled as such in the paper)

- Spontaneous recovery: `CR_recovery_first` > `CR_ext_last5` (ancestors and stable sisters).
- OSI, US delivered under omission, and conditioning strength in every arm; gate pass rates;
  training curves.
- The digestion sisters' weaker CR (`D_acq` vs stable sister).
- Pilot-only results reported as such: the search-limit check (pop 256), the gen-4000 extension
  of the digestion lineages, and recurrent PPO (the replication arm failed the gate).
- Phenomena battery, still to be implemented, with directional criteria fixed here:
  - blocking: CR to the added cue < overshadowing control;
  - latent inhibition: slower acquisition after pre-exposure;
  - partial-reinforcement extinction effect: slower extinction after 50% than after 100%;
  - inhibition of delay: positive slope of p(saliva) over a long CS;
  - second-order conditioning: CR to CS2 > unpaired CS2 control.
- Mechanism (Q4): linear probes and causal interventions on the GRU state.

## 6. Compute

About 200,000 ES generations in all: 20 × 4000 for ancestors and 60 × 2000 for sisters. On the
4-core CPU this is roughly 1 day of wall time, with 4 parallel jobs and exact resume after
interruptions.

## Amendments

- 2026-11-16 — Erratum, no change of content: the freeze date above should read 2026-11-16
  (system clock), not 2026-11-17. The frozen content is commit b4d07cf.
