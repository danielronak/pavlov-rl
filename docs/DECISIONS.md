# Decision log (append-only)

Format: `YYYY-MM-DD — decision — reason — invariant(s) touched — seen lab results? (y/n)`

- 2026-11-06 — Salivation is the agent's only effector; its only consequence is a saliva level
  that scales digestion efficiency — this makes the CR emergent rather than wired (rejects the
  hand-wired prediction→reflex design, which becomes baseline B2) — I1, I2 — n
- 2026-11-06 — Outer loop = recurrent PPO (meta-RL); replication with OpenAI-ES — gradient meta-RL
  is the standard, and ES is literal evolution — none — n
- 2026-11-06 — Primary R-O variability family = flipping saliva's digestion effect, NOT omission-like
  contingencies — keeps the omission test out of distribution — I6 — n
- 2026-11-06 — Stack = JAX on CPU (4 cores, no GPU, no torch preinstalled) — vmap over lifetimes — none — n
- 2026-11-06 — Phase-1 gate criteria finalized as G1–G4 (reflex>never; cue_oracle>reflex; cue_oracle>any_cue; always<reflex; margins ≥10% of gap) — replaces the earlier draft ordering, which required a cue-blind policy to beat reflex; that is not needed for the argument — I9 — n
- 2026-11-06 — Phase-1 gate PASSED with provisional physiology defaults (k_up .35, k_down .15, D 3, b0 .3, c .02), 512 lifetimes, seed 100: cue_oracle − reflex = +5.51 [5.16, 5.86] (≈22% over reflex), best lead k = 2; cue_oracle − any_cue = +24.8. Physiology frozen at these values unless the Phase-2 TRAINING-return gate fails — I9 — n
- 2026-11-06 — Note for Phase 2: the anticipation gain (~0.1 energy per food event) is modest. If meta-training cannot learn anticipation, increasing it (e.g. lower k_up) is allowed ONLY on training-return grounds and must be logged here before any lab run — I9 — n
- 2026-11-06 — GRU hidden 128 → 64 and one XLA thread per process, with seeds run as parallel processes. Throughput went from 25k steps/s (1 run, h=128) to 3 × 55k steps/s (h=64). Compute-motivated; hidden size is re-tested in Phase 11 robustness — none — n
