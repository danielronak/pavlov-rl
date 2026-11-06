# In- vs out-of-distribution features of each lab test

Fill in and update during Phase 3. "IN" = the feature occurs in the primary training ecology
(q = 1, v = 0).

| Lab test | Bell channel meaning | ISI | Contingency type | Schedule structure | Verdict |
|---|---|---|---|---|---|
| Acquisition (CS+/CS−) | none (resampled each lifetime) | IN (6 ∈ [3,12]) | S-O only: IN | regular trials: partially OOD | mostly IN — tests within-lifetime S-O learning |
| Extinction | — | IN | contingency → 0: IN (hazard h) | OOD block structure | IN mechanism |
| Rest → recovery | — | — | reversion: IN if ρ > 0 | OOD (long silence) | partially OOD |
| Omission | — | IN | **R→US cancellation: OOD** | — | **OOD by design (I6)** |
| Positive contingency | — | IN | R→US enabling: OOD | — | OOD |
| Blocking / LI / PREE / 2nd-order | — | IN | compound and pre-exposure structure: OOD | OOD | out-of-sample phenomena |
| Human schedules | — | variable | S-O | irregular | robustness |
