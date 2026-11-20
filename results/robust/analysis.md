# Robustness check (exploratory; 3 seeds per variant, mean (min to max); reference = 20 ancestors)

| measure | ancestors (N=20) | h128 | h32 | phys_weak | phys_strong |
|---|---|---|---|---|---|
| gap_fraction | +0.571 (+0.452 to +0.693) | +0.487 (+0.427 to +0.603) | +0.521 (+0.484 to +0.570) | +0.366 (+0.243 to +0.437) | +0.539 (+0.452 to +0.601) |
| gate_pass | +0.850 (+0.000 to +1.000) | +0.333 (+0.000 to +1.000) | +0.667 (+0.000 to +1.000) | +0.000 (+0.000 to +0.000) | +0.667 (+0.000 to +1.000) |
| D_naive | -0.004 (-0.067 to +0.044) | +0.030 (-0.019 to +0.061) | -0.021 (-0.048 to -0.006) | +0.003 (-0.004 to +0.013) | -0.008 (-0.023 to +0.005) |
| D_acq | +0.599 (+0.284 to +0.954) | +0.480 (+0.389 to +0.635) | +0.527 (+0.469 to +0.634) | +0.393 (+0.309 to +0.460) | +0.240 (-0.043 to +0.441) |
| paired_minus_random | +0.358 (+0.179 to +0.625) | +0.313 (+0.190 to +0.517) | +0.366 (+0.326 to +0.413) | +0.228 (+0.162 to +0.289) | +0.141 (-0.030 to +0.258) |
| extinction_drop | +0.447 (+0.006 to +0.684) | +0.294 (+0.231 to +0.382) | +0.479 (+0.404 to +0.560) | +0.382 (+0.276 to +0.460) | +0.159 (-0.005 to +0.349) |
| OSI | +0.010 (+0.000 to +0.105) | +0.000 (+0.000 to +0.000) | +0.003 (+0.000 to +0.008) | +0.018 (+0.000 to +0.053) | +0.012 (+0.000 to +0.037) |
| US_delivered_omission | +0.161 (+0.057 to +0.358) | +0.113 (+0.101 to +0.134) | +0.120 (+0.091 to +0.137) | +0.750 (+0.679 to +0.862) | +0.034 (+0.000 to +0.064) |

- h128: R1 met, R2 met, R3 met -> replicates (OSI defined in 3/3)
- h32: R1 met, R2 met, R3 met -> replicates (OSI defined in 3/3)
- phys_weak: R1 met, R2 met, R3 met -> replicates (OSI defined in 3/3)
- phys_strong: R1 NOT met, R2 met, R3 met -> does NOT replicate (OSI defined in 3/3)
