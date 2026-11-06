# Literature and novelty positioning (living document)

Items marked [VERIFY] were cited from memory or from search snippets. Check them against the
source before citing in the paper.

## What already exists

**TD / RW models of classical conditioning.** The prediction side is solved and well known.
CS value acquisition and the transfer of the TD error to the CS happen by construction.
- Rescorla & Wagner 1972 [VERIFY]
- Sutton & Barto 1990 [VERIFY]
- Ludvig, Sutton & Kehoe 2008/2012 (microstimulus representation; evaluating the TD model) [VERIFY]
- Schultz, Dayan & Montague 1997

**Hand-wired Pavlovian control in agents.** A fixed reflex is triggered by a learned prediction.
This is our baseline B2, the "architectural" route.
- Modayil & Sutton 2014, "Prediction driven behavior: learning predictions that drive fixed responses" [VERIFY]
- Butcher, Pilarski et al. 2022, "Frost Hollow: Pavlovian signalling" — https://arxiv.org/pdf/2203.09498
- Dalrymple et al. 2020 [VERIFY]

**Neural models of stimulus substitution.**
- "Stimulus-to-stimulus learning in RNNs with cortical inductive biases" (2024). Uses designed
  synaptic plasticity; reproduces delay/trace conditioning, extinction, blocking and overshadowing.
  https://arxiv.org/abs/2409.13471
- "Attention as Conditioning" (linear transformers): predicts blocking, excludes spontaneous
  recovery. https://arxiv.org/pdf/2508.08289

**Meta-RL as a model of cognition.**
- Wang et al. 2018, "Prefrontal cortex as a meta-RL system" [VERIFY]
- Binz et al. 2023, "Meta-learned models of cognition" (BBS) [VERIFY]
- Schubert, Jagadish, Binz & Schulz 2024: meta-RL agents fit with RW-type models.
  https://arxiv.org/abs/2402.03969
- None of these test classical conditioning phenomena or the omission test.

**Evolution of learning vs innate behavior.**
- Stephens 1991, "Change, regularity, and value in the evolution of animal learning" (Behavioral
  Ecology): learning is favored when the environment varies between generations but is
  predictable within one. [VERIFY]
- Katsnelson et al. 2012 (Proc R Soc B): learned strategy choice evolves under high fluctuation;
  innate choice evolves in stable environments.
- Giannakakis, Khajehabdollahi & Levina 2023: environmental variability determines evolved
  plasticity rules. https://arxiv.org/abs/2303.06734
- Niv, Joel, Meilijson & Ruppin 2002: evolved RL in simulated bees [VERIFY]
- Todd & Miller 1991: evolution of associative learning [VERIFY]
- An evolved neural-network fear-conditioning model (PMC5673219):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5673219

**Normative theory of Pavlovian vs instrumental control.**
- Dayan, Niv, Seymour & Daw 2006, "The misbehavior of value and the discipline of the will" [VERIFY]
- Guitart-Masip et al. 2012 (go/no-go Pavlovian bias) [VERIFY]
- Lesaint et al. 2014 (sign-/goal-tracking model) [VERIFY]
- "A Unified Theory of Dual-Process Control" (omission sensitivity after overtraining):
  https://arxiv.org/pdf/2211.07036
- Ghirlanda 2020, a unified associative model including omission training and autoshaping
  (Psychonomic Bulletin & Review). [VERIFY]

**Functional view of the CR.**
- Domjan 2005, "Pavlovian conditioning: a functional perspective" (Annual Review of Psychology):
  conditioning evolved because CRs improve interaction with the US. This is our premise; check
  the paper's exact wording on digestion. https://doi.org/10.1146/annurev.psych.55.090902.141409
- Hollis 1997 [VERIFY]; Zener 1937 [VERIFY]

**Animal benchmarks.**
- Pavlov 1927, *Conditioned Reflexes* (acquisition, extinction, spontaneous recovery, inhibition
  of delay, differentiation)
- Sheffield 1965 (salivation under omission) [VERIFY exact numbers]
- Williams & Williams 1969 (negative automaintenance) [VERIFY]
- Rescorla 1967/1968 (truly random control; contingency) [VERIFY]
- Kamin 1969 (blocking); Lubow & Moore 1959 (latent inhibition) [VERIFY]
- Bouton 2004 (extinction as new learning; renewal) [VERIFY]

**Interactive / human-delivered reinforcement** (context for the lab bench).
- TAMER: Knox & Stone 2009 [VERIFY]
- Deep TAMER: Warnell et al. 2018 [VERIFY]
- COACH: MacGlashan et al. 2017 [VERIFY]
- Thomaz & Breazeal 2008 [VERIFY]
- Robotic clicker training: Kaplan et al. 2002 [VERIFY]
- Blumberg et al. 2002, "Dobie" the synthetic dog [VERIFY]
- All of these train *tasks* (instrumental). We found none that use a human as a live Pavlovian
  experimenter with blind, yoked replay.

## Novelty claim (as of 2026-11-06 search; re-check before Phase 6)

1. The CR and the UR **both** emerge from a physiology-only objective, with no wiring from
   prediction to response. This contrasts with Modayil & Sutton and Frost Hollow, where the
   reflex is hand-wired.
2. **Pavlovian vs instrumental as an evolved outcome of ecological statistics,** shown as a phase
   diagram over S-O vs R-O variability and diagnosed with the omission + yoked test. We found no
   meta-RL or evolutionary paper that runs the omission test.
3. Phenomena battery for a meta-learned agent against RW/TD and animal data, including phenomena
   RW fails on.
4. A methods contribution: a blind, yoked, human-in-the-loop Pavlovian lab bench.
