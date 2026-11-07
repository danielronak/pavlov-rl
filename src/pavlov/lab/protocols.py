"""Lab protocols (parameters fixed in docs/DECISIONS.md, 2026-11-07, before any agent ran).

Every protocol starts with a NAIVE phase (CS+ and CS- without US) so the CR to the bell before any
pairing is always measured (invariant I8). Trial order within a phase is shuffled per subject.
"""
from __future__ import annotations

import numpy as np

from pavlov.lab.scheduler import BLANK, CS_MINUS, CS_PLUS, US_ONLY, Protocol, RandomUS, Trial


def _shuffled(rng, trials):
    trials = list(trials)
    rng.shuffle(trials)
    return trials


def _naive(rng):
    return _shuffled(rng, [Trial("naive", CS_PLUS)] * 4 + [Trial("naive", CS_MINUS)] * 4)


def pavlov_standard(rng) -> Protocol:
    tr = _naive(rng)
    tr += _shuffled(rng, [Trial("acquisition", CS_PLUS, us=True)] * 25 + [Trial("acquisition", CS_MINUS)] * 25)
    tr += [Trial("extinction", CS_PLUS)] * 20
    tr += [Trial("rest", BLANK, blank=300)]
    tr += [Trial("recovery", CS_PLUS)] * 5
    return Protocol("pavlov_standard", tr, phases=["naive", "acquisition", "extinction", "rest", "recovery"])


def truly_random(rng) -> Protocol:
    """Same CS exposure and US count as acquisition, but US timing independent of the CS."""
    tr = _naive(rng)
    tr += _shuffled(rng, [Trial("acquisition", CS_PLUS)] * 25 + [Trial("acquisition", CS_MINUS)] * 25)
    tr += [Trial("test", CS_PLUS)] * 5
    return Protocol("truly_random", tr, random_us=[RandomUS("acquisition", 25)],
                    phases=["naive", "acquisition", "test"])


def cs_alone(rng) -> Protocol:
    tr = _naive(rng)
    tr += _shuffled(rng, [Trial("acquisition", CS_PLUS)] * 25 + [Trial("acquisition", CS_MINUS)] * 25)
    tr += [Trial("test", CS_PLUS)] * 5
    return Protocol("cs_alone", tr, phases=["naive", "acquisition", "test"])


def us_alone(rng) -> Protocol:
    """Sensitization / pseudoconditioning control: US exposure without any CS."""
    tr = _naive(rng)
    tr += _shuffled(rng, [Trial("acquisition", US_ONLY, us=True)] * 25 + [Trial("acquisition", CS_MINUS)] * 25)
    tr += [Trial("test", CS_PLUS)] * 5
    return Protocol("us_alone", tr, phases=["naive", "acquisition", "test"])


def omission(rng) -> Protocol:
    """Sheffield (1965) / Williams & Williams (1969): salivating during the CS+ cancels its US."""
    tr = _naive(rng)
    tr += _shuffled(rng, [Trial("training", CS_PLUS, us=True, contingent=True)] * 40
                    + [Trial("training", CS_MINUS)] * 20)
    return Protocol("omission", tr, contingency="omission", phases=["naive", "training"])


def positive(rng) -> Protocol:
    """Dissociation control: the CS+ US is delivered only if the agent salivates during the CS."""
    tr = _naive(rng)
    tr += _shuffled(rng, [Trial("training", CS_PLUS, us=True, contingent=True)] * 40
                    + [Trial("training", CS_MINUS)] * 20)
    return Protocol("positive", tr, contingency="positive", phases=["naive", "training"])


PROTOCOLS = {
    "pavlov_standard": pavlov_standard,
    "truly_random": truly_random,
    "cs_alone": cs_alone,
    "us_alone": us_alone,
    "omission": omission,
    "positive": positive,
}

# protocols whose runs get an exact yoked partner (same cues, same delivered food, no contingency)
YOKED = {"omission", "positive"}


def make_protocol(name: str, seed: int) -> Protocol:
    return PROTOCOLS[name](np.random.default_rng(seed))
