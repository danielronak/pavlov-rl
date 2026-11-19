"""Phenomena battery (Phase 9; designs fixed in docs/DECISIONS.md 2026-11-19 before any agent ran).

BLIND BY CONSTRUCTION (I7): a schedule is built from a design, an RNG and the number of channels
only. No contingency: food = planned US. Cue roles (A, B, X) are mapped to distinct channels at
random per subject, so no role has a fixed channel (I3).

A trial is a set of cue segments (role, start offset, duration), an optional US offset, and the
offset at which its CR window ends (CR = mean p over the 3 ticks before that offset minus the mean
over the 10 ticks before trial onset, as in the main lab).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRE, BASE = 3, 10
ITI = (15, 35)


@dataclass(frozen=True)
class PTrial:
    phase: str
    segs: tuple                 # ((role, start, duration), ...)
    us_at: int | None           # US offset from trial onset, or None
    measure_end: int            # CR window = [onset + measure_end - PRE, onset + measure_end)
    length: int
    tag: str


def cs(phase, roles, us, tag, isi=6):
    """Delay CS: every role on from onset through the (would-be) US tick."""
    return PTrial(phase, tuple((r, 0, isi + 1) for r in roles), isi if us else None, isi, isi + 1, tag)


def serial(phase, first, second, tag, isi=6):
    """first for isi ticks, then second for isi + 1 ticks, no US (second-order conditioning)."""
    return PTrial(phase, ((first, 0, isi), (second, isi, isi + 1)), None, isi, 2 * isi + 1, tag)


def _sh(rng, trials):
    trials = list(trials)
    rng.shuffle(trials)
    return trials


# ---- designs: rng -> {group: [PTrial, ...]}
def blocking(rng):
    def group(pre):
        return ([cs("pretrain", (pre,), True, f"pre_{pre}")] * 20 + [cs("compound", ("A", "X"), True, "AX")] * 10
                + [cs("test", ("X",), False, "test_X")] * 5)
    return {"blocking": group("A"), "overshadowing": group("B")}


def latent_inhibition(rng):
    def group(pre):
        return [cs("preexposure", (pre,), False, f"pre_{pre}")] * 20 + [cs("acquisition", ("A",), True, "acq_A")] * 10
    return {"preexposed": group("A"), "control": group("B")}


def pree(rng):
    full = [cs("acquisition", ("A",), True, "acq_A")] * 20
    part = _sh(rng, [cs("acquisition", ("A",), True, "acq_A")] * 20 + [cs("acquisition", ("A",), False, "acq_A")] * 20)
    ext = [cs("extinction", ("A",), False, "ext_A")] * 20
    return {"p100": full + ext, "p50": part + ext}


def inhibition_of_delay(rng):
    return {"isi24": [cs("acquisition", ("A",), True, "acq_A", isi=24)] * 25,
            "isi12": [cs("acquisition", ("A",), True, "acq_A", isi=12)] * 25}


def second_order(rng):
    first = [cs("first_order", ("A",), True, "acq_A")] * 20
    paired = _sh(rng, [serial("second_order", "B", "A", "BA")] * 10 + [cs("second_order", ("A",), True, "acq_A")] * 10)
    unpaired = _sh(rng, [cs("second_order", ("B",), False, "B_alone")] * 10
                   + [cs("second_order", ("A",), False, "A_alone")] * 10
                   + [cs("second_order", ("A",), True, "acq_A")] * 10)
    test = [cs("test", ("B",), False, "test_B")] * 5
    return {"paired": first + paired + test, "unpaired": first + unpaired + test}


DESIGNS = {"blocking": blocking, "latent_inhibition": latent_inhibition, "pree": pree,
           "inhibition_of_delay": inhibition_of_delay, "second_order": second_order}


@dataclass
class PSchedule:
    cues: np.ndarray
    us_planned: np.ndarray
    us_trial: np.ndarray
    window_trial: np.ndarray
    trial_onset: np.ndarray
    trials: list
    roles: dict
    length: int


def build(trials: list[PTrial], rng: np.random.Generator, n_channels: int) -> PSchedule:
    names = sorted({r for t in trials for r, _, _ in t.segs})
    chans = rng.choice(n_channels, len(names), replace=False)
    roles = dict(zip(names, (int(c) for c in chans)))
    onsets, t = [], 0
    for tr in trials:
        t += int(rng.integers(ITI[0], ITI[1] + 1))
        onsets.append(t)
        t += tr.length
    T = t + ITI[1]
    cues = np.zeros((T, n_channels), bool)
    us = np.zeros(T, bool)
    us_trial = -np.ones(T, int)
    for i, (tr, on) in enumerate(zip(trials, onsets)):
        for r, s, d in tr.segs:
            cues[on + s:on + s + d, roles[r]] = True
        if tr.us_at is not None:
            us[on + tr.us_at] = True
            us_trial[on + tr.us_at] = i
    return PSchedule(cues, us, us_trial, -np.ones(T, int), np.array(onsets), list(trials), roles, T)


def schedules(design: str, n_subjects: int, n_channels: int, lab_seed: int):
    """{group: [PSchedule per subject]}; identical for every agent (fixed lab seed)."""
    di = list(DESIGNS).index(design)
    out = {}
    groups = DESIGNS[design](np.random.default_rng([lab_seed, 99, di]))
    for gi, (g, _) in enumerate(groups.items()):
        out[g] = []
        for s in range(n_subjects):
            rng = np.random.default_rng(np.random.SeedSequence([lab_seed, 99, di, gi, s]))
            out[g].append(build(DESIGNS[design](rng)[g], rng, n_channels))
    return out


def trial_crs(p_row: np.ndarray, sch: PSchedule) -> np.ndarray:
    return np.array([p_row[o + t.measure_end - PRE:o + t.measure_end].mean() - p_row[o - BASE:o].mean()
                     for o, t in zip(sch.trial_onset, sch.trials)])


def tagged(crs, sch, tag, sl=slice(None)):
    idx = [i for i, t in enumerate(sch.trials) if t.tag == tag]
    return crs[idx][sl]


def cs_slope(p_row, sch, tag, last=10):
    """Mean OLS slope of p(saliva) over CS ticks 1..ISI-1 (excluding the US tick), last trials."""
    sl = []
    idx = [i for i, t in enumerate(sch.trials) if t.tag == tag][-last:]
    for i in idx:
        o, t = sch.trial_onset[i], sch.trials[i]
        y = p_row[o + 1:o + t.measure_end]
        x = np.arange(len(y))
        sl.append(np.polyfit(x, y, 1)[0])
    return float(np.mean(sl))
