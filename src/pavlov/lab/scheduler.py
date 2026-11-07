"""Lab scheduler: turns a protocol (list of trials) into per-tick arrays.

BLIND BY CONSTRUCTION (invariant I7): building a schedule takes only a protocol, an RNG and the
number of channels. It never receives the agent, its parameters or its outputs. The only way the
agent's behaviour can affect what happens in the lab is a declared contingency (omission,
positive, replay), applied by the runner and logged in the results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# trial kinds
CS_PLUS, CS_MINUS, US_ONLY, BLANK = 0, 1, 2, 3
KIND_NAMES = {CS_PLUS: "CS+", CS_MINUS: "CS-", US_ONLY: "US", BLANK: "blank"}


@dataclass(frozen=True)
class Trial:
    phase: str
    kind: int                 # CS_PLUS | CS_MINUS | US_ONLY | BLANK
    us: bool = False          # a US is planned at CS onset + isi (or at onset for US_ONLY)
    contingent: bool = False  # the declared contingency (if any) applies to this trial
    blank: int = 0            # BLANK: number of empty ticks


@dataclass(frozen=True)
class RandomUS:
    """Truly-random control: n USs placed uniformly at random ticks within `phase`, independent
    of CS timing (Rescorla 1967)."""
    phase: str
    n: int


@dataclass
class Protocol:
    name: str
    trials: list[Trial]
    contingency: str = "none"            # none | omission | positive
    random_us: list[RandomUS] = field(default_factory=list)
    phases: list[str] = field(default_factory=list)


@dataclass
class LabSchedule:
    cues: np.ndarray          # [T, Nc] bool
    us_planned: np.ndarray    # [T] bool
    us_trial: np.ndarray      # [T] int: trial index whose US is planned at this tick, else -1
    window_trial: np.ndarray  # [T] int: contingent CS+ trial whose response window covers t, else -1
    trial_onset: np.ndarray   # [n_trials] int
    trial_kind: np.ndarray    # [n_trials] int
    trial_phase: np.ndarray   # [n_trials] int (index into phases)
    trial_us: np.ndarray      # [n_trials] bool
    phases: list[str]
    bell: int
    metronome: int
    isi: int
    length: int


def build_schedule(protocol: Protocol, rng: np.random.Generator, n_channels: int, isi: int = 6,
                   iti: tuple[int, int] = (15, 35)) -> LabSchedule:
    bell, metronome = rng.choice(n_channels, 2, replace=False)
    phases = protocol.phases or list(dict.fromkeys(t.phase for t in protocol.trials))
    onsets, kinds, phase_idx, has_us = [], [], [], []
    t = 0
    for tr in protocol.trials:
        if tr.kind == BLANK:
            t += tr.blank
            continue
        t += int(rng.integers(iti[0], iti[1] + 1))
        onsets.append(t)
        kinds.append(tr.kind)
        phase_idx.append(phases.index(tr.phase))
        has_us.append(tr.us)
        t += 1 if tr.kind == US_ONLY else isi + 1
    T = t + iti[1]
    cues = np.zeros((T, n_channels), bool)
    us_planned = np.zeros(T, bool)
    us_trial = -np.ones(T, int)
    window_trial = -np.ones(T, int)
    trials = [tr for tr in protocol.trials if tr.kind != BLANK]
    for i, (tr, on) in enumerate(zip(trials, onsets)):
        if tr.kind in (CS_PLUS, CS_MINUS):
            ch = bell if tr.kind == CS_PLUS else metronome
            cues[on:on + isi + 1, ch] = True        # delay CS: on through the US tick
            if tr.us:
                us_planned[on + isi] = True
                us_trial[on + isi] = i
            if tr.contingent:
                window_trial[on:on + isi] = i         # response window: CS onset .. tick before US
        elif tr.kind == US_ONLY and tr.us:
            us_planned[on] = True
            us_trial[on] = i
    for ru in protocol.random_us:
        pi = phases.index(ru.phase)
        in_phase = [o for o, p in zip(onsets, phase_idx) if p == pi]
        lo, hi = min(in_phase) - iti[0], max(in_phase) + isi + 1
        ticks = rng.choice(np.arange(lo, hi), ru.n, replace=False)
        us_planned[ticks] = True  # not tied to any trial (us_trial stays -1)
    return LabSchedule(cues, us_planned, us_trial, window_trial, np.array(onsets), np.array(kinds),
                       np.array(phase_idx), np.array(has_us), phases, int(bell), int(metronome),
                       isi, T)


def stack_schedules(scheds: list[LabSchedule]) -> dict[str, np.ndarray]:
    """Pad a batch of subject schedules (different lengths) to a common length."""
    T = max(s.length for s in scheds)
    n_tr = max(len(s.trial_onset) for s in scheds)
    B, Nc = len(scheds), scheds[0].cues.shape[1]
    out = {
        "cues": np.zeros((B, T, Nc), bool), "us_planned": np.zeros((B, T), bool),
        "us_trial": -np.ones((B, T), int), "window_trial": -np.ones((B, T), int),
        "length": np.array([s.length for s in scheds]), "n_trials": n_tr,
    }
    for b, s in enumerate(scheds):
        L = s.length
        out["cues"][b, :L] = s.cues
        out["us_planned"][b, :L] = s.us_planned
        out["us_trial"][b, :L] = s.us_trial
        out["window_trial"][b, :L] = s.window_trial
    return out
