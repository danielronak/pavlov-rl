"""Typed configs. Defaults here are PROVISIONAL until fixed by the Phase-1 oracle gate
(see docs/DECISIONS.md). Never tune them on lab metrics (invariant I9)."""
from __future__ import annotations

from dataclasses import dataclass, asdict, fields
from typing import Any

import yaml


@dataclass(frozen=True)
class PhysiologyConfig:
    k_up: float = 0.35      # saliva rise per salivation tick (fraction of remaining headroom)
    k_down: float = 0.15    # passive saliva decay per tick
    food_ticks: int = 3     # D: ticks food stays in the mouth after a US
    energy: float = 1.0     # E: total energy of one food item at perfect digestion
    b0: float = 0.3         # digestion efficiency with no saliva
    cost: float = 0.02      # c: metabolic cost per salivation tick


@dataclass(frozen=True)
class EcologyConfig:
    lifetime: int = 3000            # T ticks
    n_channels: int = 6             # N_c cue channels
    p_two_predictive: float = 0.5   # prob a lifetime has 2 predictive cues (else 1)
    isi_min: int = 3
    isi_max: int = 12
    p_trace: float = 0.5            # prob a predictive cue is a trace cue (offset before US)
    trace_gap_min: int = 1
    trace_gap_max: int = 4
    contingency_min: float = 0.5    # P(US | CS onset)
    contingency_max: float = 1.0
    pred_rate_min: float = 1 / 60   # onset rate of predictive cues (per tick, when idle)
    pred_rate_max: float = 1 / 25
    distractor_rate_min: float = 0.005
    distractor_rate_max: float = 0.03
    distractor_dur_min: int = 3
    distractor_dur_max: int = 12
    unsignalled_food_max: float = 0.005
    # Ecological knobs (the independent variables of the phase diagram)
    q_so_variability: float = 1.0   # prob cue->channel mapping is resampled this lifetime
    v_ro_variability: float = 0.0   # prob saliva's digestion effect is flipped this lifetime
    hazard: float = 1 / 1500        # per-tick prob a predictive cue toggles active/silent
    reversion: float = 0.5          # rho: prob a reactivated cue reverts to its original contingency


@dataclass(frozen=True)
class Config:
    physiology: PhysiologyConfig = PhysiologyConfig()
    ecology: EcologyConfig = EcologyConfig()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build(cls, d: dict | None):
    if not d:
        return cls()
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return Config(
        physiology=_build(PhysiologyConfig, raw.get("physiology")),
        ecology=_build(EcologyConfig, raw.get("ecology")),
    )
