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
    cost: float = 0.01      # c: metabolic cost per salivation tick (0.02 -> 0.01: amendment A1, see DECISIONS.md)
    # R-O family for flipped lifetimes (v > 0 arms only). "matched_baseline" (A4): b0*(1-S), equal to
    # the normal efficiency at S = 0, so only salivating reveals the lifetime type. "unmatched": the
    # flawed pre-A4 family 1 - (1-b0)*S used by pilot runs 112/113.
    flip_family: str = "matched_baseline"


@dataclass(frozen=True)
class EcologyConfig:
    lifetime: int = 3000            # T ticks
    n_channels: int = 4             # N_c cue channels (6 -> 4: amendment A3)
    p_two_predictive: float = 0.5   # prob a lifetime has 2 predictive cues (else 1)
    isi_min: int = 3
    isi_max: int = 12
    p_trace: float = 0.0            # prob a cue is a trace cue (0.5 -> 0: A3; trace = lab test)
    trace_gap_min: int = 1
    trace_gap_max: int = 4
    contingency_min: float = 0.5    # P(US | CS onset)
    contingency_max: float = 1.0
    exclusive_events: bool = True   # A3: one stimulus at a time (world-level events, uniform channel)
    event_rate_min: float = 1 / 40  # per idle tick, used when exclusive_events (A3 rule, DECISIONS.md)
    event_rate_max: float = 1 / 20
    cue_rate_min: float = 1 / 60    # onset rate of EVERY cue channel (per idle tick); predictive and
    cue_rate_max: float = 1 / 25    # distractor cues share all temporal statistics (fix, see DECISIONS)
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
