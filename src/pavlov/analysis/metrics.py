"""Lab metrics (definitions fixed in docs/DECISIONS.md 2026-11-07 before any agent ran).

CR (per trial) = mean p(saliva) over the 3 ticks before the (would-be) US
                 minus mean p(saliva) over the 10 ticks before CS onset.
OSI            = clip(1 - CR_omission / CR_yoked, 0, 1) on subject-averaged CRs over the
                 last 15 CS+ trials. 0 = fully Pavlovian, 1 = fully instrumental.
"""
from __future__ import annotations

import numpy as np

from pavlov.lab.scheduler import CS_MINUS, CS_PLUS, LabSchedule

PRE_US_TICKS = 3
BASELINE_TICKS = 10


def trial_crs(p: np.ndarray, scheds: list[LabSchedule]) -> list[np.ndarray]:
    """Per subject: CR for every trial (NaN for US-only trials)."""
    out = []
    for b, s in enumerate(scheds):
        cr = np.full(len(s.trial_onset), np.nan)
        for i, (o, k) in enumerate(zip(s.trial_onset, s.trial_kind)):
            if k in (CS_PLUS, CS_MINUS):
                win = p[b, o + s.isi - PRE_US_TICKS:o + s.isi].mean()
                base = p[b, o - BASELINE_TICKS:o].mean()
                cr[i] = win - base
        out.append(cr)
    return out


def select(crs, scheds, phase: str, kind: int, sl=slice(None)) -> np.ndarray:
    """[B, n_selected]: CRs of `kind` trials in `phase`, in presentation order, sliced by `sl`."""
    rows = []
    for cr, s in zip(crs, scheds):
        idx = np.where((s.trial_phase == s.phases.index(phase)) & (s.trial_kind == kind))[0]
        rows.append(cr[idx][sl])
    return np.array(rows)


def block(crs, scheds, phase, kind, sl=slice(None)) -> np.ndarray:
    """Per-subject mean CR over a block -> [B]."""
    return np.nanmean(select(crs, scheds, phase, kind, sl), axis=1)


def summarize(values: np.ndarray) -> dict:
    v = np.asarray(values, float)
    return {"mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)), "per_subject": v.tolist()}


def us_delivery(food: np.ndarray, scheds: list[LabSchedule], phase: str) -> np.ndarray:
    """Per subject: fraction of planned CS+ USs in `phase` that were actually delivered."""
    fr = []
    for b, s in enumerate(scheds):
        idx = np.where((s.trial_phase == s.phases.index(phase)) & (s.trial_kind == CS_PLUS) & s.trial_us)[0]
        fr.append(np.mean([food[b, s.trial_onset[i] + s.isi] for i in idx]) if len(idx) else np.nan)
    return np.array(fr)


def protocol_metrics(name: str, p: np.ndarray, food: np.ndarray, scheds: list[LabSchedule]) -> dict:
    crs = trial_crs(p, scheds)
    m = {
        "naive_cs_plus": summarize(block(crs, scheds, "naive", CS_PLUS)),
        "naive_cs_minus": summarize(block(crs, scheds, "naive", CS_MINUS)),
    }
    m["naive_discrimination"] = summarize(
        np.array(m["naive_cs_plus"]["per_subject"]) - np.array(m["naive_cs_minus"]["per_subject"]))
    if name == "pavlov_standard":
        acq_p = block(crs, scheds, "acquisition", CS_PLUS, slice(-10, None))
        acq_m = block(crs, scheds, "acquisition", CS_MINUS, slice(-10, None))
        ext_end = block(crs, scheds, "extinction", CS_PLUS, slice(-5, None))
        rec_first = select(crs, scheds, "recovery", CS_PLUS, slice(0, 1))[:, 0]
        m.update({
            "acq_cs_plus_last10": summarize(acq_p),
            "acq_cs_minus_last10": summarize(acq_m),
            "acq_discrimination": summarize(acq_p - acq_m),
            "ext_cs_plus_last5": summarize(ext_end),
            "recovery_first": summarize(rec_first),
            "acq_curve_cs_plus": np.nanmean(select(crs, scheds, "acquisition", CS_PLUS), 0).tolist(),
            "acq_curve_cs_minus": np.nanmean(select(crs, scheds, "acquisition", CS_MINUS), 0).tolist(),
            "ext_curve_cs_plus": np.nanmean(select(crs, scheds, "extinction", CS_PLUS), 0).tolist(),
            "recovery_curve": np.nanmean(select(crs, scheds, "recovery", CS_PLUS), 0).tolist(),
        })
    elif name in ("truly_random", "cs_alone", "us_alone"):
        m["acq_cs_minus_last10"] = summarize(block(crs, scheds, "acquisition", CS_MINUS, slice(-10, None)))
        if name != "us_alone":
            m["acq_cs_plus_last10"] = summarize(block(crs, scheds, "acquisition", CS_PLUS, slice(-10, None)))
        m["test_cs_plus"] = summarize(block(crs, scheds, "test", CS_PLUS))
    elif name in ("omission", "positive"):
        m["train_cs_plus_last15"] = summarize(block(crs, scheds, "training", CS_PLUS, slice(-15, None)))
        m["train_cs_minus_last10"] = summarize(block(crs, scheds, "training", CS_MINUS, slice(-10, None)))
        m["train_curve_cs_plus"] = np.nanmean(select(crs, scheds, "training", CS_PLUS), 0).tolist()
        m["us_delivered_fraction"] = summarize(us_delivery(food, scheds, "training"))
    return m


MIN_YOKED_CR = 0.05  # below this there is no CR for omission to act on (DECISIONS 2026-11-07)


def osi(cr_contingent: float, cr_yoked: float) -> float:
    """Omission Sensitivity Index on subject-averaged CRs; NaN (undefined) if the yoked CR is below
    MIN_YOKED_CR, i.e. there is no conditioned response to test."""
    if not np.isfinite(cr_yoked) or cr_yoked < MIN_YOKED_CR:
        return float("nan")
    return float(np.clip(1.0 - cr_contingent / cr_yoked, 0.0, 1.0))
