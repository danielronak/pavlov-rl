"""Run a lab protocol (and, where defined, its exact yoked partner) on a batch of subjects."""
from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from pavlov.analysis.metrics import osi, protocol_metrics
from pavlov.config import PhysiologyConfig
from pavlov.lab.protocols import PROTOCOLS, YOKED
from pavlov.lab.runner import run_lab
from pavlov.lab.scheduler import build_schedule, stack_schedules

LAB_SEED = 2026  # fixed: every agent/seed meets the identical set of lab schedules


def subject_schedules(protocol_name: str, n_subjects: int, n_channels: int, lab_seed: int = LAB_SEED):
    scheds, protos = [], []
    for s in range(n_subjects):
        rng = np.random.default_rng(np.random.SeedSequence([lab_seed, list(PROTOCOLS).index(protocol_name), s]))
        proto = PROTOCOLS[protocol_name](rng)
        scheds.append(build_schedule(proto, rng, n_channels))
        protos.append(proto)
    return scheds, protos[0].contingency


def _info(scheds, contingency):
    naive_end = []
    for s in scheds:
        naive = np.where(s.trial_phase == s.phases.index("naive"))[0]
        naive_end.append(s.trial_onset[naive].max() + s.isi + 1)
    return {"bell": jnp.array([s.bell for s in scheds]),
            "naive_end": jnp.array(naive_end),
            "omission": jnp.full(len(scheds), contingency == "omission")}


def run_protocol(make_policy: Callable[[int], tuple], protocol_name: str, n_subjects: int,
                 n_channels: int, pc: PhysiologyConfig, key, lab_seed: int = LAB_SEED,
                 return_raw: bool = False) -> dict:
    """make_policy(B) -> (policy_step, initial_carry). A fresh carry is used for every run, so
    yoked partners are independent subjects with identical cues and food."""
    scheds, contingency = subject_schedules(protocol_name, n_subjects, n_channels, lab_seed)
    st = stack_schedules(scheds)
    info = _info(scheds, contingency)
    k1, k2 = jax.random.split(key)
    step, carry = make_policy(n_subjects)
    main = run_lab(step, carry, st, contingency, pc, k1, info=info)
    out = {"protocol": protocol_name, "contingency": contingency, "n_subjects": n_subjects,
           "lab_seed": lab_seed, "metrics": protocol_metrics(protocol_name, main["p"], main["food"], scheds)}
    raw = {"main": main}
    if protocol_name in YOKED:
        step_y, carry_y = make_policy(n_subjects)
        info_y = dict(info, omission=jnp.zeros(n_subjects, bool))
        yoked = run_lab(step_y, carry_y, st, "replay", pc, k2, info=info_y, replay_food=main["food"])
        assert np.array_equal(yoked["food"], main["food"]), "yoked partner must receive identical food"
        ym = protocol_metrics(protocol_name, yoked["p"], yoked["food"], scheds)
        out["yoked_metrics"] = ym
        c = out["metrics"]["train_cs_plus_last15"]["mean"]
        y = ym["train_cs_plus_last15"]["mean"]
        out["OSI" if protocol_name == "omission" else "positive_index"] = osi(c, y) if protocol_name == "omission" \
            else float(c - y)
        raw["yoked"] = yoked
    if return_raw:
        out["raw"] = raw
        out["schedules"] = scheds
    return out
