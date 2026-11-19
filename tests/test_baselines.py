import jax

from pavlov.baselines.models import MODELS, make_policy
from pavlov.config import PhysiologyConfig
from pavlov.lab.experiment import run_protocol


def _acq(model):
    res = run_protocol(make_policy(model, 4), "pavlov_standard", 8, 4, PhysiologyConfig(), jax.random.PRNGKey(0))
    return res["metrics"]


def test_pavlovian_models_condition_and_start_naive():
    for m in ("rw", "td"):
        r = _acq(m)
        assert abs(r["naive_discrimination"]["mean"]) < 1e-6
        assert r["acq_discrimination"]["mean"] > 0.3


def test_pavlovian_models_are_omission_insensitive():
    for m in ("rw", "td"):
        res = run_protocol(make_policy(m, 4), "omission", 8, 4, PhysiologyConfig(), jax.random.PRNGKey(0))
        assert res["OSI"] is not None and res["OSI"] < 0.05


def test_all_models_run():
    for m in MODELS:
        _acq(m)
