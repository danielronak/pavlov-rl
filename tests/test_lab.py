"""Lab invariants: I5 (frozen weights), I7 (blind scheduler), I8 (naive phase), contingency logic."""
import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from pavlov import agent as A
from pavlov.config import PhysiologyConfig
from pavlov.lab import experiment, policies, runner, scheduler
from pavlov.lab.protocols import PROTOCOLS, Protocol, Trial
from pavlov.lab.scheduler import CS_PLUS, build_schedule, stack_schedules

PC = PhysiologyConfig()


def _imports(module):
    tree = ast.parse(inspect.getsource(module))
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            names |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            names.add(n.module or "")
    return names


def test_lab_frozen_no_optimizer_imports():
    """I5: no lab module imports an optimizer or a training loop."""
    for mod in (runner, scheduler, policies, experiment):
        imps = _imports(mod)
        assert not any(i.startswith(("optax", "pavlov.ppo", "pavlov.es")) for i in imps), (mod, imps)


def test_lab_frozen_params_unchanged():
    """I5: running an agent through the lab leaves its parameters bit-identical."""
    acfg = A.AgentConfig(hidden=8, embed=8)
    params = A.init_params(jax.random.PRNGKey(0), 8, acfg)
    before = jax.tree.map(np.array, params)
    experiment.run_protocol(lambda B: policies.agent_policy(params, 8, B), "omission", 4, 4, PC,
                            jax.random.PRNGKey(1))
    after = jax.tree.map(np.array, params)
    assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after)))


def test_scheduler_is_blind():
    """I7: building a schedule takes no agent, params or outputs."""
    assert list(inspect.signature(build_schedule).parameters) == ["protocol", "rng", "n_channels", "isi", "iti"]
    a = build_schedule(PROTOCOLS["pavlov_standard"](np.random.default_rng(0)), np.random.default_rng(1), 4)
    b = build_schedule(PROTOCOLS["pavlov_standard"](np.random.default_rng(0)), np.random.default_rng(1), 4)
    assert np.array_equal(a.cues, b.cues) and np.array_equal(a.us_planned, b.us_planned)


def test_every_protocol_has_naive_phase_first():
    """I8: CR to the bell before any pairing is always measured."""
    for name, fn in PROTOCOLS.items():
        proto = fn(np.random.default_rng(0))
        s = build_schedule(proto, np.random.default_rng(0), 4)
        assert s.phases[0] == "naive"
        first_us = np.where(s.us_planned)[0]
        naive_end = s.trial_onset[s.trial_phase == 0].max() + s.isi
        assert (first_us.size == 0) or first_us.min() > naive_end, name
        assert s.bell != s.metronome


def _toy_sched(contingent):
    proto = Protocol("toy", [Trial("naive", CS_PLUS)] + [Trial("training", CS_PLUS, us=True, contingent=contingent)] * 3,
                     contingency="omission", phases=["naive", "training"])
    s = build_schedule(proto, np.random.default_rng(0), 4)
    return s, stack_schedules([s])


def _const_policy(value):
    def step(carry, obs, t, info):
        return carry, jnp.full(obs.shape[0], value)
    return step


def test_omission_cancels_only_when_salivating():
    s, st = _toy_sched(True)
    always = runner.run_lab(_const_policy(1.0), jnp.zeros(1), st, "omission", PC, jax.random.PRNGKey(0))
    never = runner.run_lab(_const_policy(0.0), jnp.zeros(1), st, "omission", PC, jax.random.PRNGKey(0))
    assert always["food"].sum() == 0
    assert never["food"].sum() == s.us_planned.sum() == 3


def test_positive_delivers_only_when_salivating():
    s, st = _toy_sched(True)
    always = runner.run_lab(_const_policy(1.0), jnp.zeros(1), st, "positive", PC, jax.random.PRNGKey(0))
    never = runner.run_lab(_const_policy(0.0), jnp.zeros(1), st, "positive", PC, jax.random.PRNGKey(0))
    assert always["food"].sum() == 3 and never["food"].sum() == 0


def test_yoked_partner_gets_identical_cues_and_food():
    res = experiment.run_protocol(lambda B: policies.oracle_policy("bell_reader", B), "omission", 4, 4, PC,
                                  jax.random.PRNGKey(0), return_raw=True)
    assert np.array_equal(res["raw"]["main"]["food"], res["raw"]["yoked"]["food"])


def test_osi_undefined_without_a_cr():
    import math
    from pavlov.analysis.metrics import osi
    assert math.isnan(osi(6.7e-7, 6.7e-7))
    assert math.isnan(osi(0.0, 0.0))
    assert osi(0.0, 0.5) == 1.0 and osi(0.5, 0.5) == 0.0
