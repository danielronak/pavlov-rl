"""Spontaneity invariants (CLAUDE.md section 2). These tests guard the paper's central claim."""
import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from pavlov import ecology, physiology
from pavlov.config import EcologyConfig, PhysiologyConfig
from pavlov.ecology import generate_batch, generate_schedule
from pavlov.physiology import init_phys, phys_step, run_open_loop

PC = PhysiologyConfig()


def _code_identifiers(module) -> set[str]:
    """All identifiers used in CODE (names, attributes, args, defs), ignoring docstrings/comments."""
    tree = ast.parse(inspect.getsource(module))
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            ids.add(node.id)
        elif isinstance(node, ast.Attribute):
            ids.add(node.attr)
        elif isinstance(node, ast.arg):
            ids.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            ids.add(node.name)
    return {i.lower() for i in ids}
EC = EcologyConfig(lifetime=600)


def test_reward_is_physiology_only():
    """I1: the reward function's inputs are only food onset, the salivation command and the
    digestion sign. No cue, time or trial argument exists."""
    params = list(inspect.signature(phys_step).parameters)
    assert params == ["state", "food_onset", "u", "flipped", "cfg"]
    ids = _code_identifiers(physiology)
    for forbidden in ("cue", "cues", "schedule", "expected_us", "pred_on", "isi", "perm"):
        assert forbidden not in ids, forbidden


def test_reward_values():
    """I1: with no food, reward is exactly -c*u; with food, it depends on saliva only."""
    s = init_phys()
    _, r0, _ = phys_step(s, jnp.bool_(False), jnp.int32(0), jnp.bool_(False), PC)
    _, r1, _ = phys_step(s, jnp.bool_(False), jnp.int32(1), jnp.bool_(False), PC)
    assert float(r0) == 0.0 and np.isclose(float(r1), -PC.cost)
    _, rf, _ = phys_step(s, jnp.bool_(True), jnp.int32(0), jnp.bool_(False), PC)
    assert np.isclose(float(rf), PC.energy / PC.food_ticks * PC.b0)


def test_salivation_does_not_affect_world():
    """I2: the world schedule is a function of the key only. generate_schedule takes no action."""
    params = list(inspect.signature(generate_schedule).parameters)
    assert params == ["key", "cfg"]
    a = generate_schedule(jax.random.PRNGKey(3), EC)
    b = generate_schedule(jax.random.PRNGKey(3), EC)
    assert np.array_equal(a.cues, b.cues) and np.array_equal(a.food_onset, b.food_onset)


def test_anticipation_is_causal_through_saliva_only():
    """I2: salivation changes energy only via saliva level at food time."""
    food = jnp.zeros(20, bool).at[10].set(True)
    none = run_open_loop(food, jnp.zeros(20, bool), jnp.bool_(False), PC)
    early = run_open_loop(food, jnp.zeros(20, bool).at[7:10].set(True), jnp.bool_(False), PC)
    late = run_open_loop(food, jnp.zeros(20, bool).at[15:18].set(True), jnp.bool_(False), PC)
    assert early["energy"].sum() > none["energy"].sum()
    assert np.isclose(late["energy"].sum(), none["energy"].sum())  # after food: no benefit
    assert np.array_equal(none["food_present"], early["food_present"])


def test_cue_permutation():
    """I3: with q=1 the predictive cue lands on different physical channels across lifetimes;
    with q=0 it is always channel 0."""
    b1 = generate_batch(jax.random.PRNGKey(0), EcologyConfig(lifetime=600, q_so_variability=1.0), 200)
    chan = np.asarray(b1.params.perm[:, 0])
    assert len(set(chan.tolist())) == EC.n_channels
    counts = np.bincount(chan, minlength=EC.n_channels)
    assert counts.min() > 200 / EC.n_channels * 0.5  # roughly uniform
    b0 = generate_batch(jax.random.PRNGKey(0), EcologyConfig(lifetime=600, q_so_variability=0.0), 50)
    assert np.all(np.asarray(b0.params.perm[:, 0]) == 0)


def test_predictive_cue_actually_predicts_food():
    """Sanity: food follows predictive-cue onsets at the lifetime's ISI more than chance."""
    s = generate_schedule(jax.random.PRNGKey(1), EcologyConfig(lifetime=3000, hazard=0.0))
    exp = np.asarray(s.expected_us)
    food = np.asarray(s.food_onset)
    assert food[exp].mean() > 0.4 and food[~exp].mean() < 0.02


def test_ecology_has_no_omission():
    """I6: nothing in the training ecology lets actions cancel or cause food."""
    ids = _code_identifiers(ecology)
    assert not any(("omission" in i) or ("action" in i) or ("saliva" in i) or (i == "u") for i in ids), ids


def test_flip_arm_inverts_saliva_effect():
    food = jnp.zeros(20, bool).at[10].set(True)
    pre = jnp.zeros(20, bool).at[7:10].set(True)
    base = run_open_loop(food, jnp.zeros(20, bool), jnp.bool_(True), PC)["energy"].sum()
    sal = run_open_loop(food, pre, jnp.bool_(True), PC)["energy"].sum()
    assert sal < base


def _runs(on):
    """Onset count and on-run durations of one binary channel."""
    on = np.asarray(on).astype(int)
    d = np.diff(np.concatenate([[0], on, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
    return len(starts), (ends - starts)


def test_cue_types_statistically_identical_except_food():
    """I3: a predictive cue and a distractor cue must be indistinguishable from cue statistics
    alone (onset counts, durations). Only their relation to food may differ."""
    from scipy.stats import ks_2samp
    cfg = EcologyConfig(lifetime=3000, hazard=0.0)
    b = generate_batch(jax.random.PRNGKey(11), cfg, 300)
    cues, perm = np.asarray(b.cues), np.asarray(b.params.perm)
    n_pred, n_dist, d_pred, d_dist = [], [], [], []
    for i in range(cues.shape[0]):
        cp, dp = _runs(cues[i, :, perm[i, 0]])               # logical 0: always predictive
        cd, dd = _runs(cues[i, :, perm[i, cfg.n_channels - 1]])  # last logical: always distractor
        n_pred.append(cp); n_dist.append(cd); d_pred.append(dp.mean()); d_dist.append(dd.mean())  # one value per lifetime (durations are fixed within a lifetime)
    assert ks_2samp(n_pred, n_dist).pvalue > 1e-3
    assert ks_2samp(d_pred, d_dist).pvalue > 1e-3
    assert abs(np.mean(n_pred) / np.mean(n_dist) - 1) < 0.05
