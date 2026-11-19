import inspect

import numpy as np

from pavlov.lab import phenomena as P


def test_builder_is_blind():
    # schedules depend only on the design, an RNG and the number of channels (I7)
    assert list(inspect.signature(P.build).parameters) == ["trials", "rng", "n_channels"]
    assert list(inspect.signature(P.schedules).parameters) == ["design", "n_subjects", "n_channels", "lab_seed"]


def test_roles_get_distinct_random_channels():
    seen = set()
    for g, scheds in P.schedules("blocking", 16, 4, 2026).items():
        for s in scheds:
            assert len(set(s.roles.values())) == len(s.roles)
            seen.add(s.roles["X"])
    assert len(seen) > 1  # no fixed channel for a role (I3)


def test_us_only_where_planned_and_pree_matches_us_count():
    S = P.schedules("pree", 4, 4, 2026)
    for g in ("p100", "p50"):
        for s in S[g]:
            assert s.us_planned.sum() == 20 == sum(t.us_at is not None for t in s.trials)
            for o, t in zip(s.trial_onset, s.trials):
                if t.us_at is not None:
                    assert s.us_planned[o + t.us_at]


def test_compound_and_serial_cues():
    s = P.schedules("blocking", 1, 4, 2026)["blocking"][0]
    i = next(k for k, t in enumerate(s.trials) if t.tag == "AX")
    o = s.trial_onset[i]
    assert s.cues[o:o + 7, s.roles["A"]].all() and s.cues[o:o + 7, s.roles["X"]].all()
    s = P.schedules("second_order", 1, 4, 2026)["paired"][0]
    i = next(k for k, t in enumerate(s.trials) if t.tag == "BA")
    o = s.trial_onset[i]
    assert s.cues[o:o + 6, s.roles["B"]].all() and not s.cues[o:o + 6, s.roles["A"]].any()
    assert s.cues[o + 6:o + 13, s.roles["A"]].all() and not s.us_planned[o:o + 13].any()


def test_schedules_are_identical_across_calls():
    a = P.schedules("second_order", 3, 4, 2026)
    b = P.schedules("second_order", 3, 4, 2026)
    for g in a:
        for x, y in zip(a[g], b[g]):
            assert np.array_equal(x.cues, y.cues) and np.array_equal(x.us_planned, y.us_planned)
