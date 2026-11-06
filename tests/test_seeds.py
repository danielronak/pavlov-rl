import pytest

from pavlov.seeds import check_seed, prereg_frozen


def test_pilot_seeds_allowed():
    check_seed(100)
    check_seed(199)


def test_confirmatory_refused_before_prereg():
    if prereg_frozen():
        pytest.skip("prereg is frozen")
    with pytest.raises(PermissionError):
        check_seed(0)


def test_out_of_range():
    with pytest.raises(ValueError):
        check_seed(500)
