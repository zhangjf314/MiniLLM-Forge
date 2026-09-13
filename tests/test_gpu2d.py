from minillm_forge.design.gpu2d import _distribution, _nearest_rank


def test_gpu2d_nearest_rank_percentiles_are_discrete():
    values = list(range(1, 101))
    assert _nearest_rank(values, 50) == 50
    assert _nearest_rank(values, 99) == 99


def test_gpu2d_distribution_includes_required_percentiles():
    result = _distribution([1, 2, 3, 4])
    assert result == {
        "count": 4,
        "p50": 2,
        "p75": 3,
        "p90": 4,
        "p95": 4,
        "p99": 4,
        "max": 4,
        "mean": 2.5,
    }
