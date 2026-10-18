import pytest

from numcompute.benchmarking import benchmark, compare, format_table


def test_benchmark_returns_timing_summary():
    result = benchmark(sum, [1, 2, 3], repeats=2, warmup=0)
    assert result["repeats"] == 2
    assert result["minimum_seconds"] >= 0
    assert result["median_seconds"] >= result["minimum_seconds"]


def test_compare_and_format_table():
    results = compare({"sum": sum, "max": max}, [1, 2, 3], repeats=1, warmup=0)
    table = format_table(results)
    assert "Implementation" in table
    assert "sum" in table and "max" in table
    assert "Environment:" in table
    assert results["sum"]["environment"]["numpy"]


def test_benchmark_validates_repeat_count():
    with pytest.raises(ValueError, match="repeats"):
        benchmark(sum, [1], repeats=0)
