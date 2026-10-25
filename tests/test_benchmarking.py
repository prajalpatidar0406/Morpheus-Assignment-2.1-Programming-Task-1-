import pytest

from numcompute.benchmarking import (
    benchmark,
    compare,
    format_table,
    main,
)


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


def test_benchmark_counts_warmups_and_timed_calls():
    calls = []

    result = benchmark(lambda: calls.append(None), repeats=2, warmup=1)

    assert result["repeats"] == 2
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("function", "kwargs", "error", "message"),
    [
        (None, {}, TypeError, "callable"),
        (sum, {"repeats": True}, ValueError, "repeats"),
        (sum, {"warmup": -1}, ValueError, "repeats"),
    ],
)
def test_benchmark_validates_callable_and_iteration_counts(
    function, kwargs, error, message
):
    with pytest.raises(error, match=message):
        benchmark(function, [1], **kwargs)


def test_compare_validates_all_implementations_before_running():
    called = []

    def implementation():
        called.append(True)

    with pytest.raises(TypeError, match="must be callable"):
        compare(
            {"valid": implementation, "invalid": None},
            repeats=1,
            warmup=0,
        )

    assert called == []


@pytest.mark.parametrize("name", ["", "  ", 1])
def test_compare_rejects_invalid_names(name):
    with pytest.raises(ValueError, match="non-empty strings"):
        compare({name: sum}, [1], repeats=1, warmup=0)


def test_format_table_rejects_empty_results():
    with pytest.raises(ValueError, match="at least one"):
        format_table({})


def test_vectorization_benchmark_prints_comparison(capsys):
    main(["--size", "20", "--repeats", "1", "--warmup", "0"])

    output = capsys.readouterr().out
    assert "Python loop" in output
    assert "NumPy vectorised" in output
    assert "Environment:" in output
    assert "Median speedup:" in output