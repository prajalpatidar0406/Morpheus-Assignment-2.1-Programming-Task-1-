"""Helpers and a command-line benchmark for timing implementations."""

import argparse
from collections.abc import Callable
from collections.abc import Sequence
import os
import platform
from statistics import median
from time import perf_counter
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def benchmark(
    function: Callable[..., T],
    *args: object,
    repeats: int = 5,
    warmup: int = 1,
    **kwargs: object,
) -> dict[str, object]:
    """Time a function and report the results with basic environment details.

    Parameters
    ----------
    function : callable
        Function to time. If it returns a lazy iterator, the work only happens
        when that iterator is consumed, so it is not included in the timing.
    *args, **kwargs
        Arguments passed to each invocation.
    repeats : int, default=5
        Number of times to run the function while measuring.
    warmup : int, default=1
        Number of extra runs before measuring, so startup work does not affect
        the results.

    Returns
    -------
    dict
        The repeat count, minimum/median/mean time in seconds, and details about
        the Python and machine environment.

    Raises
    ------
    TypeError
        If function is not callable.
    ValueError
        If repeats is not positive or warmup is negative.

    Complexity: O((warmup + repeats) * function_cost) time.
    """
    if not callable(function):
        raise TypeError("function must be callable.")
    if (
        not isinstance(repeats, int)
        or isinstance(repeats, bool)
        or not isinstance(warmup, int)
        or isinstance(warmup, bool)
        or repeats < 1
        or warmup < 0
    ):
        raise ValueError("repeats must be positive and warmup must be non-negative.")
    # Warm-up runs are useful to the function but should not affect its timings.
    for _ in range(warmup):
        function(*args, **kwargs)
    timings: list[float] = []
    for _ in range(repeats):
        # perf_counter is intended for measuring short durations.
        started = perf_counter()
        function(*args, **kwargs)
        timings.append(perf_counter() - started)
    return {
        "repeats": repeats,
        "minimum_seconds": min(timings),
        "median_seconds": median(timings),
        "mean_seconds": sum(timings) / len(timings),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
            "cpu_count": os.cpu_count(),
        },
    }


def compare(
    functions: dict[str, Callable[..., object]],
    *args: object,
    repeats: int = 5,
    warmup: int = 1,
    **kwargs: object,
) -> dict[str, dict[str, object]]:
    """Time several named functions using the same inputs and settings.

    Check all names and functions first, so a typo does not leave you with only
    some of the comparison results.
    """
    if not functions:
        raise ValueError("functions must contain at least one implementation.")

    for name, function in functions.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("benchmark names must be non-empty strings.")
        if not callable(function):
            raise TypeError(f"implementation {name!r} must be callable.")

    return {
        name: benchmark(function, *args, repeats=repeats, warmup=warmup, **kwargs)
        for name, function in functions.items()
    }


def format_table(results: dict[str, dict[str, object]]) -> str:
    """Show benchmark timings and the runtime environment in a readable table."""
    if not results:
        raise ValueError("results must contain at least one benchmark.")
    name_width = max(len("Implementation"), *(len(name) for name in results))
    header = f"{'Implementation':<{name_width}} | Median (s) | Minimum (s)"
    separator = "-" * len(header)
    rows = []
    for name, values in results.items():
        rows.append(
            f"{name:<{name_width}} | {float(values['median_seconds']):.6g} | "
            f"{float(values['minimum_seconds']):.6g}"
        )
    # Each comparison runs in one environment, so show its details only once.
    environment = results[next(iter(results))].get("environment")
    output = "\n".join([header, separator, *rows])
    if isinstance(environment, dict):
        env_values = ", ".join(f"{key}={value}" for key, value in environment.items())
        output = f"{output}\nEnvironment: {env_values}"
    return output


def python_loop(data: np.ndarray) -> np.ndarray:
    """Apply ``value * 2 + 1`` element by element in Python."""
    result = np.empty_like(data)
    for index, value in enumerate(data):
        result[index] = value * 2.0 + 1.0
    return result


def numpy_vectorized(data: np.ndarray) -> np.ndarray:
    """Apply ``value * 2 + 1`` to an array using NumPy operations."""
    return data * 2.0 + 1.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    """Run the reproducible Python-loop versus NumPy benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=_positive_int, default=200_000)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--warmup", type=_non_negative_int, default=1)
    args = parser.parse_args(argv)

    data = np.random.default_rng(42).random(args.size)
    np.testing.assert_allclose(python_loop(data), numpy_vectorized(data))
    results = compare(
        {
            "Python loop": python_loop,
            "NumPy vectorised": numpy_vectorized,
        },
        data,
        repeats=args.repeats,
        warmup=args.warmup,
    )
    print(format_table(results))
    speedup = (
        float(results["Python loop"]["median_seconds"])
        / float(results["NumPy vectorised"]["median_seconds"])
    )
    print(f"Median speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()