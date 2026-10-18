"""Small, reproducible micro-benchmark helpers."""

from collections.abc import Callable
import os
import platform
from statistics import median
import sys
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
    """Measure a callable and return timings plus environment metadata.

    Parameters
    ----------
    function : callable
        Function to invoke.
    *args, **kwargs
        Arguments passed to each invocation.
    repeats : int, default=5
        Number of timed invocations.
    warmup : int, default=1
        Number of untimed invocations.

    Returns
    -------
    dict
        Repeat count, minimum/median/mean elapsed seconds, and a stable
        environment record (Python, NumPy, platform, processor, and CPU count).

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
    for _ in range(warmup):
        function(*args, **kwargs)
    timings: list[float] = []
    for _ in range(repeats):
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
    """Benchmark named implementations with common arguments."""
    if not functions:
        raise ValueError("functions must contain at least one implementation.")
    if any(not name for name in functions):
        raise ValueError("benchmark names must be non-empty.")
    return {
        name: benchmark(function, *args, repeats=repeats, warmup=warmup, **kwargs)
        for name, function in functions.items()
    }


def format_table(results: dict[str, dict[str, object]]) -> str:
    """Format timing comparisons and their runtime environment as a text table."""
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
    environment = results[next(iter(results))].get("environment")
    output = "\n".join([header, separator, *rows])
    if isinstance(environment, dict):
        env_values = ", ".join(f"{key}={value}" for key, value in environment.items())
        output = f"{output}\nEnvironment: {env_values}"
    return output
