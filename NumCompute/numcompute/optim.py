from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray


def grad(
    f: Callable[[float | NDArray[np.float64]], float],
    x: npt.ArrayLike,
    h: float = 1e-5,
    method: str = "central",
) -> float | NDArray[np.float64]:
    """
    Estimate how quickly a scalar-valued function changes near a point.

    Parameters
    ----------
    f : callable
        Function that accepts x and returns a scalar.
    x : scalar or array-like
        Point at which the gradient is evaluated.
    h : float, default=1e-5
        Small change in the input used to estimate the derivative.
    method : {"central", "forward"}, default="central"
        Finite-difference method.

    Returns
    -------
    float or numpy.ndarray
        Estimated derivative for a scalar input, or gradient vector
        with the same shape as an array input.

    Raises
    ------
    ValueError
        If h is not positive/finite or method is unsupported.

    Complexity: O(d) function evaluations and O(d) workspace for d inputs.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite.")

    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")

    x_array: NDArray[np.float64] = np.asarray(x, dtype=float)

    if x_array.ndim == 0:
        x0: float = float(x_array)
        if method == "central":
            return (f(x0 + h) - f(x0 - h)) / (2 * h)
        return (f(x0 + h) - f(x0)) / h

    gradient: NDArray[np.float64] = np.zeros_like(x_array, dtype=float)
    # Forward differences reuse this value instead of reevaluating f(x).
    baseline = f(x_array.copy()) if method == "forward" else None

    for index in np.ndindex(x_array.shape):
        # Change one coordinate at a time to estimate that input's derivative.
        x_plus: NDArray[np.float64] = x_array.copy()
        x_minus: NDArray[np.float64] = x_array.copy()

        x_plus[index] += h
        x_minus[index] -= h

        if method == "central":
            gradient[index] = (f(x_plus) - f(x_minus)) / (2 * h)
        else:
            gradient[index] = (f(x_plus) - baseline) / h

    return gradient


def jacobian(
    F: Callable[[NDArray[np.float64]], npt.ArrayLike],
    x: npt.ArrayLike,
    h: float = 1e-5,
    method: str = "central",
) -> NDArray[np.float64]:
    """
    Estimate how each output of a function changes as each input changes.

    Parameters
    ----------
    F : callable
        Function that accepts x and returns an array-like output.
    x : array-like
        Point at which the Jacobian is evaluated.
    h : float, default=1e-5
        Small change in the input used to estimate each derivative.
    method : {"central", "forward"}, default="central"
        Finite-difference method.

    Returns
    -------
    numpy.ndarray
        Jacobian matrix with shape (number of outputs, number of inputs).

    Raises
    ------
    ValueError
        If h is not positive/finite or method is unsupported.

    Complexity: O(d) function evaluations (two per input for central mode),
    O(m*d) output space for m outputs and d inputs.
    """
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite.")

    if method not in {"central", "forward"}:
        raise ValueError("method must be 'central' or 'forward'.")

    x_array: NDArray[np.float64] = np.asarray(x, dtype=float)

    if x_array.ndim == 0:
        x_array = x_array.reshape(1)

    output: NDArray[np.float64] = np.asarray(F(x_array), dtype=float).reshape(-1)
    # Each column of the Jacobian describes one input; each row describes one output.
    jac: NDArray[np.float64] = np.zeros((output.size, x_array.size), dtype=float)

    for i, index in enumerate(np.ndindex(x_array.shape)):
        x_plus: NDArray[np.float64] = x_array.copy()
        x_minus: NDArray[np.float64] = x_array.copy()

        x_plus[index] += h
        x_minus[index] -= h

        if method == "central":
            f_plus: NDArray[np.float64] = np.asarray(F(x_plus), dtype=float).reshape(-1)
            f_minus: NDArray[np.float64] = np.asarray(F(x_minus), dtype=float).reshape(
                -1
            )
            jac[:, i] = (f_plus - f_minus) / (2 * h)
        else:
            f_plus: NDArray[np.float64] = np.asarray(F(x_plus), dtype=float).reshape(-1)
            jac[:, i] = (f_plus - output) / h

    return jac