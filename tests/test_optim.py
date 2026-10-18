"""Behavior and edge-case tests for finite-difference derivatives."""

import numpy as np
import pytest

from numcompute.optim import grad, jacobian


# Scalar gradients should work for positive, negative, and zero-valued points.
@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (-3.0, -6.0),
        (0.0, 0.0),
        (2.0, 4.0),
    ],
)
def test_grad_central_difference_for_scalar_inputs(x: float, expected: float) -> None:
    derivative = grad(lambda value: value**2, x)

    assert derivative == pytest.approx(expected, abs=1e-8)


def test_grad_uses_central_difference_by_default() -> None:
    # A cubic has a noticeable forward-difference error, so this checks the default.
    derivative = grad(lambda value: value**3, 2.0)

    assert derivative == pytest.approx(12.0, abs=1e-8)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (-2.0, -4.0 + 1e-5),
        (0.0, 1e-5),
        (2.0, 4.0 + 1e-5),
    ],
)
def test_grad_forward_difference_for_scalar_inputs(
    x: float, expected: float
) -> None:
    # Forward differences have a small step-sized bias for a squared function.
    derivative = grad(lambda value: value**2, x, method="forward")

    assert derivative == pytest.approx(expected, abs=1e-8)


def test_grad_accepts_python_and_numpy_scalar_values() -> None:
    python_scalar_result = grad(lambda value: value**2, 3)
    numpy_scalar_result = grad(lambda value: value**2, np.float64(3.0))

    assert python_scalar_result == pytest.approx(6.0, abs=1e-8)
    assert numpy_scalar_result == pytest.approx(6.0, abs=1e-8)


def test_grad_returns_a_float_for_scalar_input() -> None:
    derivative = grad(lambda value: 5.0 * value, 2.0)

    assert isinstance(derivative, float)
    assert derivative == pytest.approx(5.0, abs=1e-8)


def test_grad_accepts_a_list_as_an_array_input() -> None:
    derivative = grad(lambda values: float(np.sum(values**2)), [1.0, -2.0])

    np.testing.assert_allclose(derivative, [2.0, -4.0], atol=1e-8)


@pytest.mark.parametrize(
    "method",
    ["central", "forward"],
)
def test_grad_finds_each_component_of_a_vector(method: str) -> None:
    point = np.array([1.0, 2.0, -3.0])
    derivative = grad(
        lambda values: float(np.sum(values**2)),
        point,
        method=method,
    )

    assert isinstance(derivative, np.ndarray)
    assert derivative.shape == point.shape
    tolerance = 2e-5 if method == "forward" else 1e-8
    np.testing.assert_allclose(derivative, [2.0, 4.0, -6.0], atol=tolerance)


def test_grad_preserves_the_shape_of_a_multidimensional_point() -> None:
    point = np.array([[1.0, 2.0], [3.0, 4.0]])
    derivative = grad(lambda values: float(np.sum(values**2)), point)

    assert derivative.shape == point.shape
    np.testing.assert_allclose(derivative, 2.0 * point, atol=1e-8)


def test_grad_returns_an_empty_gradient_for_an_empty_point() -> None:
    derivative = grad(lambda values: float(np.sum(values)), np.array([]))

    assert isinstance(derivative, np.ndarray)
    assert derivative.shape == (0,)


def test_grad_handles_non_contiguous_array_inputs() -> None:
    # Slicing with a step creates a view whose values are not stored contiguously.
    point = np.array([1.0, 99.0, 2.0, 99.0])[::2]
    derivative = grad(lambda values: float(np.sum(values**2)), point)

    np.testing.assert_allclose(derivative, [2.0, 4.0], atol=1e-8)


@pytest.mark.parametrize(
    "method",
    ["central", "forward"],
)
def test_grad_does_not_modify_the_callers_array(method: str) -> None:
    point = np.array([1.0, 2.0])
    original = point.copy()

    grad(
        lambda values: float(np.sum(values**2)),
        point,
        method=method,
    )

    np.testing.assert_array_equal(point, original)


def test_grad_uses_a_custom_finite_difference_step() -> None:
    derivative = grad(lambda value: value**2, 2.0, h=1e-3)

    assert derivative == pytest.approx(4.0, abs=1e-10)


@pytest.mark.parametrize("h", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_grad_rejects_nonpositive_or_nonfinite_steps(h: float) -> None:
    with pytest.raises(ValueError, match="h must be positive and finite"):
        grad(lambda value: value**2, 1.0, h=h)


@pytest.mark.parametrize("method", ["backward", "Central", "", None])
def test_grad_rejects_methods_it_does_not_support(method: str | None) -> None:
    with pytest.raises(ValueError, match="method must be 'central' or 'forward'"):
        grad(lambda value: value**2, 1.0, method=method)


# The Jacobian rows correspond to outputs; columns correspond to input values.
def test_jacobian_matches_a_linear_function_matrix() -> None:
    matrix = np.array([[1.0, 2.0], [-3.0, 4.0], [0.5, -1.0]])
    point = np.array([2.0, -1.0])

    result = jacobian(lambda values: matrix @ values, point)

    assert result.shape == matrix.shape
    np.testing.assert_allclose(result, matrix, atol=1e-8)


def test_jacobian_forward_difference_matches_a_linear_function_matrix() -> None:
    matrix = np.array([[1.0, 2.0], [-3.0, 4.0]])
    point = np.array([2.0, -1.0])

    result = jacobian(
        lambda values: matrix @ values,
        point,
        method="forward",
    )

    np.testing.assert_allclose(result, matrix, atol=2e-5)


@pytest.mark.parametrize(
    ("method", "tolerance"),
    [
        ("central", 1e-8),
        ("forward", 5e-5),
    ],
)
def test_jacobian_differentiates_nonlinear_vector_outputs(
    method: str, tolerance: float
) -> None:
    point = np.array([2.0, 3.0])

    result = jacobian(
        lambda values: np.array(
            [values[0] ** 2 + values[1], values[0] * values[1] ** 2]
        ),
        point,
        method=method,
    )

    np.testing.assert_allclose(result, [[4.0, 1.0], [9.0, 12.0]], atol=tolerance)


def test_jacobian_accepts_scalar_input_and_returns_one_column() -> None:
    result = jacobian(
        lambda values: np.array([values[0] ** 2, 3.0 * values[0]]),
        2.0,
    )

    assert result.shape == (2, 1)
    np.testing.assert_allclose(result, [[4.0], [3.0]], atol=1e-8)


def test_jacobian_accepts_a_list_as_an_input() -> None:
    result = jacobian(
        lambda values: np.array([values[0] + values[1]]),
        [3.0, 5.0],
    )

    np.testing.assert_allclose(result, [[1.0, 1.0]], atol=1e-8)


def test_jacobian_flattens_multidimensional_inputs_into_columns() -> None:
    point = np.array([[1.0, 2.0], [3.0, 4.0]])

    result = jacobian(
        lambda values: np.array([np.sum(values), np.sum(values**2)]),
        point,
    )

    assert result.shape == (2, point.size)
    np.testing.assert_allclose(
        result,
        [[1.0, 1.0, 1.0, 1.0], [2.0, 4.0, 6.0, 8.0]],
        atol=1e-8,
    )


def test_jacobian_flattens_multidimensional_function_outputs_into_rows() -> None:
    result = jacobian(
        lambda values: np.array(
            [
                [values[0], values[1]],
                [values[0] + values[1], values[0] - values[1]],
            ]
        ),
        [2.0, 3.0],
    )

    np.testing.assert_allclose(
        result,
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]],
        atol=1e-8,
    )


def test_jacobian_handles_a_scalar_function_output() -> None:
    result = jacobian(lambda values: values[0] ** 2 + values[1] ** 2, [2.0, 3.0])

    assert result.shape == (1, 2)
    np.testing.assert_allclose(result, [[4.0, 6.0]], atol=1e-8)


def test_jacobian_handles_an_empty_input() -> None:
    result = jacobian(
        lambda values: np.array([1.0, 2.0]) + values.size,
        np.array([]),
    )

    assert result.shape == (2, 0)


def test_jacobian_handles_an_empty_function_output() -> None:
    result = jacobian(lambda values: values[:0], [1.0, 2.0])

    assert result.shape == (0, 2)


def test_jacobian_handles_non_contiguous_array_inputs() -> None:
    # The Jacobian should follow the values in the view, not the unused entries.
    point = np.array([1.0, 99.0, 2.0, 99.0])[::2]

    result = jacobian(lambda values: values**2, point)

    np.testing.assert_allclose(result, [[2.0, 0.0], [0.0, 4.0]], atol=1e-8)


@pytest.mark.parametrize(
    "method",
    ["central", "forward"],
)
def test_jacobian_does_not_modify_the_callers_array(method: str) -> None:
    point = np.array([1.0, 2.0])
    original = point.copy()

    jacobian(lambda values: values**2, point, method=method)

    np.testing.assert_array_equal(point, original)


def test_jacobian_uses_a_custom_finite_difference_step() -> None:
    result = jacobian(lambda values: np.array([values[0] ** 2]), [2.0], h=1e-3)

    np.testing.assert_allclose(result, [[4.0]], atol=1e-10)


@pytest.mark.parametrize("h", [0.0, -1.0, np.inf, -np.inf, np.nan])
def test_jacobian_rejects_nonpositive_or_nonfinite_steps(h: float) -> None:
    with pytest.raises(ValueError, match="h must be positive and finite"):
        jacobian(lambda values: values, [1.0], h=h)


@pytest.mark.parametrize("method", ["backward", "Central", "", None])
def test_jacobian_rejects_methods_it_does_not_support(method: str | None) -> None:
    with pytest.raises(ValueError, match="method must be 'central' or 'forward'"):
        jacobian(lambda values: values, [1.0], method=method)
