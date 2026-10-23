"""Behavior and edge-case tests for classification and regression metrics."""

import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_auc_score,
    roc_curve,
)

# Shared example: TP=3, FP=1, FN=2, TN=4 when 1 is the positive label.
Y_TRUE = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 1, 0, 0, 1, 0, 0, 0, 0]


def test_accuracy_of_a_mixed_prediction() -> None:
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.7)


def test_precision_of_a_mixed_prediction() -> None:
    assert precision(Y_TRUE, Y_PRED) == pytest.approx(3 / 4)


def test_recall_of_a_mixed_prediction() -> None:
    assert recall(Y_TRUE, Y_PRED) == pytest.approx(3 / 5)


def test_f1_of_a_mixed_prediction() -> None:
    p, r = 3 / 4, 3 / 5

    assert f1(Y_TRUE, Y_PRED) == pytest.approx(2 * p * r / (p + r))


def test_confusion_matrix_for_binary_labels() -> None:
    result = confusion_matrix(Y_TRUE, Y_PRED)

    # Rows are actual labels; columns are predicted labels.
    np.testing.assert_array_equal(result, [[4, 1], [2, 3]])


def test_confusion_matrix_for_multiclass_labels() -> None:
    result = confusion_matrix([0, 1, 2, 2, 1], [0, 2, 2, 1, 1])

    np.testing.assert_array_equal(result, [[1, 0, 0], [0, 1, 1], [0, 1, 1]])


def test_mse_of_a_simple_prediction() -> None:
    assert mse([1, 2, 3], [1, 2, 5]) == pytest.approx(4 / 3)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        ([], [], "must not be empty"),
        ([1, 0, 1], [1, 0], "must have the same shape"),
    ],
)
def test_metrics_reject_empty_or_mismatched_inputs(
    y_true: list, y_pred: list, message: str
) -> None:
    for func in (accuracy, confusion_matrix, precision, recall, f1):
        with pytest.raises(ValueError, match=message):
            func(y_true, y_pred)

    with pytest.raises(ValueError, match=message):
        mse(y_true, y_pred)


@pytest.mark.parametrize("func", [precision, recall, f1])
@pytest.mark.parametrize(
    ("zero_division", "expected"),
    [
        (0, 0.0),
        (1, 1.0),
    ],
)
def test_zero_division_fallback_when_there_are_no_positives(
    func, zero_division: int, expected: float
) -> None:
    # No positive labels appear anywhere, so the ratio is 0 / 0.
    assert func([0, 0], [0, 0], zero_division=zero_division) == expected


# ROC example: scores for two negatives (0.1, 0.4) and two positives (0.35, 0.8).
ROC_TRUE = [0, 0, 1, 1]
ROC_SCORES = [0.1, 0.4, 0.35, 0.8]


def test_roc_curve_returns_known_points() -> None:
    fpr, tpr, thresholds = roc_curve(ROC_TRUE, ROC_SCORES)

    np.testing.assert_allclose(fpr, [0.0, 0.0, 0.5, 0.5, 1.0])
    np.testing.assert_allclose(tpr, [0.0, 0.5, 0.5, 1.0, 1.0])
    np.testing.assert_allclose(thresholds, [np.inf, 0.8, 0.4, 0.35, 0.1])


def test_roc_curve_merges_tied_scores_into_one_threshold() -> None:
    fpr, tpr, thresholds = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])

    np.testing.assert_allclose(fpr, [0.0, 1.0])
    np.testing.assert_allclose(tpr, [0.0, 1.0])
    np.testing.assert_allclose(thresholds, [np.inf, 0.5])


def test_roc_curve_uses_a_custom_positive_label() -> None:
    fpr, tpr, _ = roc_curve(
        ["a", "b", "a", "b"],
        [0.1, 0.9, 0.2, 0.8],
        positive_label="b",
    )

    np.testing.assert_allclose(fpr, [0.0, 0.0, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(tpr, [0.0, 0.5, 1.0, 1.0, 1.0])


@pytest.mark.parametrize(
    ("y_true", "y_score", "kwargs", "error", "message"),
    [
        ([], [], {}, ValueError, "must not be empty"),
        ([0, 1, 1], [0.1, 0.9], {}, ValueError, "must have the same shape"),
        ([[0, 1]], [[0.1, 0.9]], {}, ValueError, "must be 1D arrays"),
        ([0, 1], [0.1, np.nan], {}, ValueError, "must not contain NaN or infinity"),
        ([0, 1], ["a", "b"], {}, TypeError, "real numeric values"),
        ([1, 1, 1], [0.1, 0.5, 0.9], {}, ValueError, "exactly two classes"),
        ([0, 1, 2], [0.1, 0.5, 0.9], {}, ValueError, "exactly two classes"),
        ([0, 1], [0.1, 0.9], {"positive_label": 5}, ValueError, "positive_label"),
    ],
)
def test_roc_curve_rejects_invalid_inputs(
    y_true: list, y_score: list, kwargs: dict, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        roc_curve(y_true, y_score, **kwargs)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        ([0, 1], [0, 1], 0.5),
        ([0, 0.5, 1], [0, 1, 1], 0.75),
        # Decreasing x gives the same non-negative area.
        ([1, 0.5, 0], [1, 1, 0], 0.75),
    ],
)
def test_auc_uses_the_trapezoidal_rule(x: list, y: list, expected: float) -> None:
    assert auc(x, y) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("x", "y", "error", "message"),
    [
        ([0, 1, 2], [0, 1], ValueError, "must have the same length"),
        ([0], [0], ValueError, "at least two points"),
        ([[0, 1]], [[0, 1]], ValueError, "must be 1D arrays"),
        (["a", "b"], [0, 1], TypeError, "real numeric values"),
        ([0, np.nan], [0, 1], ValueError, "must not contain NaN or infinity"),
        ([0, 2, 1], [0, 1, 2], ValueError, "must be monotonic"),
    ],
)
def test_auc_rejects_invalid_inputs(
    x: list, y: list, error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        auc(x, y)


@pytest.mark.parametrize(
    ("y_true", "y_score", "expected"),
    [
        (ROC_TRUE, ROC_SCORES, 0.75),
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),
        ([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], 0.0),
        ([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], 0.5),
    ],
)
def test_roc_auc_score_for_known_rankings(
    y_true: list, y_score: list, expected: float
) -> None:
    assert roc_auc_score(y_true, y_score) == pytest.approx(expected)


def test_roc_auc_score_matches_auc_of_the_roc_curve() -> None:
    fpr, tpr, _ = roc_curve(ROC_TRUE, ROC_SCORES)

    assert roc_auc_score(ROC_TRUE, ROC_SCORES) == pytest.approx(auc(fpr, tpr))