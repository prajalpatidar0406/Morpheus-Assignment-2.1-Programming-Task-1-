import numpy as np


def _validate_labels(y_true, y_pred):
    """Validate and return equally shaped, non-empty 1D label arrays."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.ndim != 1 or pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")

    if true.size == 0 or pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity")

    if pred.dtype.kind in "fc" and not np.all(np.isfinite(pred)):
        raise ValueError("y_pred must not contain NaN or infinity")

    return true, pred


def _safe_divide(numerator, denominator, zero_division=0):
    """Divide safely when the denominator is zero."""
    if zero_division not in (0, 1):
        raise ValueError("zero_division must be 0 or 1")

    if denominator == 0:
        return float(zero_division)

    return float(numerator / denominator)


def accuracy(y_true, y_pred):
    """Return the fraction of correct predictions."""
    true, pred = _validate_labels(y_true, y_pred)
    return float(np.mean(true == pred))


def confusion_matrix(y_true, y_pred, labels=None):
    """Return a confusion matrix with true labels in rows.

    Rows represent actual labels; columns represent predicted labels.
    If labels is omitted, all observed labels are included in sorted order.
    """
    true, pred = _validate_labels(y_true, y_pred)

    if labels is None:
        # Include labels that appear only in predictions as well.
        labels = np.unique(np.concatenate((true, pred)))
    else:
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")

        if labels.size == 0:
            raise ValueError("labels must not be empty")

        if np.unique(labels).size != labels.size:
            raise ValueError("labels must not contain duplicates")

    # Map each label to its position in the matrix.
    label_to_index = {label: i for i, label in enumerate(labels.tolist())}
    matrix = np.zeros((labels.size, labels.size), dtype=np.int64)

    for actual, predicted in zip(true, pred):
        if actual not in label_to_index:
            raise ValueError(f"actual label {actual!r} is not in labels")
        if predicted not in label_to_index:
            raise ValueError(f"predicted label {predicted!r} is not in labels")

        matrix[label_to_index[actual], label_to_index[predicted]] += 1

    return matrix


def precision(y_true, y_pred, positive_label=1, zero_division=0):
    """Return binary precision for the selected positive label.

    Precision = TP / (TP + FP)
    """
    true, pred = _validate_labels(y_true, y_pred)

    actual_positive = true == positive_label
    predicted_positive = pred == positive_label

    tp = np.count_nonzero(actual_positive & predicted_positive)
    fp = np.count_nonzero(~actual_positive & predicted_positive)

    return _safe_divide(tp, tp + fp, zero_division)


def recall(y_true, y_pred, positive_label=1, zero_division=0):
    """Return binary recall for the selected positive label.

    Recall = TP / (TP + FN)
    """
    true, pred = _validate_labels(y_true, y_pred)

    actual_positive = true == positive_label
    predicted_positive = pred == positive_label

    tp = np.count_nonzero(actual_positive & predicted_positive)
    fn = np.count_nonzero(actual_positive & ~predicted_positive)

    return _safe_divide(tp, tp + fn, zero_division)


def f1(y_true, y_pred, positive_label=1, zero_division=0):
    """Return the binary F1 score."""
    p = precision(
        y_true, y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )
    r = recall(
        y_true, y_pred,
        positive_label=positive_label,
        zero_division=zero_division,
    )

    return _safe_divide(2 * p * r, p + r, zero_division)


def mse(y_true, y_pred):
    """Return mean squared error for numeric regression targets."""
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)

    if true.dtype.kind not in "iuf" or pred.dtype.kind not in "iuf":
        raise TypeError("y_true and y_pred must contain real numeric values")

    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    if true.size == 0:
        raise ValueError("y_true and y_pred must not be empty")

    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("y_true and y_pred must not contain NaN or infinity")

    true = true.astype(float, copy=False)
    pred = pred.astype(float, copy=False)

    return float(np.mean((true - pred) ** 2))


def roc_curve(y_true, y_score, positive_label=1):
    """Compute ROC curve points for binary classification.

    Parameters
    ----------
    y_true : array-like
        True binary class labels.
    y_score : array-like
        Numeric scores or probabilities for the positive class.
        Higher scores indicate a stronger prediction of the positive class.
    positive_label : scalar, default=1
        Label identifying the positive class.

    Returns
    -------
    fpr : numpy.ndarray
        False positive rates.
    tpr : numpy.ndarray
        True positive rates.
    thresholds : numpy.ndarray
        Descending score thresholds. The first threshold is infinity.
    """
    true = np.asarray(y_true)
    scores = np.asarray(y_score)

    if true.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and y_score must be 1D arrays")

    if true.size == 0 or scores.size == 0:
        raise ValueError("y_true and y_score must not be empty")

    if true.shape != scores.shape:
        raise ValueError("y_true and y_score must have the same shape")

    if scores.dtype.kind not in "iuf":
        raise TypeError("y_score must contain real numeric values")

    scores = scores.astype(float, copy=False)

    if not np.all(np.isfinite(scores)):
        raise ValueError("y_score must not contain NaN or infinity")

    if true.dtype.kind in "fc" and not np.all(np.isfinite(true)):
        raise ValueError("y_true must not contain NaN or infinity")

    classes = np.unique(true)

    if classes.size != 2:
        raise ValueError("ROC curve requires exactly two classes")

    if not np.any(true == positive_label):
        raise ValueError("positive_label must be present in y_true")

    is_positive = true == positive_label
    n_positive = np.count_nonzero(is_positive)
    n_negative = true.size - n_positive

    # Sort scores from highest to lowest.
    order = np.argsort(scores, kind="stable")[::-1]
    sorted_scores = scores[order]
    sorted_positive = is_positive[order]

    # Calculate cumulative true and false positives.
    cumulative_tp = np.cumsum(sorted_positive)
    cumulative_fp = np.cumsum(~sorted_positive)

    # Keep the last observation at each distinct score, so tied scores
    # are handled together rather than treated as separate thresholds.
    distinct_indices = np.where(np.diff(sorted_scores) != 0)[0]
    threshold_indices = np.r_[distinct_indices, sorted_scores.size - 1]

    tp = cumulative_tp[threshold_indices]
    fp = cumulative_fp[threshold_indices]

    # Include the starting point (0, 0).
    tpr = np.r_[0.0, tp / n_positive]
    fpr = np.r_[0.0, fp / n_negative]
    thresholds = np.r_[np.inf, sorted_scores[threshold_indices]]

    return fpr, tpr, thresholds


def auc(x, y):
    """Calculate the area under a curve using the trapezoidal rule.

    Parameters
    ----------
    x, y : array-like
        Coordinates of the curve. Both arrays must be 1D, have equal
        lengths, and contain at least two finite values. The x values
        must be monotonic.

    Returns
    -------
    float
        Area under the curve, returned as a non-negative value.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    if x.size < 2:
        raise ValueError("at least two points are required")

    if x.dtype.kind not in "iuf" or y.dtype.kind not in "iuf":
        raise TypeError("x and y must contain real numeric values")

    x = x.astype(float, copy=False)
    y = y.astype(float, copy=False)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("x and y must not contain NaN or infinity")

    differences = np.diff(x)

    if not (np.all(differences >= 0) or np.all(differences <= 0)):
        raise ValueError("x values must be monotonic")

    return float(abs(np.trapezoid(y, x)))


def roc_auc_score(y_true, y_score, positive_label=1):
    """Compute the area under the ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(
        y_true, y_score, positive_label=positive_label
    )
    return auc(fpr, tpr)