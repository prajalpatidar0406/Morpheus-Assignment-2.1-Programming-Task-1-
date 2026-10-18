"""A small pipeline for chaining NumPy transformers and a final estimator."""

from collections.abc import Iterable
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


class Transformer(Protocol):
    """Protocol for objects that can fit and transform feature arrays."""

    def fit(self, X: npt.ArrayLike) -> Any: ...

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike: ...


class Estimator(Protocol):
    """Protocol for final pipeline estimators."""

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Any: ...

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike: ...


class Compose:
    """Compose named 2D transformers sequentially into one transformer.

    Each step's output becomes the next step's input. The number of samples is
    preserved; feature counts may change. Runtime and storage depend on the
    underlying transforms, with one intermediate output held at a time.
    """

    def __init__(self, steps: Iterable[tuple[str, object]]) -> None:
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("steps must contain at least one named transformer.")
        names = [name for name, _ in self.steps]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("step names must be non-empty and unique.")

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Compose":
        """Fit each transformer sequentially."""
        current = X
        for _, transformer in self.steps:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError("each composed step must implement fit and transform.")
            transformer.fit(current)
            current = transformer.transform(current)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Apply each fitted transformer in sequence."""
        current = X
        for _, transformer in self.steps:
            current = transformer.transform(current)
        return current

    def fit_transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> npt.ArrayLike:
        """Fit and sequentially transform X."""
        current = X
        for _, transformer in self.steps:
            transformer.fit(current)
            current = transformer.transform(current)
        return current


class FeatureUnion:
    """Fit independent transformers and concatenate their 2D outputs by column.

    Inputs have shape ``(n_samples, n_features)`` and output has shape
    ``(n_samples, sum(branch_feature_counts))``. Branches must preserve the
    sample count. Concatenation requires output-sized memory.
    """

    def __init__(self, transformer_list: Iterable[tuple[str, object]]) -> None:
        self.transformer_list = list(transformer_list)
        if not self.transformer_list:
            raise ValueError("transformer_list must contain at least one transformer.")
        names = [name for name, _ in self.transformer_list]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("transformer names must be non-empty and unique.")

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "FeatureUnion":
        """Fit each transformer against the same input data."""
        for _, transformer in self.transformer_list:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError("each union step must implement fit and transform.")
            transformer.fit(X)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Concatenate two-dimensional transformer outputs along features."""
        outputs = [np.asarray(transformer.transform(X)) for _, transformer in self.transformer_list]
        if any(output.ndim != 2 for output in outputs):
            raise ValueError("FeatureUnion transformers must return two-dimensional arrays.")
        if len({output.shape[0] for output in outputs}) != 1:
            raise ValueError("FeatureUnion outputs must have equal sample counts.")
        return np.concatenate(outputs, axis=1)

    def fit_transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> npt.NDArray[np.generic]:
        """Fit independent branches and return their concatenated outputs."""
        return self.fit(X, y).transform(X)


class Pipeline:
    """Apply ordered transformers and optionally a final estimator.

    ``steps`` is a non-empty iterable of unique ``(name, component)`` pairs.
    All but a final object implementing ``predict`` must implement fit/transform.
    Pipeline transformation preserves sample count; output feature shape is set
    by the final transformer. Each call costs the sum of step costs.
    """

    def __init__(self, steps: Iterable[tuple[str, object]]) -> None:
        self.steps = list(steps)
        if not self.steps:
            raise ValueError("steps must contain at least one named component.")
        names = [name for name, _ in self.steps]
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("step names must be non-empty and unique.")
        if any(component is None for _, component in self.steps):
            raise ValueError("pipeline components must not be None.")
        self.named_steps = dict(self.steps)

    def _transformers(self) -> list[tuple[str, object]]:
        """Return all non-final-estimator steps."""
        if hasattr(self.steps[-1][1], "predict"):
            return self.steps[:-1]
        return self.steps

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Pipeline":
        """Fit transformers in sequence and the final estimator when present.

        X has shape ``(n_samples, n_features)``; y is required for an estimator
        final step and has the matching sample count. Returns self. Raises
        TypeError for invalid components and ValueError for a missing target.
        Runtime is the sum of the steps' fit/transform costs.
        """
        current = X
        transformers = self._transformers()
        for _, transformer in transformers:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError("non-final pipeline steps must implement fit and transform.")
            transformer.fit(current)
            current = transformer.transform(current)
        final = self.steps[-1][1]
        if hasattr(final, "predict"):
            if y is None:
                raise ValueError("y is required when the final pipeline step is an estimator.")
            if not hasattr(final, "fit"):
                raise TypeError("the final estimator must implement fit and predict.")
            final.fit(current, y)
        self._is_fitted = True
        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Transform X through transformer steps, returning the final feature array.

        Requires a fitted pipeline; otherwise raises RuntimeError. Output has
        the same sample count as X. Runtime is the sum of step transform costs.
        """
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("Pipeline must be fitted before transform.")
        current = X
        for _, transformer in self._transformers():
            current = transformer.transform(current)
        return current

    def fit_transform(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> npt.ArrayLike:
        """Fit all applicable steps and return transformed features.

        Returns an array shaped ``(n_samples, n_output_features)``. For a final
        estimator, y is required and that estimator is fitted on the transformed
        data. Raises the same TypeError/ValueError conditions as :meth:`fit`.
        """
        current = X
        transformers = self._transformers()
        for _, transformer in transformers:
            if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                raise TypeError("non-final pipeline steps must implement fit and transform.")
            transformer.fit(current)
            current = transformer.transform(current)
        final = self.steps[-1][1]
        if hasattr(final, "predict"):
            if y is None:
                raise ValueError("y is required when the final pipeline step is an estimator.")
            if not hasattr(final, "fit"):
                raise TypeError("the final estimator must implement fit and predict.")
            final.fit(current, y)
        self._is_fitted = True
        return current

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Transform X and predict with the fitted final estimator.

        Returns the final estimator's predictions. Raises RuntimeError before
        fitting and AttributeError if no estimator is configured.
        """
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("Pipeline must be fitted before predict.")
        final = self.steps[-1][1]
        if not hasattr(final, "predict"):
            raise AttributeError("the final pipeline step does not implement predict.")
        return final.predict(self.transform(X))
