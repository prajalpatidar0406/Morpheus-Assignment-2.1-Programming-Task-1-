"""A small pipeline for chaining NumPy transformers and a final estimator."""

from collections.abc import Iterable
from copy import deepcopy
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler


def _require_transformer(transformer: object, message: str) -> "Transformer":
    if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
        raise TypeError(message)
    return cast(Transformer, transformer)


class Transformer(Protocol):
    """Protocol for objects that can fit and transform feature arrays."""

    def fit(self, X: npt.ArrayLike) -> object: ...

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike: ...


class Estimator(Protocol):
    """Protocol for final pipeline estimators."""

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> object: ...

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

    def _fit_transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        current = X
        for _, transformer in self.steps:
            transformer = _require_transformer(
                transformer, "each composed step must implement fit and transform."
            )
            transformer.fit(current)
            current = transformer.transform(current)
        return current

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Compose":
        """Fit each transformer sequentially."""
        self._fit_transform(X)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Apply each fitted transformer in sequence."""
        current = X
        for _, transformer in self.steps:
            transformer = _require_transformer(
                transformer, "each composed step must implement fit and transform."
            )
            current = transformer.transform(current)
        return current

    def fit_transform(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> npt.ArrayLike:
        """Fit and sequentially transform X."""
        return self._fit_transform(X)


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
            transformer = _require_transformer(
                transformer, "each union step must implement fit and transform."
            )
            transformer.fit(X)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Concatenate two-dimensional transformer outputs along features."""
        outputs = [
            np.asarray(
                _require_transformer(
                    transformer, "each union step must implement fit and transform."
                ).transform(X)
            )
            for _, transformer in self.transformer_list
        ]
        if any(output.ndim != 2 for output in outputs):
            raise ValueError(
                "FeatureUnion transformers must return two-dimensional arrays."
            )
        if len({output.shape[0] for output in outputs}) != 1:
            raise ValueError("FeatureUnion outputs must have equal sample counts.")
        return np.concatenate(outputs, axis=1)

    def fit_transform(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.generic]:
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
        self._fitted_transformers: dict[str, list[Transformer]] = {}
        self._columnwise_steps: set[str] = set()
        self._is_fitted = False

    @staticmethod
    def _as_2d_output(
        output: npt.ArrayLike, sample_count: int
    ) -> npt.NDArray[np.generic]:
        """Normalize a feature transform's output for column-wise composition."""
        result = np.asarray(output)
        if result.ndim == 1:
            # A single output value per row is still one feature column.
            result = result.reshape(-1, 1)
        if result.ndim != 2 or result.shape[0] != sample_count:
            raise ValueError("transformer outputs must preserve the sample count.")
        return result

    def _fit_transform_step(
        self, name: str, transformer: object, X: npt.ArrayLike
    ) -> npt.NDArray[np.generic]:
        """Fit a 1-D transformer independently on each input feature."""
        transformer = _require_transformer(
            transformer, "non-final pipeline steps must implement fit and transform."
        )
        values = np.asarray(X)
        columnwise = isinstance(
            transformer, (Imputer, MinMaxScaler, OneHotEncoder, StandardScaler)
        )
        if values.ndim != 2 or not columnwise:
            transformer.fit(X)
            self._fitted_transformers[name] = [transformer]
            return np.asarray(transformer.transform(X))

        self._columnwise_steps.add(name)
        fitted: list[Transformer] = []
        outputs = []
        for index in range(values.shape[1]):
            # Each feature needs its own learned statistics and category set.
            feature_transformer = transformer if index == 0 else deepcopy(transformer)
            feature = values[:, index]
            feature_transformer.fit(feature)
            fitted.append(feature_transformer)
            outputs.append(
                self._as_2d_output(
                    feature_transformer.transform(feature), values.shape[0]
                )
            )

        self._fitted_transformers[name] = fitted
        return np.concatenate(outputs, axis=1)

    def _transform_step(
        self, name: str, transformer: object, X: npt.ArrayLike
    ) -> npt.ArrayLike:
        """Transform each feature with its corresponding fitted transformer."""
        values = np.asarray(X)
        if values.ndim != 2 or name not in self._columnwise_steps:
            return _require_transformer(
                transformer,
                "non-final pipeline steps must implement fit and transform.",
            ).transform(X)

        fitted = self._fitted_transformers[name]
        if values.shape[1] != len(fitted):
            raise ValueError("input feature count differs from the data used to fit.")
        outputs = []
        for index, feature_transformer in enumerate(fitted):
            outputs.append(
                self._as_2d_output(
                    feature_transformer.transform(values[:, index]), values.shape[0]
                )
            )
        return np.concatenate(outputs, axis=1)

    def _transformers(self) -> list[tuple[str, object]]:
        """Return all non-final-estimator steps."""
        # A final object with predict() is a model, not another data transform.
        if hasattr(self.steps[-1][1], "predict"):
            return self.steps[:-1]
        return self.steps

    def _validate_final_estimator(self, y: npt.ArrayLike | None) -> Estimator | None:
        """Validate the final estimator and target before fitting transformers."""
        final = self.steps[-1][1]
        if not hasattr(final, "predict"):
            return None
        if not hasattr(final, "fit"):
            raise TypeError("the final estimator must implement fit and predict.")
        if y is None:
            raise ValueError(
                "y is required when the final pipeline step is an estimator."
            )
        return cast(Estimator, final)

    def _fit_transformers(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Fit and apply every transformer step in order."""
        current = X
        self._fitted_transformers = {}
        self._columnwise_steps = set()
        for name, transformer in self._transformers():
            transformer = _require_transformer(
                transformer,
                "non-final pipeline steps must implement fit and transform.",
            )
            current = self._fit_transform_step(name, transformer, current)
        return current

    def _fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None) -> npt.ArrayLike:
        self._is_fitted = False
        final = self._validate_final_estimator(y)
        current = self._fit_transformers(X)
        if final is not None:
            assert y is not None
            # Train the final model on features after all preprocessing steps.
            final.fit(current, y)
        self._is_fitted = True
        return current

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> "Pipeline":
        """Fit transformers in sequence and the final estimator when present.

        X has shape ``(n_samples, n_features)``; y is required for an estimator
        final step and has the matching sample count. Returns self. Raises
        TypeError for invalid components and ValueError for a missing target.
        Runtime is the sum of the steps' fit/transform costs.
        """
        self._fit(X, y)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Transform X through transformer steps, returning the final feature array.

        Requires a fitted pipeline; otherwise raises RuntimeError. Output has
        the same sample count as X. Runtime is the sum of step transform costs.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform.")
        current = X
        for name, transformer in self._transformers():
            current = self._transform_step(name, transformer, current)
        return current

    def fit_transform(
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> npt.ArrayLike:
        """Fit all applicable steps and return transformed features.

        Returns an array shaped ``(n_samples, n_output_features)``. For a final
        estimator, y is required and that estimator is fitted on the transformed
        data. Raises the same TypeError/ValueError conditions as :meth:`fit`.
        """
        return self._fit(X, y)

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """Transform X and predict with the fitted final estimator.

        Returns the final estimator's predictions. Raises RuntimeError before
        fitting and AttributeError if no estimator is configured.
        """
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before predict.")
        final = self.steps[-1][1]
        if not hasattr(final, "predict"):
            raise AttributeError("the final pipeline step does not implement predict.")
        return cast(Estimator, final).predict(self.transform(X))