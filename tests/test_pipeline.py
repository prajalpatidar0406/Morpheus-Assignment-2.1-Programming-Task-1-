import numpy as np
import pytest

from numcompute.pipeline import Pipeline
from numcompute.preprocessing import Imputer, OneHotEncoder, StandardScaler


def test_pipeline_fit_transform_chains_transformers():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pipeline = Pipeline([("scale", StandardScaler())])

    transformed = pipeline.fit_transform(values)

    np.testing.assert_allclose(transformed.mean(axis=0), [0.0, 0.0])
    np.testing.assert_allclose(pipeline.transform(values), transformed)


def test_pipeline_transform_requires_fit():
    pipeline = Pipeline([("scale", StandardScaler())])

    with pytest.raises(RuntimeError, match="fitted"):
        pipeline.transform([[1.0]])


def test_pipeline_names_must_be_unique():
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("step", StandardScaler()), ("step", StandardScaler())])


def test_pipeline_requires_target_for_estimator():
    class Estimator:
        def fit(self, X, y):
            self.fitted = True

        def predict(self, X):
            return np.zeros(len(X))

    pipeline = Pipeline([("scale", StandardScaler()), ("model", Estimator())])
    with pytest.raises(ValueError, match="y is required"):
        pipeline.fit([[1.0], [2.0]])


def test_pipeline_predict_applies_transformers():
    class Estimator:
        def fit(self, X, y):
            self.width = X.shape[1]

        def predict(self, X):
            return np.full(len(X), self.width)

    pipeline = Pipeline([("scale", StandardScaler()), ("model", Estimator())])
    pipeline.fit([[1.0], [2.0]], [0, 1])

    np.testing.assert_array_equal(pipeline.predict([[3.0], [4.0]]), [1, 1])


def test_pipeline_with_imputer_and_encoder():
    values = np.array([["red"], ["blue"], ["NaN"]], dtype=object)
    pipeline = Pipeline([("impute", Imputer(fill_value="missing")), ("encode", OneHotEncoder())])

    result = pipeline.fit_transform(values)

    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result.sum(axis=1), [1.0, 1.0, 1.0])
