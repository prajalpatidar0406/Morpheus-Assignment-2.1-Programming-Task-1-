"""Check that composed transformers and estimators pass data through correctly."""

import numpy as np
import pytest

from numcompute.pipeline import Compose, FeatureUnion, Pipeline
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


def test_pipeline_failed_refit_disables_prediction():
    class Estimator:
        def fit(self, X, y):
            self.fitted = True

        def predict(self, X):
            return np.zeros(len(X))

    pipeline = Pipeline([("scale", StandardScaler()), ("model", Estimator())])
    pipeline.fit([[1.0], [2.0]], [0, 1])

    with pytest.raises(ValueError, match="y is required"):
        pipeline.fit([[10.0], [20.0]])

    with pytest.raises(RuntimeError, match="fitted"):
        pipeline.predict([[1.0]])


def test_pipeline_with_imputer_and_encoder():
    values = np.array([["red"], ["blue"], ["NaN"]], dtype=object)
    pipeline = Pipeline(
        [("impute", Imputer(fill_value="missing")), ("encode", OneHotEncoder())]
    )

    result = pipeline.fit_transform(values)

    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result.sum(axis=1), [1.0, 1.0, 1.0])


class AddOffset:
    def __init__(self, offset):
        self.offset = offset

    def fit(self, X):
        return self

    def transform(self, X):
        return np.asarray(X) + self.offset


def test_compose_fits_and_applies_steps_in_order():
    compose = Compose([("add", AddOffset(1)), ("add_again", AddOffset(2))])
    values = np.array([[1.0], [2.0]])

    assert compose.fit(values) is compose
    np.testing.assert_array_equal(compose.transform(values), [[4.0], [5.0]])

    fitted = compose.fit_transform(values)

    np.testing.assert_array_equal(fitted, [[4.0], [5.0]])
    np.testing.assert_array_equal(compose.transform(values), fitted)


def test_compose_rejects_steps_without_fit_and_transform():
    compose = Compose([("invalid", object())])

    with pytest.raises(TypeError, match="fit and transform"):
        compose.fit_transform([[1.0]])


def test_composition_helpers_validate_configuration_and_components():
    with pytest.raises(ValueError, match="at least one"):
        Compose([])
    with pytest.raises(ValueError, match="non-empty and unique"):
        FeatureUnion([("same", AddOffset(0)), ("same", AddOffset(1))])

    union = FeatureUnion([("invalid", object())])
    with pytest.raises(TypeError, match="fit and transform"):
        union.fit([[1.0]])


def test_feature_union_concatenates_branch_outputs():
    class FirstColumn(AddOffset):
        def transform(self, X):
            return np.asarray(X)[:, :1] + self.offset

    class SecondColumn(AddOffset):
        def transform(self, X):
            return np.asarray(X)[:, 1:2] + self.offset

    values = np.array([[1.0, 10.0], [2.0, 20.0]])
    union = FeatureUnion([("first", FirstColumn(1)), ("second", SecondColumn(-1))])

    result = union.fit_transform(values)

    np.testing.assert_array_equal(result, [[2.0, 9.0], [3.0, 19.0]])


@pytest.mark.parametrize(
    "transform",
    [
        lambda X: np.asarray(X)[:, 0],
        lambda X: np.asarray(X)[:-1],
    ],
)
def test_feature_union_rejects_invalid_branch_shapes(transform):
    class Transformer:
        def fit(self, X):
            return self

        def transform(self, X):
            return transform(X)

    class Identity:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X)

    union = FeatureUnion([("branch", Transformer()), ("identity", Identity())])

    with pytest.raises(ValueError, match="two-dimensional|equal sample counts"):
        union.fit_transform([[1.0], [2.0]])


def test_pipeline_rejects_changed_feature_count_and_predict_without_estimator():
    pipeline = Pipeline([("scale", StandardScaler())])
    pipeline.fit([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="feature count"):
        pipeline.transform([[1.0]])

    with pytest.raises(AttributeError, match="does not implement predict"):
        pipeline.predict([[1.0, 2.0]])


def test_pipeline_rejects_invalid_transformer_and_estimator_components():
    with pytest.raises(TypeError, match="fit and transform"):
        Pipeline([("invalid", object())]).fit([[1.0]])

    class PredictOnly:
        def predict(self, X):
            return np.zeros(len(X))

    with pytest.raises(TypeError, match="fit and predict"):
        Pipeline([("model", PredictOnly())]).fit([[1.0]], [0])