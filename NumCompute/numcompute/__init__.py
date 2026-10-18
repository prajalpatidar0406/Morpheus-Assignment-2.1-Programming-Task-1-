"""NumCompute numerical utilities implemented with Python and NumPy."""

# from .benchmarking import benchmark, compare, format_table
from .io import iter_csv, load_csv
# from .metrics import (
#     accuracy,
#     auc,
#     confusion_matrix,
#     f1,
#     mse,
#     precision,
#     recall,
#     roc_curve,
# )
from .optim import grad, jacobian
from .pipeline import Compose, Estimator, FeatureUnion, Pipeline, Transformer
# from .preprocessing import Imputer, MinMaxScaler, OneHotEncoder, StandardScaler
# from .rank import percentile, rank
# from .sort_search import binary_search, multi_key_sort, quickselect, stable_sort, topk
# from .stats import (
#     StreamingStats,
#     histogram,
#     maximum,
#     mean,
#     median,
#     minimum,
#     quantiles,
#     standard_deviation,
#     variance,
# )
from .utils import (
    batches,
    cosine_similarity,
    euclidean_distance,
    logsumexp,
    manhattan_distance,
    relu,
    sigmoid,
    softmax,
    topk_indices,
)

__all__ = [
    "Compose",
    "Estimator",
    "FeatureUnion",
    # "Imputer",
    # "MinMaxScaler",
    # "OneHotEncoder",
    "Pipeline",
    # "StandardScaler",
    # "StreamingStats",
    "Transformer",
    # "accuracy",
    # "auc",
    "batches",
    # "benchmark",
    # "binary_search",
    # "compare",
    # "confusion_matrix",
    "cosine_similarity",
    "euclidean_distance",
    # "f1",
    # "format_table",
    "grad",
    # "histogram",
    "iter_csv",
    "jacobian",
    "load_csv",
    "logsumexp",
    "manhattan_distance",
    # "maximum",
    # "mean",
    # "median",
    # "minimum",
    # "mse",
    # "multi_key_sort",
    # "percentile",
    # "precision",
    # "quickselect",
    # "quantiles",
    # "rank",
    # "recall",
    "relu",
    # "roc_curve",
    "sigmoid",
    "softmax",
    # "stable_sort",
    # "standard_deviation",
    # "topk",
    "topk_indices",
    # "variance",
]
