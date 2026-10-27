# NumCompute

NumCompute is a numerical computing toolkit implemented using **plain Python and NumPy**.

The project implements numerical data loading, preprocessing, sorting and searching, ranking, statistics, evaluation metrics, finite-difference differentiation, and pipeline functionality.

External machine learning/deep learning libraries such as **scikit-learn, Pandas and PyTorch are not used**.

---

## Features

- Load CSV files and process large files in chunks.
- Impute missing values, scale numerical features, and encode categorical values.
- Sort, search, rank, and select values.
- Calculate descriptive statistics, quantiles, percentiles, and histograms.
- Evaluate classification and regression predictions.
- Estimate gradients and Jacobians using finite differences.
- Compose preprocessing steps into reusable pipelines.
- Benchmark Python and NumPy implementations.

---

## Project Structure

The tree below shows the project layout and modules.

```text
.
├── NumCompute/
│   └── numcompute/
│       ├── __init__.py
│       ├── io.py
│       ├── preprocessing.py
│       ├── sort_search.py
│       ├── rank.py
│       ├── stats.py
│       ├── metrics.py
│       ├── optim.py
│       ├── pipeline.py
│       ├── utils.py
│       └── benchmarking.py
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py 
│   ├── test_sort_search.py
│   ├── test_rank_stats.py
│   ├── test_metrics.py
│   ├── test_optim.py
│   ├── test_pipeline.py
│   ├── test_utils.py
│   ├── test_benchmarking.py
│   └── test_streaming_and_package.py
├── demo/
│   └── quickstart.ipynb
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── README.md
├── pyproject.toml
└── CODING_STYLE.md
```

---

## Requirements

* Python 3.10 or newer
* NumPy (installed from `requirements.txt`)
* Plain Python standard library where required

The project does not uses any external ML/DL libraries such as:
* scikit-learn
* pandas
* PyTorch

---

## Installation

Clone the repository and move into its root directory:

```bash
git clone [https://github.com/prajalpatidar0406/Morpheus-Assignment-2.1-Programming-Task-1-.git](https://github.com/prajalpatidar0406/Morpheus-Assignment-2.1-Programming-Task-1-.git)
cd Morpheus-Assignment-2.1-Programming-Task-1-
```

Create and activate a virtual environment.

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the package and development dependencies:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install .
python -m pip install jupyter
```

The installed package is named `numcompute` and can be imported from Python scripts or Jupyter notebooks.

---

## Modules

### `io.py` — Data loading

Supports:

* `load_csv(path, ...) -> (header, data)` for complete-table loading
* `iter_csv(path, chunksize=1000, ...)` for bounded-memory iteration
* Custom delimiters, requested or inferred dtypes, and fill/drop missing modes
* Numeric arrays for numeric tables and object arrays for mixed columns

`load_csv` returns the column names and a 2D NumPy array using
`numpy.loadtxt`. Use `iter_csv` to process data in bounded-memory chunks.

---
### `preprocessing.py` — Data preprocessing

Preprocessing components:

* `StandardScaler`
* `MinMaxScaler`
* `OneHotEncoder`
* `Imputer` for replacing NaN or configured missing-value markers

Preprocessors provide `fit(X)`, `transform(X)`, and `fit_transform(X)` methods.

```python
fit(X) -> self
transform(X) -> X_out
fit_transform(X) -> X_out
```

---

### `sort_search.py` - Sorting and searching algorithms

Includes:

* Stable sorting
* Multi-key sorting
* `topk(values, k, largest=True, return_indices=True)`
* Top-k selection using `np.argpartition`
* Quickselect
* Binary search

Binary search returns the insertion index and whether the value exists.

---

### `rank.py` - Ranking and percentile

`rank(data, method="average")` supports the `average`, `dense`, and `ordinal` ranking methods and handles tied values.

```python
rank(data, method="average")
```

Percentile interpolation supports `linear`, `lower`, `higher`, and `midpoint` methods.

```python
percentile(data, q, interpolation="linear")
```

---

### `stats.py` — Descriptive statistics

Basic statistical functions:

* Mean
* Median
* Standard deviation
* Minimum
* Maximum
* Histogram
* Quantiles
* NaN-aware batch statistics and exact quantiles

---

### `metrics.py` — Model evaluation

Classification metrics:

```python
accuracy(y_true, y_pred)
precision(y_true, y_pred)
recall(y_true, y_pred)
f1(y_true, y_pred)
confusion_matrix(y_true, y_pred)
```

Regression metric:

```python
mse(y_true, y_pred)
```

Optional bonus functionality:

```python
roc_curve(y_true, y_score)
auc(x, y)
```

Precision, recall, and F1 accepts a `positive_label` argument. In the confusion matrix, rows represent true labels and columns represent predicted labels.

---

### `optim.py` — Numerical differentiation

The `grad` function estimates gradients using finite differences:

```python
grad(f, x, h=1e-5, method="central")
```

Supported methods are `central` and  `forward`. Jacobian estimation is provided through:

```python
jacobian(F, x, ...)
```

---

### `pipeline.py` — Pipeline composition

Provides a minimal interface for composing transformers and estimators.

Preprocessors use:

```python
fit(X)
transform(X)
```

Models use:

```python
fit(X, y)
predict(X)
```

No machine learning model implementation is used.

The module also provides pipeline chaining, `Compose` for sequential
transformers, and `FeatureUnion` for concatenating independent feature outputs.

Example:

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler

pipe = Pipeline([("scale", StandardScaler())])
X_transformed = pipe.fit_transform(X)
```

---

### `utils.py` — Shared utilities

Contains reusable helper functionality shared across the project.

Utilities should only be added here when they are genuinely reusable across multiple modules.

---

### `benchmarking.py` — Performance measurement

Contains reusable micro-benchmarking functionality for measuring implementation performance.

`benchmark(function, ..., repeats=5, warmup=1)` returns minimum, median, and
mean elapsed times in seconds plus Python, NumPy, OS/platform, processor, and
CPU-count metadata. `compare()` benchmarks named implementations with shared
arguments, and `format_table()` renders timing and environment information.

---

## Running Tests

Run the full test suite from the repository root:

```bash
python -m pytest
```

Run an individual test file when focusing on a particular module. For example:

```bash
python -m pytest tests/test_optim.py
```

The tests cover normal usage, boundary conditions, invalid inputs, empty arrays, duplicate and tied values, NaN handling, input shapes, top-k boundary values, non-contiguous arrays, and chunked data processing.

---

# Demo

The end-to-end demonstration is located at:

```text
demo/quickstart.ipynb
```

Open `demo/quickstart.ipynb` and run the cells in order.The demo demonstrates the implemented functionality, including:

1. CSV loading
2. Preprocessing
3. Sorting and searching
4. Ranking and percentiles
5. Statistics
6. Gradients and Jacobians
7. Pipeline usage
8. Benchmark results

---

# Benchmark

The quickstart notebook compares implementations, particularly:

```text
Python loops
        vs.
NumPy vectorised implementations
```

The notebook uses a seeded 200,000-element input, one warm-up, and five timed repeats. Its results include median and minimum seconds and the runtime environment. Measurements depend on machine and system load; rerun to measure your system. `format_table(compare(...))` generates current results rather than relying on fixed timing claims.

Example local measurement (Python 3.13.7, NumPy 2.5.4, macOS 27.0.1,
arm64, 10 reported CPUs; seeded 200,000-element input, one warm-up and five
timed repeats):

| Implementation | Median (s) | Minimum (s) |
|---|---:|---:|
| Python loop | 0.00737 | 0.00692 |
| NumPy vectorised | 0.0000735 | 0.0000306 |

---

# Project Team

| Team Member   | Main Contribution |
| -------- | ----------------- |
| Prajal Patidar | io.py (shared contribution), optim.py, benchmarking.py, utils.py, pipeline.py, and project integration |
| Manthan Lad | stats.py, metrics.py, display_histogram.py, and technical report |
| Mansi Khare | io.py (shared contribution), rank.py, and presentation |
| Sarthak Sharma | preprocessing.py, sort_search.py, and technical report |
