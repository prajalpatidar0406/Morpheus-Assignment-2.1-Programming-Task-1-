# NumCompute

NumCompute is a numerical computing toolkit implemented using **plain Python and NumPy**.

The project implements numerical data loading, preprocessing, sorting and searching, ranking, statistics, evaluation metrics, finite-difference differentiation, and pipeline functionality.

External machine learning/deep learning libraries such as **scikit-learn, pandas, and PyTorch are not used**.

---

## Project Structure

The tree below shows the project layout and assignment modules.

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

The assignment does not permit external ML/DL libraries such as:

* scikit-learn
* pandas
* PyTorch

Install the package from the repository root:

```bash
python -m pip install .
```

The importable and installable package name is `numcompute`.

---

## Modules

### `io.py`

CSV data loading using NumPy.

Supports:

* `load_csv(path, ...) -> (header, data)` for complete-table loading
* `iter_csv(path, chunksize=1000, ...)` for bounded-memory iteration
* Custom delimiters, requested or inferred dtypes, and fill/drop missing modes
* Numeric arrays for numeric tables and object arrays for mixed columns

`load_csv` returns the column names and a 2D NumPy array using
`numpy.loadtxt`. Use `iter_csv` to process data in bounded-memory chunks.

---

### `preprocessing.py`

Preprocessing components:

* `StandardScaler`
* `MinMaxScaler`
* `OneHotEncoder`
* `Imputer` for replacing NaN or configured missing-value markers

Preprocessors use:

```python
fit(X) -> self
transform(X) -> X_out
fit_transform(X) -> X_out
```

The preprocessing implementations should use NumPy vectorisation and avoid unnecessary Python loops.

---

### `sort_search.py`

Sorting and searching algorithms:

* Stable sorting
* Multi-key sorting
* `topk(values, k, largest=True, return_indices=True)`
* Top-k selection using `np.argpartition`
* Quickselect
* Binary search

Binary search returns the insertion index and whether the value exists.

---

### `rank.py`

Ranking and percentile functions.

Ranking supports:

```python
rank(data, method="average")
```

Methods:

* `average`
* `dense`
* `ordinal`

The ranking implementation handles tied values.

Percentiles support:

```python
percentile(
    data,
    q,
    interpolation="linear"
)
```

Interpolation methods:

* `linear`
* `lower`
* `higher`
* `midpoint`

---

### `stats.py`

Basic statistical functions:

* Mean
* Median
* Standard deviation
* Minimum
* Maximum
* Histogram
* Quantiles
* NaN-aware batch statistics and exact quantiles
* `StreamingStats`, which merges batches with Welford's algorithm using O(p)
  memory for p features

Functions should have clearly defined input and output dimensions when using `axis`.

---

### `metrics.py`

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

for binary classification.

Precision, recall, and F1 support binary, micro, macro, and support-weighted
averaging. Confusion-matrix rows represent true labels and columns represent
predicted labels.

---

### `optim.py`

Finite-difference numerical differentiation.

Gradient API:

```python
grad(
    f,
    x,
    h=1e-5,
    method="central"
)
```

Supported methods:

* `central`
* `forward`

Jacobian estimation is provided through:

```python
jacobian(F, x, ...)
```

---

### `pipeline.py`

Provides a minimal transformer/estimator interface.

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

No machine learning model implementation is required.

The module also provides pipeline chaining, `Compose` for sequential
transformers, and `FeatureUnion` for concatenating independent feature outputs.

Example:

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler

pipe = Pipeline([("scale", StandardScaler())])
X_tr = pipe.fit_transform(X)
```

---

### `utils.py`

Contains reusable helper functionality shared across the project.

Utilities should only be added here when they are genuinely reusable across multiple modules.

---

### `benchmarking.py`

Contains reusable micro-benchmarking functionality for measuring implementation
performance.

`benchmark(function, ..., repeats=5, warmup=1)` returns minimum, median, and
mean elapsed times in seconds plus Python, NumPy, OS/platform, processor, and
CPU-count metadata. `compare()` benchmarks named implementations with shared
arguments, and `format_table()` renders timing and environment information.

---

# Testing

Tests are located in:

```text
tests/
```

Each major module should have corresponding tests.

The test suite should cover:

* Normal cases
* Boundary cases
* Invalid inputs
* Empty arrays where applicable
* Duplicate values
* Ties
* NaN values
* Different input shapes
* Different values of `k`
* Non-contiguous arrays where applicable

The test suite contains over 100 unit tests, including invalid shapes, missing
values, ties, duplicate values, `k` boundaries, empty inputs, non-contiguous
optimization inputs, and chunked data.

After following the installation steps below, run tests from the repository root:

```bash
python -m pytest
```

Run a specific test file:

```bash
python -m pytest tests/test_optim.py
```

---

# Demo

The end-to-end demonstration is located at:

```text
demo/quickstart.ipynb
```

The demo demonstrates the implemented functionality, including:

1. CSV loading
2. Preprocessing
3. Sorting and searching
4. Ranking and percentiles
5. Statistics
6. Gradients and Jacobians
7. Pipeline usage
8. Benchmark results

The notebook should use the functions and classes from the `numcompute` package rather than reimplementing them inside notebook cells.

---

# Benchmark

The quickstart notebook compares implementations, particularly:

```text
Python loops
        vs.
NumPy vectorised implementations
```

The notebook uses a seeded 200,000-element input, one warm-up, and five timed
repeats. Its results include median and minimum seconds and the runtime
environment. Measurements depend on machine and system load; rerun to measure
your system. `format_table(compare(...))` generates current results rather than
relying on fixed timing claims.

Example local measurement (Python 3.13.7, NumPy 2.5.4, macOS 27.0.1,
arm64, 10 reported CPUs; seeded 200,000-element input, one warm-up and five
timed repeats):

| Implementation | Median (s) | Minimum (s) |
|---|---:|---:|
| Python loop | 0.00737 | 0.00692 |
| NumPy vectorised | 0.0000735 | 0.0000306 |

---

# Installation

Clone the repository and enter its root directory:

```bash
git clone <repository-url>
cd <repository-directory>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the project dependencies and test tools:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install .
python -m pip install jupyter
```

---

# Running the Demo

Install Jupyter and start it:

```bash
jupyter notebook
```

Open:

```text
demo/quickstart.ipynb
```

---

# Assignment Deliverables

The group submission consists of:

### Git Repository

The repository should contain:

```text
numcompute/
tests/
demo/quickstart.ipynb
benchmark/
README.md
pyproject.toml
```

### Demo Video

Maximum duration: **5 minutes**.

The demonstration should show:

* Setting up and running the project
* Running the demo
* Using the pipeline
* Reading outputs such as metrics, ranks, and gradients

Submit the video as an `.mp4` file or a shareable link.

### Technical Report

Maximum length: **4 pages** in PDF format.

The report should cover:

* Design choices
* Vectorisation and API design
* Testing and edge cases
* Benchmark results
* Teamwork and reflections

---

# Team

| Member   | Main Contribution |
| -------- | ----------------- |
| Member 1 | Prajal Patidar    |
| Member 2 | Manthan Lad       |
| Member 3 | Mansi Khare       |
| Member 4 | Sarthak Sharma    |
