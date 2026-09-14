# NumCompute

NumCompute is a numerical computing toolkit implemented using **plain Python and NumPy**.

The project implements numerical data loading, preprocessing, sorting and searching, ranking, statistics, evaluation metrics, finite-difference differentiation, and pipeline functionality.

External machine learning/deep learning libraries such as **scikit-learn, pandas, and PyTorch are not used**.

---

## Project Structure

```text
NumCompute/
│
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
│
├── tests/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_sort_search.py
│   ├── test_rank.py
│   ├── test_stats.py
│   ├── test_metrics.py
│   ├── test_optim.py
│   └── test_pipeline.py
│
├── benchmark/
│   └── ...
│
├── demo/
│   └── quickstart.ipynb
│
├── README.md
├── CODING_STYLE.md
├── pyproject.toml
└── .gitignore
```

---

## Requirements

* Python
* NumPy
* Plain Python standard library where required

The assignment does not permit external ML/DL libraries such as:

* scikit-learn
* pandas
* PyTorch

---

## Modules

### `io.py`

CSV data loading using NumPy.

Supports:

* CSV loading with `numpy.loadtxt` or `numpy.genfromtxt`
* Custom delimiters, such as comma or tab
* Missing-value handling
* Returning data as NumPy arrays

---

### `preprocessing.py`

Preprocessing components:

* `StandardScaler`
* `MinMaxScaler`
* `OneHotEncoder`
* Optional simple imputer for replacing NaN values with a constant

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
* NaN handling
* Axis-wise statistics

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

The module also provides pipeline chaining.

Example:

```python
from numcompute.pipeline import Pipeline
from numcompute.preprocessing import StandardScaler, OneHotEncoder

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("encode", OneHotEncoder())
])

X_tr = pipe.fit_transform(X)
```

---

### `utils.py`

Contains reusable helper functionality shared across the project.

Utilities should only be added here when they are genuinely reusable across multiple modules.

---

### `benchmarking.py`

Contains reusable benchmarking functionality for measuring implementation performance.

The benchmark directory contains the actual benchmark scripts.

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

The assignment requires **at least 20 unit tests**.

Run all tests:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run one test file:

```bash
pytest tests/test_stats.py
```

---

# Demo

The end-to-end demonstration is located at:

```text
demo/quickstart.ipynb
```

The demo should demonstrate the implemented functionality, including:

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

Benchmark scripts are located in:

```text
benchmark/
```

The benchmarks should compare relevant implementations, particularly:

```text
Python loops
        vs.
NumPy vectorised implementations
```

Benchmark results should include sufficient information to reproduce and interpret the comparison.

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd NumCompute
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

Install the project:

```bash
pip install -e .
```

Install testing and notebook dependencies:

```bash
pip install pytest jupyter
```

---

# Running the Demo

Start Jupyter:

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
