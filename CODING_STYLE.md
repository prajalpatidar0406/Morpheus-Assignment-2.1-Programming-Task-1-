# NumCompute — Team Coding Guidelines

This document defines the conventions the team will use when developing NumCompute.

The purpose is to keep implementations consistent when different team members work on different modules.

---

## 1. General Principles

All code should be:

* Correct
* Readable
* Consistent
* Testable
* Appropriately documented

Follow the existing project conventions before introducing a new style or API.

If a change affects another module, discuss it with the relevant team member before merging.

---

## 2. Naming

Use standard Python naming conventions.

### Functions and variables

Use `snake_case`:

```python
def binary_search(sorted_array, x):
    ...


input_array = ...
```

### Classes

Use `PascalCase`:

```python
class StandardScaler:
    ...
```

### Constants

Use `UPPER_CASE`:

```python
DEFAULT_TOLERANCE = 1e-8
```

Use descriptive names rather than unclear abbreviations.

---

## 3. Imports

Use:

```python
import numpy as np
```

Keep imports at the beginning of the file.

Remove unused imports.

Do not use wildcard imports:

```python
from numpy import *
```

---

## 4. NumPy and Vectorisation

Use NumPy operations for numerical computation whenever practical.

Prefer:

```python
mean = np.mean(X, axis=0)
```

over manually iterating through the array.

Avoid unnecessary Python loops in core numerical computations.

If a loop is required by the algorithm, it is acceptable.

For example, an educational quickselect implementation may require iterative partitioning.

---

## 5. Input Validation

Validate inputs where invalid data could produce incorrect or confusing results.

Example:

```python
if k < 1 or k > len(values):
    raise ValueError(
        "k must be between 1 and the number of values"
    )
```

Error messages should state what was expected.

Prefer:

```text
ValueError: X must be a 2D array.
Received shape (10,).
```

over:

```text
ValueError: Invalid input.
```

---

## 6. Array Shapes

Use consistent terminology:

```text
n_samples  = number of observations
n_features = number of features
```

For tabular data:

```text
X: (n_samples, n_features)
```

For target values:

```text
y: (n_samples,)
```

Functions using `axis` must clearly define their output shape.

Do not silently reshape inputs unless the function's API explicitly requires it.

---

## 7. Public API Consistency

Use the assignment's specified function names and parameters.

Examples:

```python
topk(values, k, largest=True, return_indices=True)
```

```python
binary_search(sorted_array, x)
```

```python
rank(data, method="average")
```

```python
percentile(data, q, interpolation="linear")
```

```python
grad(f, x, h=1e-5, method="central")
```

Do not rename or change the parameters of shared functions without team agreement.

---

## 8. Class Interfaces

Preprocessing classes should follow the agreed interface:

```python
fit(X)
transform(X)
fit_transform(X)
```

`fit()` should return `self`.

Example:

```python
def fit(self, X):
    ...
    return self
```

Pipeline components should follow the interfaces defined in `pipeline.py`.

---

## 9. Docstrings

Every public function and class must have a docstring.

Use a consistent structure:

```python
def mse(y_true, y_pred):
    """
    Calculate mean squared error.

    Parameters
    ----------
    y_true : np.ndarray
        True values with shape (n_samples,).

    y_pred : np.ndarray
        Predicted values with shape (n_samples,).

    Returns
    -------
    float
        Mean squared error.

    Raises
    ------
    ValueError
        If the input arrays have different shapes.

    Complexity
    ----------
    Time: O(n)
    Space: O(1)
    """
```

Document:

* Parameters
* Return values
* Shapes where relevant
* Exceptions
* Time complexity
* Space complexity

---

## 10. Numerical Behaviour

Consider numerical edge cases when implementing algorithms.

Pay attention to:

* NaN values
* Infinite values
* Very large values
* Very small values
* Overflow
* Underflow
* Division by zero
* Empty inputs where applicable
* Duplicate values
* Ties

Use numerically stable approaches where appropriate.

Do not introduce arbitrary numerical constants without understanding their effect.

---

## 11. NaN Handling

NaN behaviour should be deliberate and consistent with the function's purpose.

For functions that support NaN handling, document whether NaNs are:

* Ignored
* Propagated
* Replaced

Do not silently change NaN behaviour between similar functions.

---

## 12. Avoid Code Duplication

Before creating a helper, check whether the functionality already exists.

Reusable helpers may be placed in:

```text
numcompute/utils.py
```

Only place functionality in `utils.py` when it is useful to multiple parts of the project.

Module-specific algorithms should remain in their relevant module.

---

## 13. Comments

Comments should explain reasoning that is not obvious from the code.

Avoid:

```python
# Add one
x = x + 1
```

Prefer:

```python
# Add epsilon to avoid division by zero for constant input.
result = denominator + epsilon
```

Do not add comments to every line.

---

## 14. Testing

Every implemented feature should have corresponding tests.

Tests should include:

### Normal cases

```python
def test_mean():
    ...
```

### Edge cases

```python
def test_rank_with_ties():
    ...


def test_topk_k_equals_one():
    ...
```

### Invalid inputs

```python
def test_topk_invalid_k():
    ...
```

Tests should verify expected results, not simply whether the function executes.

---

## 15. Test Organisation

Keep tests aligned with their implementation module.

```text
numcompute/io.py
        ↓
tests/test_io.py

numcompute/preprocessing.py
        ↓
tests/test_preprocessing.py

numcompute/sort_search.py
        ↓
tests/test_sort_search.py
```

Keep related tests together.

The final project must contain at least **20 meaningful unit tests**.

---

## 16. Git Branches

Use feature branches instead of developing directly on `main`.

Example:

```text
main
│
├── feature/io
├── feature/preprocessing
├── feature/sort-search
├── feature/rank
├── feature/stats
├── feature/metrics
├── feature/optim
└── feature/pipeline
```

Create a branch with:

```bash
git checkout -b feature/preprocessing
```

Choose a branch name that clearly describes the work.

---

## 17. Commits

Keep commits focused and descriptive.

Good examples:

```text
Add StandardScaler implementation
Implement topk using argpartition
Add rank tie handling
Add metrics tests
Fix binary search insertion index
```

Avoid vague messages such as:

```text
update
changes
stuff
final
```

---

## 18. Pull Requests

Before merging:

```text
[ ] Implementation works
[ ] Tests pass
[ ] New functionality has tests
[ ] No debug code remains
[ ] No unnecessary files are included
[ ] Public API has not been changed unexpectedly
[ ] Documentation is updated where necessary
```

Run:

```bash
pytest
```

before merging.

---

## 19. Shared Interfaces

Some functions and classes will be used by multiple team members.

Before changing a shared interface:

1. Check where it is used.
2. Check the existing tests.
3. Discuss the change with the team.
4. Update dependent code if the change is approved.
5. Add or update tests.

Do not change a shared API just for personal preference.

---

## 20. Notebook Rules

`demo/quickstart.ipynb` is for demonstrating the package.

Do not implement the actual project algorithms inside the notebook.

Import functionality from:

```text
numcompute/
```

Keep the notebook organised and remove unnecessary experimental cells before submission.

---

## 21. Benchmark Rules

Benchmark implementations should be compared fairly.

Where applicable, compare:

```text
Loop implementation
        vs.
Vectorised NumPy implementation
```

Record the conditions needed to understand the result, such as:

* Input size
* Number of repetitions
* Execution time
* Python version
* NumPy version

Do not modify an implementation solely to make its benchmark appear better.

---

## 22. Before Pushing

Run the test suite:

```bash
pytest
```

Then check:

```text
[ ] Tests pass
[ ] No debug print statements
[ ] No unused imports
[ ] No temporary files
[ ] No virtual environment committed
[ ] No unnecessary changes
[ ] Public API is consistent
[ ] Commit message is descriptive
```

Then push the branch:

```bash
git push -u origin <branch-name>
```

---

## 23. Team Consistency

When two implementations satisfy the requirements, prefer the approach that is:

1. Consistent with existing code
2. Easy to understand
3. Easy to test
4. Appropriately efficient

The team should prioritise:

```text
Assignment Requirements
        ↓
Correctness
        ↓
API Consistency
        ↓
Readable Code
        ↓
Appropriate Performance
```

The final codebase should look like one project rather than separate pieces written independently.
