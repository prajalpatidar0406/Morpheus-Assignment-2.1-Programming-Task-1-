import numpy as np

class Ranker:
    """
    Ranking + percentile utilities with strict validation similar to io.py.
    All data is converted to float; invalid values become NaN.
    Only 1D input arrays/lists are allowed.
    """

    # ----------------------------------------------------
    # INTERNAL SAFE CASTING (STRICT 1D VALIDATION)
    # ----------------------------------------------------
    def _safe_array(self, data):
        """
        Convert input to a 1D float array.
        Rejects multi-dimensional arrays and invalid types.
        """

        # Reject None
        if data is None:
            raise ValueError("Input data cannot be None.")

        # Reject single string
        if isinstance(data, str):
            raise ValueError("Input data cannot be a single string. Provide a list or 1D array.")

        # Convert to numpy array
        arr = np.asarray(data, dtype=object)

        # Reject multi-dimensional arrays
        if arr.ndim != 1:
            raise ValueError(
                f"Input data must be 1D. Received array with {arr.ndim} dimensions."
            )

        # Reject empty arrays
        if arr.size == 0:
            raise ValueError("Input data is empty.")

        # Convert each element to float safely
        safe = []
        for x in arr:
            try:
                safe.append(float(x))
            except (ValueError, TypeError):
                safe.append(np.nan)

        return np.array(safe, dtype=float)

    # ----------------------------------------------------
    # PARAMETER VALIDATION (io.py-style)
    # ----------------------------------------------------
    def _validate_rank_params(self, method):
        valid_methods = {"average", "dense", "ordinal"}
        if method not in valid_methods:
            raise ValueError(
                f"Invalid ranking method '{method}'. Must be one of {valid_methods}."
            )

    def _validate_percentile_params(self, q, interpolation):
        valid_modes = {"linear", "lower", "higher", "midpoint"}

        if interpolation not in valid_modes:
            raise ValueError(
                f"Invalid interpolation '{interpolation}'. Must be one of {valid_modes}."
            )

        try:
            q = float(q)
        except Exception:
            raise ValueError("Percentile 'q' must be numeric.")

        if not (0 <= q <= 100):
            raise ValueError("Percentile 'q' must be between 0 and 100.")

        return q

    # ----------------------------------------------------
    # RANK FUNCTION (VALIDATED)
    # ----------------------------------------------------
    def rank(self, data, method="average"):
        self._validate_rank_params(method)

        arr = self._safe_array(data)

        # Replace NaN with sentinel for sorting
        nan_mask = np.isnan(arr)
        arr[nan_mask] = -np.inf

        sorted_idx = np.argsort(arr)
        ranks = np.empty_like(sorted_idx, dtype=float)

        if method == "ordinal":
            ranks[sorted_idx] = np.arange(1, len(arr) + 1)

        elif method == "dense":
            unique_vals, inv = np.unique(arr, return_inverse=True)
            dense_ranks = np.arange(1, len(unique_vals) + 1)
            ranks = dense_ranks[inv]

        elif method == "average":
            sorted_data = arr[sorted_idx]
            i = 0
            while i < len(arr):
                j = i
                while j + 1 < len(arr) and sorted_data[j + 1] == sorted_data[i]:
                    j += 1
                avg_rank = (i + j + 2) / 2.0
                ranks[sorted_idx[i:j+1]] = avg_rank
                i = j + 1

        return ranks

    # ----------------------------------------------------
    # PERCENTILE FUNCTION (VALIDATED)
    # ----------------------------------------------------
    def percentile(self, data, q, interpolation="linear"):
        q = self._validate_percentile_params(q, interpolation)

        arr = self._safe_array(data)

        # Remove NaN
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            raise ValueError("Input data contains no valid numeric values.")

        arr = np.sort(arr)
        n = len(arr)

        pos = (q / 100) * (n - 1)
        lower = int(np.floor(pos))
        upper = int(np.ceil(pos))

        if interpolation == "lower":
            return arr[lower]

        elif interpolation == "higher":
            return arr[upper]

        elif interpolation == "midpoint":
            return (arr[lower] + arr[upper]) / 2.0

        elif interpolation == "linear":
            if lower == upper:
                return arr[lower]
            weight = pos - lower
            return arr[lower] * (1 - weight) + arr[upper] * weight
