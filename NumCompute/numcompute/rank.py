import numpy as np


class Ranker:
    """
    Rank values and calculate percentiles from one-dimensional data.

    Values are converted to numbers. Items that cannot be converted are treated
    like missing values and receive the lowest rank; percentiles ignore them.
    """

    # Convert mixed input to numbers before either ranking or calculating percentiles.
    def _safe_array(self, data):
        """
        Convert input to a 1D float array.
        Rejects multi-dimensional arrays and invalid types.
        """

        # Give common input mistakes a clearer message than NumPy's conversion error.
        if data is None:
            raise ValueError("Input data cannot be None.")

        # A string is iterable, but treating its characters as separate values is
        # almost always a mistake for these methods.
        if isinstance(data, str):
            raise ValueError(
                "Input data cannot be a single string. Provide a list or 1D array."
            )

        arr = np.asarray(data, dtype=object)

        # This class ranks one list of values at a time, not a table.
        if arr.ndim != 1:
            raise ValueError(
                f"Input data must be 1D. Received array with {arr.ndim} dimensions."
            )

        if arr.size == 0:
            raise ValueError("Input data is empty.")

        # Keep valid numbers and mark anything else as missing.
        try:
            return arr.astype(float)
        except (ValueError, TypeError):
            # Mixed object arrays need per-item conversion to preserve invalid
            # entries as NaN while keeping every convertible value.
            safe = []
            for value in arr:
                try:
                    safe.append(float(value))
                except (ValueError, TypeError):
                    safe.append(np.nan)
            return np.array(safe, dtype=float)

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

    def rank(self, data, method="average"):
        """Return each value's rank, using the selected rule for ties."""
        self._validate_rank_params(method)

        arr = self._safe_array(data)

        # Missing values sort first and get the lowest ranks, without being
        # confused with real negative-infinity values.
        nan_mask = np.isnan(arr)
        valid_idx = np.flatnonzero(~nan_mask)
        ranks = np.ones(len(arr), dtype=float)
        invalid_count = int(np.count_nonzero(nan_mask))

        if method == "ordinal":
            sorted_idx = valid_idx[np.argsort(arr[valid_idx], kind="stable")]
            order = np.concatenate((np.flatnonzero(nan_mask), sorted_idx))
            ranks[order] = np.arange(1, len(arr) + 1)

        elif method == "dense":
            unique_vals, inv = np.unique(arr[valid_idx], return_inverse=True)
            first_valid_rank = 2 if invalid_count else 1
            dense_ranks = np.arange(
                first_valid_rank, first_valid_rank + len(unique_vals)
            )
            ranks[valid_idx] = dense_ranks[inv]

        elif method == "average":
            _, inverse, counts = np.unique(
                arr[valid_idx], return_inverse=True, return_counts=True
            )
            group_starts = np.cumsum(counts) - counts
            average_ranks = group_starts + (counts + 1) / 2.0 + invalid_count
            ranks[valid_idx] = average_ranks[inverse]

        return ranks

    def percentile(self, data, q, interpolation="linear"):
        """Return the requested percentile, ignoring values that are not numeric."""
        q = self._validate_percentile_params(q, interpolation)

        arr = self._safe_array(data)

        # Invalid entries should not affect the position of a percentile.
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            raise ValueError("Input data contains no valid numeric values.")

        arr = np.sort(arr)
        n = len(arr)

        # Percentiles span from the first to last sorted value, inclusive.
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