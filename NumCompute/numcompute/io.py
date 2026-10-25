from collections.abc import Iterator
from pathlib import Path
import warnings

import numpy as np
import numpy.typing as npt


def _read_chunk(file, delimiter: str, rows: int) -> npt.NDArray[np.str_]:
    """Read up to ``rows`` records with NumPy, including quoted fields."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="loadtxt: input contained no data")
        return np.loadtxt(
            file,
            delimiter=delimiter,
            dtype=str,
            encoding="utf-8",
            # Keep characters such as "#" in application names.
            comments=None,
            quotechar='"',
            ndmin=2,
            max_rows=rows,
        )


def _convert_rows(
    raw: npt.NDArray[np.str_],
    missing_strategy: str,
    fill_value: object,
    dtype: object | None,
) -> npt.NDArray[np.generic]:
    """Fill or drop missing values, then infer or apply the output dtype."""
    missing = (raw == "") | (np.char.lower(raw) == "nan")

    if missing_strategy == "drop":
        # Drop a whole record if any column is missing, so rows stay aligned.
        raw = raw[~missing.any(axis=1)]
    else:
        raw = raw.astype(object)
        raw[missing] = str(fill_value)

    if dtype is not None:
        try:
            return raw.astype(dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"CSV values cannot be converted to dtype {dtype}."
            ) from error

    # Infer each column separately: numeric columns become floats, while names
    # and other labels stay as strings.
    columns = []
    for column in raw.T:
        try:
            columns.append(column.astype(float))
        except ValueError:
            columns.append(column)

    if all(column.dtype.kind == "f" for column in columns):
        return np.column_stack(columns)

    data = np.empty(raw.shape, dtype=object)
    for index, column in enumerate(columns):
        data[:, index] = column
    return data


def iter_csv(
    path: str | Path,
    chunksize: int = 1000,
    delimiter: str = ",",
    skip_header: int = 0,
    missing_strategy: str = "fill",
    fill_value: object = "NaN",
    dtype: object | None = None,
) -> Iterator[tuple[list[str], npt.NDArray[np.generic]]]:
    """Yield CSV chunks as ``(header, data)`` pairs."""
    if not isinstance(chunksize, (int, np.integer)) or chunksize <= 0:
        raise ValueError("chunksize must be a positive integer.")
    if len(delimiter) != 1:
        raise ValueError("delimiter must be a single character.")
    if skip_header < 0:
        raise ValueError("skip_header must be non-negative.")
    if missing_strategy not in {"fill", "drop"}:
        raise ValueError("missing_strategy must be 'fill' or 'drop'.")

    with Path(path).open("r", encoding="utf-8") as file:
        header_row = _read_chunk(file, delimiter, 1)
        if header_row.size == 0:
            raise ValueError("CSV file must contain a header row.")
        header = header_row[0].tolist()

        for _ in range(max(skip_header - 1, 0)):
            next(file, None)

        while True:
            raw = _read_chunk(file, delimiter, chunksize)
            if raw.size == 0:
                break
            data = _convert_rows(raw, missing_strategy, fill_value, dtype)
            if data.shape[0]:
                yield header, data
            # A short read means we reached the end of the file.
            if raw.shape[0] < chunksize:
                break


def load_csv(
    path: str | Path,
    delimiter: str = ",",
    skip_header: int = 0,
    missing_strategy: str = "fill",
    fill_value: object = "NaN",
    dtype: object | None = None,
) -> tuple[list[str], npt.NDArray[np.generic]]:
    """Load a CSV table with NumPy and return its header and data array."""
    chunks = list(
        iter_csv(
            path,
            delimiter=delimiter,
            skip_header=skip_header,
            missing_strategy=missing_strategy,
            fill_value=fill_value,
            dtype=dtype,
        )
    )
    if chunks:
        header = chunks[0][0]
        # Combine the chunks only for load_csv; iter_csv can stay memory-friendly.
        data = np.concatenate([chunk for _, chunk in chunks], axis=0)
        return header, data

    with Path(path).open("r", encoding="utf-8") as file:
        header_row = _read_chunk(file, delimiter, 1)
    if header_row.size == 0:
        raise ValueError("CSV file must contain a header row.")
    header = header_row[0].tolist()
    empty_dtype = np.dtype(dtype) if dtype is not None else float
    return header, np.empty((0, len(header)), dtype=empty_dtype)