import numpy as np

def load_csv(
    path: str,
    delimiter: str = ",",
    skip_header: int = 0,
    missing_strategy: str = "fill",
    fill_value: str = "NaN",
) -> np.ndarray:
    """Loads CSV files containing mixed numeric and categorical data into a NumPy array."""

    # ---- READ HEADER SEPARATELY ----
    with open(path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        header = header_line.split(delimiter)

    # ---- LOAD RAW DATA AS STRINGS ----
    raw_data = np.genfromtxt(
        path,
        delimiter=delimiter,
        skip_header=1,      # skip header because we already read it
        dtype=str,
        encoding="utf-8",
    )

    # Ensure 2D shape
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(-1, 1)

    # ---- MISSING VALUE HANDLING ----
    missing_mask = (
        (raw_data == "") | (raw_data == "nan") | (raw_data == fill_value)
    )

    if missing_strategy == "drop":
        valid_rows = ~missing_mask.any(axis=1)
        raw_data = raw_data[valid_rows]

    elif missing_strategy == "fill":
        raw_data[missing_mask] = str(fill_value)

    num_rows, num_cols = raw_data.shape
    converted_cols = []
    is_all_numeric = True

    # ---- COLUMN TYPE PARSING ----
    for col in range(num_cols):
        column_data = raw_data[:, col]
        try:
            float_col = column_data.astype(float)
            converted_cols.append(float_col)
        except ValueError:
            converted_cols.append(column_data)
            is_all_numeric = False

    # ---- FINAL ARRAY ----
    if is_all_numeric:
        final_data = np.column_stack(converted_cols)
    else:
        final_data = np.empty((num_rows, num_cols), dtype=object)
        for col in range(num_cols):
            final_data[:, col] = converted_cols[col]

    return header, final_data


# ---- TEST ----
header, data = load_csv(
    "C:/Users/mansi/AppData/Local/Temp/1f3f08fb-3a71-45ef-8bbb-16a0faabc060_8836201-6f9306ad21398ea43cba4f7d537619d0e07d5ae3.zip.060/8836201-6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv",
    delimiter=",",
    skip_header=0
)

# PRINT 1: Header array
print("HEADER:")
print(header)

# PRINT 2: Full dataset (your existing output)
# print("\nFULL DATA:")
# print(data)

# PRINT 3: Array of arrays (each inner array = one column)
print("\nCOLUMNS AS ARRAY OF ARRAYS:")
column_arrays = data.T        # transpose → columns become rows
print(column_arrays)

array = np.array(column_arrays)