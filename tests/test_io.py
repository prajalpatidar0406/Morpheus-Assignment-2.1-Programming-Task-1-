import numpy as np
import pytest

from numcompute.io import load_csv


def test_load_csv_preserves_numeric_missing_values(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("age,score\n20,4\n,5\n", encoding="utf-8")

    header, data = load_csv(str(path))

    assert header == ["age", "score"]
    assert data.shape == (2, 2)
    assert np.isnan(data[1, 0])


def test_load_csv_drops_rows_with_missing_values(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n,3\n4,5\n", encoding="utf-8")

    _, data = load_csv(str(path), missing_strategy="drop")

    np.testing.assert_array_equal(data, [[1.0, 2.0], [4.0, 5.0]])


def test_load_csv_supports_categorical_columns(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,score\nAda,3\nLin,4\n", encoding="utf-8")

    header, data = load_csv(str(path))

    assert header == ["name", "score"]
    assert data.dtype == object
    np.testing.assert_array_equal(data[:, 0], ["Ada", "Lin"])


def test_load_csv_rejects_invalid_missing_strategy(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("a\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing_strategy"):
        load_csv(str(path), missing_strategy="ignore")


def test_load_csv_supports_explicit_string_dtype_for_missing_entries(tmp_path):
    path = tmp_path / "strings.csv"
    path.write_text("name,city\nAda,\n", encoding="utf-8")

    _, data = load_csv(path, dtype=str)

    np.testing.assert_array_equal(data, [["Ada", "NaN"]])


def test_load_csv_supports_custom_delimiter(tmp_path):
    path = tmp_path / "tab-separated.tsv"
    path.write_text("age\tscore\n20\t4\n30\t5\n", encoding="utf-8")

    header, data = load_csv(path, delimiter="\t")

    assert header == ["age", "score"]
    np.testing.assert_array_equal(data, [[20.0, 4.0], [30.0, 5.0]])


def test_load_csv_keeps_hashes_in_quoted_fields(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text('name,note\n"Store #1","likes, commas"\n', encoding="utf-8")

    header, data = load_csv(path)

    assert header == ["name", "note"]
    np.testing.assert_array_equal(data, [["Store #1", "likes, commas"]])


def test_load_csv_fills_missing_values_with_placeholder(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("name,city\nAda,\n", encoding="utf-8")

    _, data = load_csv(path, dtype=str, fill_value="Unknown")

    np.testing.assert_array_equal(data, [["Ada", "Unknown"]])
