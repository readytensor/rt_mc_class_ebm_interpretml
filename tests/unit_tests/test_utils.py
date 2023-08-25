import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    read_csv_in_directory,
    read_json_as_dict,
    save_dataframe_as_csv,
    set_seeds,
    split_train_val,
)


def test_read_json_as_dict_with_file_path():
    """
    Test if `read_json_as_dict` function can correctly read a JSON file
    and return its content as a dictionary when given a file path.
    """
    # Given
    input_dict = {"key": "value"}
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "test.json")

    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
        json.dump(input_dict, temp_file)

    # When
    result_dict = read_json_as_dict(temp_file_path)

    # Then
    assert result_dict == input_dict

    # Cleanup
    shutil.rmtree(temp_dir)


def test_read_json_as_dict_with_dir_path():
    """
    Test if `read_json_as_dict` function can correctly find a JSON file in a directory,
    read the file, and return its content as a dictionary when given a directory path.
    """
    # Given
    input_dict = {"key": "value"}
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, "test.json")

    with open(temp_file_path, "w", encoding="utf-8") as temp_file:
        json.dump(input_dict, temp_file)

    # When
    result_dict = read_json_as_dict(temp_dir)

    # Then
    assert result_dict == input_dict

    # Cleanup
    shutil.rmtree(temp_dir)


def test_read_json_as_dict_with_invalid_path():
    """
    Test if `read_json_as_dict` function correctly raises a ValueError when given
    an invalid path.
    """
    # Given
    invalid_path = "/invalid/path"

    # When/Then
    with pytest.raises(ValueError):
        read_json_as_dict(invalid_path)


def test_read_json_as_dict_no_json_file(tmpdir):
    """
    Test the read_json_as_dict function for a directory that does not contain any
    JSON file. The function should raise an exception.
    """
    # Given: A temporary directory without any JSON file
    directory_path = tmpdir.mkdir("subdir")

    # When: read_json_as_dict is called with the directory path
    # Then: A ValueError should be raised
    with pytest.raises(ValueError):
        read_json_as_dict(directory_path)


def test_read_json_as_dict_invalid_json_file(tmpdir):
    """
    Test the read_json_as_dict function with an invalid JSON file.
    """
    # Given: Create a file with invalid JSON content
    invalid_json_file = tmpdir.join("invalid.json")
    invalid_json_file.write("this is not valid JSON content")

    # When & Then: Attempt to read the file and assert it raises a JSONDecodeError
    with pytest.raises(json.JSONDecodeError):
        read_json_as_dict(invalid_json_file.strpath)


def test_read_csv_in_directory(tmpdir):
    """
    Test the function read_csv_in_directory with valid and invalid inputs.
    """
    # Valid case: Create a CSV file in a temporary directory
    file_path = tmpdir.mkdir("sub").join("test.csv")
    df_test = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df_test.to_csv(file_path, index=False)
    df_read = read_csv_in_directory(str(tmpdir.join("sub")))
    pd.testing.assert_frame_equal(df_read, df_test)

    # Invalid case: Directory does not exist
    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        read_csv_in_directory(r"C:\nonexistent_directory")

    # Invalid case: Directory exists but no CSV files
    empty_dir_path = tmpdir.mkdir("empty_sub")
    with pytest.raises(ValueError, match="No CSV file found in directory"):
        read_csv_in_directory(str(empty_dir_path))

    # Invalid case: Multiple CSV files in the directory
    sub_dir = tmpdir.mkdir("sub2")
    file_path_1 = sub_dir.join("test1.csv")
    file_path_2 = sub_dir.join("test2.csv")
    df_test.to_csv(file_path_1, index=False)
    df_test.to_csv(file_path_2, index=False)
    with pytest.raises(ValueError, match="Multiple CSV files found in directory"):
        read_csv_in_directory(str(sub_dir))


def test_split_train_val():
    """
    Test the function split_train_val with valid inputs.
    Ensures that the training and validation sets are split correctly.
    """
    df = pd.DataFrame(np.random.randint(0, 100, size=(100, 4)), columns=list("ABCD"))
    df["target"] = np.random.randint(0, 2, size=(100,))  # Add a target column
    train_data, val_data = split_train_val(df, "target", 0.2)

    # Verify that the total number of rows equals the original number of rows
    assert len(df) == len(train_data) + len(val_data)

    # Verify that the percentage split is correct
    assert len(val_data) / len(df) == pytest.approx(0.2, 0.01)

    # Verify that all the original columns are present in the training and
    # validation sets
    assert list(df.columns) == list(train_data.columns)
    assert list(df.columns) == list(val_data.columns)


def test_set_seeds():
    """
    Test the function set_seeds with valid and invalid inputs and check reproducibility.
    """
    # Test valid integer case
    try:
        set_seeds(42)
        data = pd.DataFrame(np.random.randn(10, 5))
        data["target"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Manually set target column
        train_data_1, val_data_1 = split_train_val(data, "target", 0.2)

        set_seeds(42)
        data = pd.DataFrame(np.random.randn(10, 5))
        data["target"] = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]  # Manually set target column
        train_data_2, val_data_2 = split_train_val(data, "target", 0.2)

        pd.testing.assert_frame_equal(train_data_1, train_data_2)
        pd.testing.assert_frame_equal(val_data_1, val_data_2)
    except Exception as e:
        pytest.fail(f"set_seeds(42) raised exception: {e}")

    # Test invalid cases
    with pytest.raises(Exception, match="Invalid seed value"):
        set_seeds(42.0)
    with pytest.raises(Exception, match="Invalid seed value"):
        set_seeds("Invalid")


def test_save_dataframe_as_csv():
    """
    Test that the 'save_dataframe_as_csv' function correctly saves a DataFrame
    to disk as a CSV file, with the correct formatting for float values.
    """
    # Create a sample DataFrame
    df = pd.DataFrame(
        {
            "A": [1, 2, 3],
            "B": [4.123456, 5.123456, 6.123456],
            "C": ["foo", "bar", "baz"],
        }
    )

    # Create a temporary file path
    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, "test.csv")

    # Save the DataFrame to CSV
    save_dataframe_as_csv(df, file_path)

    # Read the CSV file back into a DataFrame
    df_loaded = pd.read_csv(file_path)

    # Check that the loaded DataFrame is equal to the original DataFrame
    pd.testing.assert_frame_equal(df, df_loaded, atol=1e-4)

    # Check that float values are saved with correct precision
    assert df_loaded["B"].apply(lambda x: len(str(x).split(".")[1])).max() <= 4
