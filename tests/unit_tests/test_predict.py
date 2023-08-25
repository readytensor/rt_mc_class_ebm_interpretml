import numpy as np
import pandas as pd
import pytest

from src.predict import create_predictions_dataframe


def test_create_predictions_dataframe_return_probs_true():
    """
    Test the function 'create_predictions_dataframe' with 'return_probs' set to True.
    Checks if the output is a DataFrame, if its shape and column names are correct,
    and if the ID values match the input.
    """
    np.random.seed(0)
    predictions_arr = np.random.rand(5, 2)
    class_names = ["class_1", "class_2"]
    prediction_field_name = "predicted_class"
    ids = pd.Series(np.random.choice(1000, 5))
    id_field_name = "id"
    return_probs = True

    df = create_predictions_dataframe(
        predictions_arr,
        class_names,
        prediction_field_name,
        ids,
        id_field_name,
        return_probs,
    )

    assert isinstance(df, pd.DataFrame), "Output is not a pandas DataFrame"
    assert df.shape == (5, 3), "Output shape is not correct"
    assert (
        list(df.columns) == [id_field_name] + class_names
    ), "Column names are incorrect"
    assert df[id_field_name].equals(ids), "Ids are not correct"


def test_create_predictions_dataframe_return_probs_false():
    """
    Test the function 'create_predictions_dataframe' with 'return_probs' set to False.
    Checks if the output is a DataFrame, if its shape and column names are correct,
    and if the ID values and predicted classes match the input.
    """
    np.random.seed(0)
    predictions_arr = np.random.rand(5, 3)
    class_names = ["class_1", "class_2", "class_3"]
    prediction_field_name = "predicted_class"
    ids = pd.Series(np.random.choice(1000, 5))
    id_field_name = "id"
    return_probs = False

    df = create_predictions_dataframe(
        predictions_arr,
        class_names,
        prediction_field_name,
        ids,
        id_field_name,
        return_probs,
    )

    assert isinstance(df, pd.DataFrame), "Output is not a pandas DataFrame"
    assert df.shape == (5, 2), "Output shape is not correct"
    assert list(df.columns) == [
        id_field_name,
        prediction_field_name,
    ], "Column names are incorrect"
    assert df[id_field_name].equals(ids), "Ids are not correct"
    assert all(
        df[prediction_field_name].isin(class_names)
    ), "Some predicted classes are not from the class_names"


def test_create_predictions_dataframe_mismatch_ids_and_predictions():
    """
    Test the function 'create_predictions_dataframe' for a case where the length of
    the 'ids' series doesn't match the number of rows in 'predictions_arr'.
    Expects a ValueError with a specific message.
    """
    np.random.seed(0)
    predictions_arr = np.random.rand(5, 3)
    class_names = ["class_1", "class_2", "class_3"]
    prediction_field_name = "predicted_class"
    ids = pd.Series(np.random.choice(1000, 4))  # Mismatch in size
    id_field_name = "id"
    return_probs = True

    with pytest.raises(ValueError) as exception_info:
        _ = create_predictions_dataframe(
            predictions_arr,
            class_names,
            prediction_field_name,
            ids,
            id_field_name,
            return_probs,
        )

    assert (
        str(exception_info.value)
        == "Length of ids does not match number of predictions"
    ), "Exception message does not match"


def test_create_predictions_dataframe_mismatch_class_names_and_predictions():
    """
    Test the function 'create_predictions_dataframe' for a case where the length of
    the 'class_names' list doesn't match the number of columns in 'predictions_arr'.
    Expects a ValueError with a specific message.
    """
    np.random.seed(0)
    predictions_arr = np.random.rand(5, 3)
    class_names = ["class_1", "class_2"]  # Mismatch in size
    prediction_field_name = "predicted_class"
    ids = pd.Series(np.random.choice(1000, 5))
    id_field_name = "id"
    return_probs = True

    with pytest.raises(ValueError) as exception_info:
        _ = create_predictions_dataframe(
            predictions_arr,
            class_names,
            prediction_field_name,
            ids,
            id_field_name,
            return_probs,
        )

    assert (
        str(exception_info.value)
        == "Length of class names does not match number of prediction columns"
    ), "Exception message does not match"
