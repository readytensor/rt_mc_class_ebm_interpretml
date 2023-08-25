import os

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.preprocess import (
    handle_class_imbalance,
    insert_nulls_in_nullable_features,
    load_pipeline_and_target_encoder,
    save_pipeline_and_target_encoder,
    train_pipeline_and_target_encoder,
    transform_data,
)


# Fixture to create a sample train dataFrame for testing
@pytest.fixture
def train_split_provider():
    """Provide a valid train_split DataFrame for testing."""
    data = pd.DataFrame(
        {
            "id": range(1, 6),
            "numeric_feature_1": [10, 20, 30, 40, 50],
            "numeric_feature_2": [1.0, -2.0, 3, -4, 5],
            "categorical_feature_1": ["A", "B", "C", "A", "B"],
            "categorical_feature_2": ["P", "Q", "R", "S", "T"],
            "target_field": ["A", "B", "A", "B", "A"],
        }
    )
    return data


# Fixture to create a sample validation dataFrame for testing
@pytest.fixture
def val_split_provider():
    """Provide a valid val_split DataFrame for testing."""
    data = pd.DataFrame(
        {
            "id": range(6, 11),
            "numeric_feature_1": [60, 70, 80, 90, 100],
            "numeric_feature_2": [-1.0, 2.0, -3, 4, -5],
            "categorical_feature_1": ["A", "B", "C", "A", "B"],
            "categorical_feature_2": ["P", "Q", "R", "S", "T"],
            "target_field": ["A", "B", "A", "B", "A"],
        }
    )
    return data


def test_train_pipeline_and_target_encoder(
    schema_provider, train_split_provider, preprocessing_config
):
    """Test the training of the pipeline and target encoder."""
    pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, preprocessing_config
    )
    assert pipeline is not None
    assert target_encoder is not None


def test_transform_data_with_train_split(
    schema_provider, train_split_provider, preprocessing_config
):
    """
    Test if train data is properly transformed using the preprocessing pipeline
    and target encoder.
    """
    preprocess_pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, preprocessing_config
    )
    transformed_inputs, transformed_targets = transform_data(
        preprocess_pipeline, target_encoder, train_split_provider
    )
    assert transformed_inputs is not None
    assert transformed_targets is not None
    assert len(transformed_inputs) == len(train_split_provider)
    assert len(transformed_targets) == len(train_split_provider)


def test_transform_data_with_valid_split(
    schema_provider, train_split_provider, preprocessing_config, val_split_provider
):
    """
    Test if validation data is properly transformed using the preprocessing pipeline and
    target encoder.
    """
    preprocess_pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, preprocessing_config
    )
    transformed_inputs, transformed_targets = transform_data(
        preprocess_pipeline, target_encoder, val_split_provider
    )
    assert transformed_inputs is not None
    assert transformed_targets is not None
    assert len(transformed_inputs) == len(val_split_provider)
    assert len(transformed_targets) == len(val_split_provider)


def test_save_and_load_pipeline_and_target_encoder(
    tmpdir, schema_provider, train_split_provider, preprocessing_config
):
    """
    Test that the trained pipeline and target encoder can be saved and loaded correctly,
    and that the transformation results before and after saving/loading are the same.
    """
    preprocess_pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, preprocessing_config
    )
    transformed_inputs, transformed_targets = transform_data(
        preprocess_pipeline, target_encoder, train_split_provider
    )
    preprecessing_dir = os.path.join(tmpdir, "preprocessing")
    save_pipeline_and_target_encoder(
        preprocess_pipeline, target_encoder, preprecessing_dir
    )
    (
        loaded_preprocess_pipeline,
        loaded_target_encoder,
    ) = load_pipeline_and_target_encoder(preprecessing_dir)
    assert loaded_preprocess_pipeline is not None
    assert loaded_target_encoder is not None
    transformed_inputs_2, transformed_targets_2 = transform_data(
        loaded_preprocess_pipeline, loaded_target_encoder, train_split_provider
    )
    assert transformed_inputs.equals(transformed_inputs_2)
    assert transformed_targets.equals(transformed_targets_2)


def test_handle_class_imbalance(
    schema_provider, train_split_provider, preprocessing_config
):
    """
    Test if class imbalance is properly handled using SMOTE.
    Also ensures that there is more than one class after balancing,
    and that the counts of each class are approximately equal.
    """
    preprocess_pipeline, target_encoder = train_pipeline_and_target_encoder(
        schema_provider, train_split_provider, preprocessing_config
    )
    transformed_inputs, transformed_targets = transform_data(
        preprocess_pipeline, target_encoder, train_split_provider
    )

    balanced_data, balanced_labels = handle_class_imbalance(
        transformed_inputs, transformed_targets
    )
    assert balanced_data is not None
    assert balanced_labels is not None
    assert (
        len(np.unique(balanced_labels)) > 1
    )  # Ensure there is more than one class after balancing

    # Ensure that the counts of each class are approximately equal
    class_counts = pd.Series(balanced_labels).value_counts()
    assert max(class_counts) - min(class_counts) <= 1


def test_insert_nulls_in_nullable_features(schema_provider, preprocessing_config):
    """
    Tests the function `insert_nulls_in_nullable_features` to ensure that it correctly
    inserts nulls into nullable columns that don't already contain any nulls. This
    function will generate a DataFrame with no null values and will verify that after
    the function call, the nullable columns in the DataFrame contain at least one null
    value.
    """

    # Set a seed to ensure reproducibility of random operations
    np.random.seed(0)

    # Create a DataFrame - no nulls in `numeric_feature_1` and
    # `categorical_feature_1` which are nullable
    train_data = pd.DataFrame(
        {
            "id": range(1, 6),
            "numeric_feature_1": [10, 20, 30, 40, 50],
            "numeric_feature_2": [1.0, -2.0, 3, -4, 5],
            "categorical_feature_1": ["A", "B", "C", "A", "B"],
            "categorical_feature_2": ["P", "Q", "R", "S", "T"],
            "target_field": ["A", "B", "A", "B", "A"],
        }
    )

    # Ensure there were no nulls before the function call
    assert (
        train_data.isnull().sum().sum() == 0
    ), "Input DataFrame already contains null values."

    # Call the function and get the result
    df = insert_nulls_in_nullable_features(
        train_data, schema_provider, preprocessing_config
    )

    # Get nullable features from the schema
    nullable_features = schema_provider.nullable_features

    # Ensure that the nulls were only inserted in the nullable features
    null_columns = df.columns[df.isna().any()].tolist()
    assert set(null_columns) == set(
        nullable_features
    ), "Nulls were inserted in non-nullable columns"

    # Check if at least one null was added in each nullable feature
    for col in nullable_features:
        assert df[col].isnull().sum() > 0, f"No nulls were inserted in column {col}"

    # Check if non-nullable features do not contain nulls
    non_nullable_features = schema_provider.non_nullable_features
    for col in non_nullable_features:
        assert (
            df[col].isnull().sum() == 0
        ), f"Nulls were inserted in non-nullable column {col}"


def test_insert_nulls_in_nullable_features_no_insert(
    schema_provider, preprocessing_config
):
    """
    Tests the function `insert_nulls_in_nullable_features` to ensure that it correctly
    handles nullable columns that already contain nulls. This function will generate a
    DataFrame with null values in some nullable features, then it verifies that after
    the function call, no new nulls were inserted into these columns.
    """

    # Set a seed to ensure reproducibility of random operations
    np.random.seed(0)

    # Create a DataFrame
    # Nullable feature `numeric_feature_1` contains 1 null
    # Nullable feature `categorical_feature_1` contains 2 nulls
    train_data = pd.DataFrame(
        {
            "id": range(1, 6),
            "numeric_feature_1": [10, 20, np.nan, 40, 50],
            "numeric_feature_2": [1.0, -2.0, 3, -4, 5],
            "categorical_feature_1": ["A", np.nan, "C", "A", np.nan],
            "categorical_feature_2": ["P", "Q", "R", "S", "T"],
            "target_field": ["A", "B", "A", "B", "A"],
        }
    )

    # Record the number of nulls in the DataFrame
    original_null_count = train_data.isnull().sum()

    # Call the function and get the result
    df = insert_nulls_in_nullable_features(
        train_data, schema_provider, preprocessing_config
    )

    # Record the number of nulls in the resulting DataFrame
    new_null_count = df.isnull().sum()

    # Check that no new nulls were added to the columns that already had nulls
    assert original_null_count.equals(
        new_null_count
    ), "New nulls were inserted into columns that already contained nulls"
