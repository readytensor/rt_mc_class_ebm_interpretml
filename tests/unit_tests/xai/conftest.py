from typing import Tuple

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from prediction.predictor_model import train_predictor_model


@pytest.fixture
def transformed_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a multiclass classification dataset using sklearn's make_classification.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple with two elements.
        The first element is a DataFrame with feature values,
        and the second element is a Series with the target values.
    """
    # Create a multiclass classification dataset with 2 informative features
    features, targets = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=42,
        shuffle=True,
    )
    # Convert to pandas DataFrame and Series
    features_df = pd.DataFrame(
        features, columns=[f"feature_{i}" for i in range(1, features.shape[1] + 1)]
    )
    targets = pd.Series(targets, name="target")
    return features_df, targets


@pytest.fixture
def class_names(transformed_data):
    _, targets = transformed_data
    class_names_ = list(set(targets))
    return class_names_


@pytest.fixture
def transformed_train_and_test_data(transformed_data):
    """Fixture to create a sample transformed train DataFrame"""
    features_df, targets = transformed_data
    num_train = int(len(features_df) * 0.8)
    num_test = len(features_df) - num_train
    features_df_train = features_df.head(num_train)
    targets_train = targets.head(num_train)
    features_df_test = features_df.tail(num_test)
    targets_test = targets.tail(num_test)
    return features_df_train, targets_train, features_df_test, targets_test


@pytest.fixture
def transformed_train_inputs(transformed_train_and_test_data):
    """Get training inputs"""
    return transformed_train_and_test_data[0]


@pytest.fixture
def transformed_test_inputs(transformed_train_and_test_data):
    """Get testing inputs"""
    return transformed_train_and_test_data[2]


@pytest.fixture
def predictor(transformed_train_and_test_data, default_hyperparameters):
    # Train a simple model for testing
    features_df_train, targets_train, _, _ = transformed_train_and_test_data
    return train_predictor_model(
        features_df_train, targets_train, default_hyperparameters
    )
