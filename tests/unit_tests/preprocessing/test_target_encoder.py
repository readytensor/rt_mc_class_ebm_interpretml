import pandas as pd
import pytest

from src.preprocessing.target_encoder import CustomTargetEncoder


def test_custom_target_encoder_valid():
    # Given
    df = pd.DataFrame({"target": ["A", "B", "A", "B"]})
    target_encoder = CustomTargetEncoder(
        target_field="target", target_classes=["B", "A"]
    )
    expected_result = pd.Series([1, 0, 1, 0], name="target")
    # When
    actual_result = target_encoder.fit_transform(df)
    # Then
    pd.testing.assert_series_equal(actual_result, expected_result)


def test_custom_target_encoder_invalid_target_field():
    """
    Test CustomTargetEncoder with an invalid target_field.
    The transformation should return None.
    """
    # Given
    df = pd.DataFrame({"target": ["A", "B", "A", "B"]})
    target_encoder = CustomTargetEncoder(
        target_field="invalid", target_classes=["A", "B"]
    )
    # When
    result = target_encoder.fit_transform(df)
    # Then
    assert result is None


def test_custom_target_encoder_invalid_target_classes():
    """
    Test CustomTargetEncoder with invalid target_classes.
    The transformation should raise a ValueError.
    """
    # Given
    df = pd.DataFrame({"target": ["A", "B", "A", "B"]})
    # When/Then
    with pytest.raises(ValueError):
        target_encoder = CustomTargetEncoder(
            target_field="target", target_classes=["C", "D"]
        )
        _ = target_encoder.fit_transform(df)


def test_custom_target_encoder_empty_dataframe():
    """
    Test CustomTargetEncoder with an empty DataFrame.
    The transformation should return an empty DataFrame.
    """
    # Given
    df = pd.DataFrame()
    target_encoder = CustomTargetEncoder(
        target_field="target", target_classes=["A", "B"]
    )
    # When
    result = target_encoder.fit_transform(df)
    # Then
    assert result is None


def test_custom_target_encoder_only_one_class():
    """
    Test CustomTargetEncoder with a DataFrame containing single class.
    Should raise ValueError.
    """
    # Given
    df = pd.DataFrame({"target": ["A", "A", "A"]})
    # When/Then
    with pytest.raises(ValueError):
        target_encoder = CustomTargetEncoder(
            target_field="target", target_classes=["A", "A"]
        )
        _ = target_encoder.fit_transform(df)
