import numpy as np
import pandas as pd
import pytest
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer
from sklearn.preprocessing import StandardScaler

from src.preprocessing.custom_transformers import (
    ColumnSelector,
    MostFrequentImputer,
    OneHotEncoderMultipleCols,
    TransformerWrapper,
    TypeCaster,
    ValueClipper,
)


# ColumnSelector tests
def test_column_selector_invalid_selector_type():
    """
    Test ColumnSelector transformer with an invalid selector_type.
    It should raise an AssertionError.
    """
    with pytest.raises(AssertionError):
        ColumnSelector(columns=["col1", "col2"], selector_type="invalid")


def test_column_selector_keep_valid_columns():
    """
    Test ColumnSelector transformer with selector_type='keep' and valid columns.
    It should only keep the specified columns in the output DataFrame.
    """
    data = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    column_selector = ColumnSelector(columns=["A", "C"], selector_type="keep")
    result = column_selector.fit_transform(data)
    expected_result = pd.DataFrame({"A": [1, 2], "C": [5, 6]})
    pd.testing.assert_frame_equal(result, expected_result)


def test_column_selector_keep_invalid_columns():
    """
    Test ColumnSelector transformer with selector_type='keep' and invalid columns.
    It should return an empty DataFrame.
    """
    data = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    column_selector = ColumnSelector(columns=["X", "Y"], selector_type="keep")
    result = column_selector.fit_transform(data)
    expected_result = pd.DataFrame(index=[0, 1], columns=[], dtype="object")
    pd.testing.assert_frame_equal(result, expected_result)


def test_column_selector_drop_valid_columns():
    """
    Test ColumnSelector transformer with selector_type='drop' and valid columns.
    It should drop the specified columns from the output DataFrame.
    """
    data = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    column_selector = ColumnSelector(columns=["A", "C"], selector_type="drop")
    result = column_selector.fit_transform(data)
    expected_result = pd.DataFrame({"B": [3, 4]})
    pd.testing.assert_frame_equal(result, expected_result)


def test_column_selector_drop_invalid_columns():
    """
    Test ColumnSelector transformer with selector_type='drop' and invalid columns.
    It should return the dataframe unchanged.
    """
    data = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5, 6]})
    column_selector = ColumnSelector(columns=["X", "Y"], selector_type="drop")
    result = column_selector.fit_transform(data)
    expected_result = data.copy()
    pd.testing.assert_frame_equal(result, expected_result)


def test_column_selector_empty_dataframe():
    """
    Test ColumnSelector transformer with selector_type='keep' and an empty dataframe.
    It should return an empty DataFrame.
    """
    data = pd.DataFrame()
    column_selector = ColumnSelector(columns=["A", "B"], selector_type="keep")
    result = column_selector.fit_transform(data)
    expected_result = pd.DataFrame(index=[], columns=[], dtype="object")
    pd.testing.assert_frame_equal(result, expected_result)


# TypeCaster tests
def test_type_caster_int_valid_columns():
    """
    Test TypeCaster transformer with cast_type=int.
    It should cast the specified columns to int type in the output DataFrame.
    """
    data = pd.DataFrame({"A": ["1", "2"], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    type_caster = TypeCaster(vars=["A", "B"], cast_type=int)
    result = type_caster.fit_transform(data)
    expected_result = pd.DataFrame({"A": [1, 2], "B": [3, 4], "C": [5.0, 6.0]})
    pd.testing.assert_frame_equal(result, expected_result)


def test_type_caster_int_invalid_columns():
    """
    Test TypeCaster transformer with cast_type=int and invalid columns to cast.
    It return the dataframe unchanged.
    """
    data = pd.DataFrame({"A": ["1", "2"], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    type_caster = TypeCaster(vars=["X", "Y"], cast_type=int)
    result = type_caster.fit_transform(data)
    expected_result = data.copy()
    pd.testing.assert_frame_equal(result, expected_result)


def test_type_caster_string_casting():
    """
    Test TypeCaster transformer with cast_type=str and valid columns to cast.
    It should return the dataframe with columns updated to string type.
    """
    data = pd.DataFrame({"A": [1, 2], "B": [3.0, 4.0], "C": [5.0, 6.0]})
    type_caster = TypeCaster(vars=["A", "B"], cast_type=str)
    result = type_caster.fit_transform(data)
    expected_result = pd.DataFrame(
        {"A": ["1", "2"], "B": ["3.0", "4.0"], "C": [5.0, 6.0]}
    )
    pd.testing.assert_frame_equal(result, expected_result)


def test_type_caster_mix_valid_invalid_columns():
    """
    Test TypeCaster transformer with a mix of valid and invalid column names.
    It should cast only the valid columns and ignore the invalid ones.
    """
    df = pd.DataFrame({"col1": [1, 2], "col2": ["3", "4"], "col3": [5, 6]})
    tc = TypeCaster(vars=["col1", "invalid"], cast_type=int)
    result = tc.transform(df)
    expected = pd.DataFrame({"col1": [1, 2], "col2": ["3", "4"], "col3": [5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_type_caster_empty_dataframe():
    """
    Test TypeCaster transformer with an empty DataFrame.
    It should return the empty DataFrame unchanged.
    """
    df = pd.DataFrame()
    tc = TypeCaster(vars=["col1"], cast_type=int)
    result = tc.transform(df)
    expected = pd.DataFrame()
    pd.testing.assert_frame_equal(result, expected)


def test_type_caster_all_null_values():
    """
    Test TypeCaster transformer with all null values in specified columns.
    It should return the DataFrame unchanged.
    """
    df = pd.DataFrame({"col1": [None, None], "col2": ["3", "4"]})
    tc = TypeCaster(vars=["col1"], cast_type=int)
    result = tc.transform(df)
    expected = pd.DataFrame({"col1": [None, None], "col2": ["3", "4"]})
    pd.testing.assert_frame_equal(result, expected)


def test_type_caster_mix_null_non_null_values():
    """
    Test TypeCaster transformer with columns containing a mix of null and
    non-null values.
    It should cast the non-null values in the specified columns.
    """
    df = pd.DataFrame({"col1": [1, None], "col2": ["3", "4"]})
    tc = TypeCaster(vars=["col1"], cast_type=float)
    result = tc.transform(df)
    expected = pd.DataFrame({"col1": [1, None], "col2": ["3", "4"]})
    pd.testing.assert_frame_equal(result, expected)


# ValueClipper tests
def test_value_clipper_valid_fields_range():
    """
    Test ValueClipper transformer with valid field names and a valid range.
    It should clip the values in the specified columns within the given range.
    """
    df = pd.DataFrame({"col1": [1, 2, 5], "col2": [6, 7, 8], "col3": [3, 4, 9]})
    vc = ValueClipper(fields_to_clip=["col1", "col2"], min_val=2, max_val=7)
    result = vc.transform(df)
    expected = pd.DataFrame({"col1": [2, 2, 5], "col2": [6, 7, 7], "col3": [3, 4, 9]})
    pd.testing.assert_frame_equal(result, expected)


def test_value_clipper_invalid_fields():
    """
    Test ValueClipper transformer with invalid field names.
    It should not change the input DataFrame.
    """
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    vc = ValueClipper(fields_to_clip=["invalid"], min_val=2, max_val=5)
    result = vc.transform(df)
    expected = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_value_clipper_mix_valid_invalid_fields():
    """
    Test ValueClipper transformer with a mix of valid and invalid field names.
    It should clip the values only in the valid fields and ignore the invalid ones.
    """
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    vc = ValueClipper(fields_to_clip=["col1", "invalid"], min_val=2, max_val=5)
    result = vc.transform(df)
    expected = pd.DataFrame({"col1": [2, 2, 3], "col2": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_value_clipper_only_lower_bound():
    """
    Test ValueClipper transformer with only a lower bound specified.
    It should clip the values only from the lower end.
    """
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    vc = ValueClipper(fields_to_clip=["col1", "col2"], min_val=2, max_val=None)
    result = vc.transform(df)
    expected = pd.DataFrame({"col1": [2, 2, 3], "col2": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_value_clipper_only_upper_bound():
    """
    Test ValueClipper transformer with only an upper bound specified.
    It should clip the values only from the upper end.
    """
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    vc = ValueClipper(fields_to_clip=["col1", "col2"], min_val=None, max_val=5)
    result = vc.transform(df)
    expected = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 5]})
    pd.testing.assert_frame_equal(result, expected)


def test_value_clipper_no_bounds():
    """
    Test ValueClipper transformer with no bounds specified.
    It should not change the input DataFrame.
    """
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    vc = ValueClipper(fields_to_clip=["col1", "col2"], min_val=None, max_val=None)
    result = vc.transform(df)
    expected = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    pd.testing.assert_frame_equal(result, expected)


def test_value_clipper_empty_dataframe():
    """
    Test ValueClipper transformer with an empty DataFrame.
    It should not change the input DataFrame.
    """
    df = pd.DataFrame()
    vc = ValueClipper(fields_to_clip=["col1", "col2"], min_val=2, max_val=5)
    result = vc.transform(df)
    expected = pd.DataFrame()
    pd.testing.assert_frame_equal(result, expected)


# MostFrequentImputer Tests
def test_most_frequent_imputer_valid_cat_vars_threshold():
    """
    Test MostFrequentImputer with valid categorical variables and different thresholds.
    It should impute the missing values with the most frequent class in the specified
    columns.
    """
    df = pd.DataFrame(
        {"col1": ["a", "a", np.nan, "b"], "col2": ["x", np.nan, np.nan, np.nan]}
    )
    mfi = MostFrequentImputer(cat_vars=["col1", "col2"], threshold=0.5)
    result = mfi.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": ["a", "a", "a", "b"], "col2": ["x", np.nan, np.nan, np.nan]}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_most_frequent_imputer_invalid_cat_vars():
    """
    Test MostFrequentImputer with invalid categorical variables.
    It should not change the input DataFrame.
    """
    df = pd.DataFrame(
        {"col1": ["a", "a", np.nan, "b"], "col2": ["x", "y", "x", np.nan]}
    )
    mfi = MostFrequentImputer(cat_vars=["invalid"], threshold=0.5)
    result = mfi.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": ["a", "a", np.nan, "b"], "col2": ["x", "y", "x", np.nan]}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_most_frequent_imputer_mix_valid_invalid_cat_vars():
    """
    Test MostFrequentImputer with a mix of valid and invalid categorical variables.
    It should impute the missing values only in the valid columns.
    """
    df = pd.DataFrame(
        {"col1": ["a", "a", np.nan, "b"], "col2": ["x", "y", "x", np.nan]}
    )
    mfi = MostFrequentImputer(cat_vars=["col1", "invalid"], threshold=0.5)
    result = mfi.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": ["a", "a", "a", "b"], "col2": ["x", "y", "x", np.nan]}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_most_frequent_imputer_empty_dataframe():
    """
    Test MostFrequentImputer with an empty DataFrame.
    It should not change the input DataFrame.
    """
    df = pd.DataFrame()
    mfi = MostFrequentImputer(cat_vars=["col1", "col2"], threshold=0.5)
    result = mfi.fit_transform(df)
    expected = pd.DataFrame()
    pd.testing.assert_frame_equal(result, expected)


def test_most_frequent_imputer_no_missing_values():
    """
    Test MostFrequentImputer with no missing values in the input DataFrame.
    It should not change the input DataFrame.
    """
    df = pd.DataFrame({"col1": ["a", "a", "b", "b"], "col2": ["x", "y", "x", "y"]})
    mfi = MostFrequentImputer(cat_vars=["col1", "col2"], threshold=0.5)
    result = mfi.fit_transform(df)
    expected = pd.DataFrame(
        {"col1": ["a", "a", "b", "b"], "col2": ["x", "y", "x", "y"]}
    )
    pd.testing.assert_frame_equal(result, expected)


def test_most_frequent_imputer_all_missing_values():
    """
    Test MostFrequentImputer with all missing values in the input DataFrame.
    It should not change the input DataFrame since there's no most frequent class.
    """
    df = pd.DataFrame(
        {
            "col1": [np.nan, np.nan, np.nan, np.nan],
            "col2": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    mfi = MostFrequentImputer(cat_vars=["col1", "col2"], threshold=0.5)
    result = mfi.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": [np.nan, np.nan, np.nan, np.nan],
            "col2": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


#  TransformerWrapper Tests
def test_feature_engine_transformer_wrapper_valid_transformer_valid_vars_valid_kwargs():
    """
    Test TransformerWrapper with a valid transformer, valid categorical variables,
    and valid kwargs.
    """
    df = pd.DataFrame(
        {
            "col1": [np.nan, "A", "B", np.nan],
            "col2": [1, 2, 3, 4],
            "col3": ["X", np.nan, "Y", np.nan],
        }
    )
    wrapper = TransformerWrapper(
        transformer=CategoricalImputer,
        variables=["col1", "col3"],
        imputation_method="missing",
        fill_value="Unknown",
    )
    result = wrapper.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": ["Unknown", "A", "B", "Unknown"],
            "col2": [1, 2, 3, 4],
            "col3": ["X", "Unknown", "Y", "Unknown"],
        }
    )
    cols = ["col1", "col2", "col3"]
    pd.testing.assert_frame_equal(result[cols], expected[cols])


def test_feature_engine_transformer_wrapper_invalid_transformer():
    """
    Test TransformerWrapper with an invalid transformer.
    This test case should raise a TypeError.
    """
    df = pd.DataFrame(
        {
            "col1": [np.nan, "A", "B", np.nan],
            "col2": [1, 2, 3, 4],
            "col3": ["X", np.nan, "Y", np.nan],
        }
    )
    with pytest.raises(TypeError):
        wrapper = TransformerWrapper(
            transformer="InvalidTransformer",
            variables=["col1", "col3"],
            imputation_method="missing",
            fill_value="Unknown",
        )
        _ = wrapper.fit_transform(df)


def test_feature_engine_transformer_wrapper_invalid_vars():
    """
    Test TransformerWrapper with invalid variables.
    The transformer should not change the input DataFrame.
    """
    df = pd.DataFrame(
        {
            "col1": [np.nan, 1, 2, np.nan],
            "col2": [1, 2, 3, 4],
            "col3": ["X", np.nan, "Y", np.nan],
        }
    )
    wrapper = TransformerWrapper(
        transformer=AddMissingIndicator,
        variables=["invalid_col"],
        imputation_method="missing",
        fill_value="Unknown",
    )
    result = wrapper.fit_transform(df)
    pd.testing.assert_frame_equal(result, df)


def test_feature_engine_transformer_wrapper_mix_valid_invalid_vars():
    """
    Test TransformerWrapper with a mix of valid and invalid variables.
    The transformer should only apply the transformation to the valid columns.
    """
    df = pd.DataFrame(
        {
            "col1": [np.nan, "A", "B", np.nan],
            "col2": [1, 2, 3, 4],
            "col3": ["X", np.nan, "Y", np.nan],
        }
    )
    wrapper = TransformerWrapper(
        transformer=CategoricalImputer,
        variables=["col1", "invalid_col"],
        imputation_method="missing",
        fill_value="Unknown",
    )
    result = wrapper.fit_transform(df)
    expected = pd.DataFrame(
        {
            "col1": ["Unknown", "A", "B", "Unknown"],
            "col2": [1, 2, 3, 4],
            "col3": ["X", np.nan, "Y", np.nan],
        }
    )
    cols = ["col1", "col2", "col3"]
    pd.testing.assert_frame_equal(result[cols], expected[cols])


def test_feature_engine_transformer_wrapper_empty_dataframe():
    """
    Test TransformerWrapper with an empty DataFrame.
    The transformer should return an empty DataFrame.
    """
    df = pd.DataFrame()
    wrapper = TransformerWrapper(
        transformer=CategoricalImputer,
        variables=["col1", "col3"],
        imputation_method="missing",
        fill_value="Unknown",
    )
    result = wrapper.fit_transform(df)
    pd.testing.assert_frame_equal(result, df)


def test_feature_engine_transformer_wrapper_valid_invalid_kwargs():
    """
    Test TransformerWrapper with valid and invalid kwargs.
    The transformer should raise a TypeError due to the invalid kwargs.
    """
    df = pd.DataFrame(
        {
            "col1": [np.nan, "A", "B", np.nan],
            "col2": [1, 2, 3, 4],
            "col3": ["X", np.nan, "Y", np.nan],
        }
    )
    with pytest.raises(TypeError):
        wrapper = TransformerWrapper(
            transformer=StandardScaler, cat_vars=["col2"], invalid_kwarg="invalid"
        )
        _ = wrapper.fit_transform(df)


def test_valid_ohe_columns_and_max_num_categories():
    # Test with valid ohe_columns and different max_num_categories values.
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red", "red", "green", "blue"],
            "size": ["small", "medium", "large", "medium", "small", "small", "medium"],
            "material": ["wood", "steel", "plastic", "steel", "wood", "sand", "paper"],
            "shape": [
                "circle",
                "square",
                "circle",
                "square",
                "square",
                "circle",
                "square",
            ],
        }
    )
    ohe_columns = ["color", "size", "material", "shape"]
    # column        color       size        material    shape
    # #categories   3           3           5           2
    # #ohe classes  2           2           4           1       <- we drop 1 cat
    max_num_categories_values = [2, 3, 4, 5]
    expected_num_cols_arr = [2 + 2 + 2 + 1, 2 + 2 + 3 + 1, 2 + 2 + 4 + 1, 2 + 2 + 4 + 1]
    for max_num_categories, expected_num_cols in zip(
        max_num_categories_values, expected_num_cols_arr
    ):
        ohe = OneHotEncoderMultipleCols(ohe_columns, max_num_categories)
        ohe.fit(X)
        transformed_X = ohe.transform(X)
        assert transformed_X.shape[1] == expected_num_cols


def test_invalid_ohe_columns():
    # Test with invalid ohe_columns. Should return with invalid columns unchanged.
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green"],
            "size": ["small", "medium", "large"],
            "material": ["wood", "steel", "plastic"],
            "shape": ["circle", "square", "circle"],
        }
    )
    ohe_columns = ["non-existing"]
    ohe = OneHotEncoderMultipleCols(ohe_columns)
    transformed_X = ohe.fit_transform(X)
    assert transformed_X.shape[1] == 4


def test_transforming_with_missing_ohe_columns():
    # Test transforming with missing ohe fitted column - should raise ValueError.
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green"],
            "size": ["small", "medium", "large"],
            "material": ["wood", "steel", "plastic"],
            "shape": ["circle", "square", "circle"],
        }
    )
    ohe_columns = ["color", "material"]
    ohe = OneHotEncoderMultipleCols(ohe_columns)
    ohe.fit(X)
    X_missing_color = X.drop("color", axis=1)
    with pytest.raises(ValueError):
        ohe.transform(X_missing_color)


def test_empty_dataframe():
    # Test with an empty DataFrame.
    X = pd.DataFrame()
    ohe_columns = ["color", "size", "material", "shape"]
    ohe = OneHotEncoderMultipleCols(ohe_columns)
    transformed_X = ohe.fit_transform(X)
    assert transformed_X.shape == (0, 0)


def test_ohe_columns_with_one_category():
    # Test with ohe_columns having only one category.
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green"],
            "size": ["small", "small", "small"],
            "material": ["wood", "wood", "wood"],
            "shape": ["circle", "square", "circle"],
        }
    )
    ohe_columns = ["size", "material"]  # 'size' and 'material' have only one category
    ohe = OneHotEncoderMultipleCols(ohe_columns)
    ohe.fit(X)
    transformed_X = ohe.transform(X)
    assert transformed_X.shape == (3, 4)


def test_ohe_columns_with_more_than_max_num_categories():
    # Test with ohe_columns having more than max_num_categories.
    X = pd.DataFrame(
        {
            "color": ["red", "blue", "green", "red", "red", "green", "blue"],
            "size": ["small", "medium", "large", "medium", "small", "small", "medium"],
            "material": [
                "wood",
                "steel",
                "plastic",
                "steel",
                "wood",
                "wood",
                "plastic",
            ],
            "shape": [
                "circle",
                "square",
                "triangle",
                "square",
                "square",
                "circle",
                "square",
            ],
        }
    )
    ohe_columns = ["color", "size", "material", "shape"]
    max_num_categories = 2
    ohe = OneHotEncoderMultipleCols(ohe_columns, max_num_categories)
    ohe.fit(X)
    transformed_X = ohe.transform(X)
    assert transformed_X.shape[1] == len(ohe_columns) * max_num_categories
