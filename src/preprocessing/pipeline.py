from typing import Any

import joblib
import pandas as pd
from feature_engine.encoding import RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    SmartCorrelatedSelection,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocessing import custom_transformers as transformers


def get_preprocess_pipeline(data_schema: Any, preprocessing_config: dict) -> Pipeline:
    """
    Create a preprocessor pipeline to transform data as defined by data_schema.

    Args:
        data_schema (MulticlassClassificationSchema): An instance of the
            MulticlassClassificationSchema.
        preprocessing_config (dict): dictionary for preprocessing configuration.
    Returns:
        Pipeline: A pipeline to transform data as defined by data_schema.
    """

    num_config = preprocessing_config["numeric_transformers"]
    clip_min_val = num_config["outlier_clipper"]["min_val"]
    clip_max_val = num_config["outlier_clipper"]["max_val"]
    imputation_method = num_config["mean_median_imputer"]["imputation_method"]

    cat_config = preprocessing_config["categorical_transformers"]
    cat_imputer_freq_threshold = cat_config["cat_most_frequent_imputer"]["threshold"]
    cat_imputer_missing_method = cat_config["missing_tag_imputer"]["imputation_method"]
    cat_imputer_missing_fill_val = cat_config["missing_tag_imputer"]["fill_value"]
    rare_label_tol = cat_config["rare_label_encoder"]["tol"]
    rare_label_n_categories = cat_config["rare_label_encoder"]["n_categories"]

    feat_sel_pp_config = preprocessing_config["feature_selection_preprocessing"]
    constant_feature_tol = feat_sel_pp_config["constant_feature_dropper"]["tol"]
    constant_feature_missing = feat_sel_pp_config["constant_feature_dropper"][
        "missing_values"
    ]
    correl_feature_threshold = feat_sel_pp_config["correlated_feature_dropper"][
        "threshold"
    ]

    column_selector = transformers.ColumnSelector(columns=data_schema.features)
    nan_col_dropper = transformers.DropAllNaNFeatures(columns=data_schema.features)
    string_caster = transformers.TypeCaster(
        vars=data_schema.categorical_features + [data_schema.id, data_schema.target],
        cast_type=str,
    )
    float_caster = transformers.TypeCaster(
        vars=data_schema.numeric_features, cast_type=float
    )
    missing_indicator_numeric = transformers.TransformerWrapper(
        transformer=AddMissingIndicator, variables=data_schema.numeric_features
    )
    mean_imputer_numeric = transformers.TransformerWrapper(
        transformer=MeanMedianImputer,
        variables=data_schema.numeric_features,
        imputation_method=imputation_method,
    )
    standard_scaler = transformers.TransformerWrapper(
        transformer=StandardScaler, variables=data_schema.numeric_features
    )
    outlier_value_clipper = transformers.ValueClipper(
        fields_to_clip=data_schema.numeric_features,
        min_val=clip_min_val,
        max_val=clip_max_val,
    )
    cat_most_frequent_imputer = transformers.MostFrequentImputer(
        cat_vars=data_schema.categorical_features, threshold=cat_imputer_freq_threshold
    )
    cat_imputer_with_missing_tag = transformers.TransformerWrapper(
        transformer=CategoricalImputer,
        variables=data_schema.categorical_features,
        imputation_method=cat_imputer_missing_method,
        fill_value=cat_imputer_missing_fill_val,
    )
    rare_label_encoder = transformers.TransformerWrapper(
        transformer=RareLabelEncoder,
        variables=data_schema.categorical_features,
        tol=rare_label_tol,
        n_categories=rare_label_n_categories,
    )
    constant_feature_dropper = DropConstantFeatures(
        variables=None,
        tol=constant_feature_tol,
        missing_values=constant_feature_missing,
    )
    duplicated_feature_dropper = DropDuplicateFeatures(
        variables=None, missing_values="raise"
    )
    correlated_feature_dropper = SmartCorrelatedSelection(
        variables=None,
        selection_method="variance",
        threshold=correl_feature_threshold,
        missing_values="raise",
    )
    one_hot_encoder = transformers.OneHotEncoderMultipleCols(
        ohe_columns=data_schema.categorical_features
    )
    column_sorter = transformers.ColumnOrderTransformer()

    pipeline = Pipeline(
        [
            ("column_selector", column_selector),
            ("nan_col_dropper", nan_col_dropper),
            ("string_caster", string_caster),
            ("float_caster", float_caster),
            ("missing_indicator_numeric", missing_indicator_numeric),
            ("mean_imputer_numeric", mean_imputer_numeric),
            ("standard_scaler", standard_scaler),
            ("outlier_value_clipper", outlier_value_clipper),
            ("cat_most_frequent_imputer", cat_most_frequent_imputer),
            ("cat_imputer_with_missing_tag", cat_imputer_with_missing_tag),
            ("rare_label_encoder", rare_label_encoder),
            ("constant_feature_dropper", constant_feature_dropper),
            ("duplicated_feature_dropper", duplicated_feature_dropper),
            ("one_hot_encoder", one_hot_encoder),
            ("correlated_feature_dropper", correlated_feature_dropper),
            ("column_sorter", column_sorter),
        ]
    )

    return pipeline


def train_pipeline(pipeline: Pipeline, train_data: pd.DataFrame) -> pd.DataFrame:
    """
    Train the preprocessing pipeline.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        train_data (pd.DataFrame): The training data as a pandas DataFrame.

    Returns:
        Pipeline: Fitted preprocessing pipeline.
    """
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")
    if train_data.empty:
        raise ValueError("train_data cannot be empty")
    pipeline.fit(train_data)
    return pipeline


def transform_inputs(pipeline: Pipeline, input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the input data using the preprocessing pipeline.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        input_data (pd.DataFrame): The input data as a pandas DataFrame.

    Returns:
        pd.DataFrame: The transformed data.
    """
    return pipeline.transform(input_data)


def save_pipeline(pipeline: Pipeline, file_path_and_name: str) -> None:
    """Save the fitted pipeline to a pickle file.

    Args:
        pipeline (Pipeline): The fitted pipeline to be saved.
        file_path_and_name (str): The path where the pipeline should be saved.
    """
    joblib.dump(pipeline, file_path_and_name)


def load_pipeline(file_path_and_name: str) -> Pipeline:
    """Load the fitted pipeline from the given path.

    Args:
        file_path_and_name: Path to the saved pipeline.

    Returns:
        Fitted pipeline.
    """
    return joblib.load(file_path_and_name)
