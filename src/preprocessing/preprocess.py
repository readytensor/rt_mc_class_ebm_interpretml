import os
import random
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from preprocessing.pipeline import (
    get_preprocess_pipeline,
    load_pipeline,
    save_pipeline,
    train_pipeline,
    transform_inputs,
)
from preprocessing.target_encoder import (
    get_target_encoder,
    load_target_encoder,
    save_target_encoder,
    train_target_encoder,
    transform_targets,
)

PIPELINE_FILE_NAME = "pipeline.joblib"
TARGET_ENCODER_FILE_NAME = "target_encoder.joblib"


def train_pipeline_and_target_encoder(
    data_schema: Any, train_split: pd.DataFrame, preprocessing_config: Dict
) -> Tuple[Any, Any]:
    """
    Train the pipeline and target encoder

    Args:
        data_schema (Any): A dictionary containing the data schema.
        train_split (pd.DataFame): A pandas DataFrame containing the train data split.
        preprocessing_config (Dict): A dictionary containing the preprocessing params.

    Returns:
        A tuple containing the pipeline and target encoder.
    """
    # create input trnasformation pipeline and target encoder
    preprocess_pipeline = get_preprocess_pipeline(
        data_schema=data_schema, preprocessing_config=preprocessing_config
    )
    target_encoder = get_target_encoder(data_schema=data_schema)

    # train pipeline and target encoder
    trained_pipeline = train_pipeline(preprocess_pipeline, train_split)
    trained_target_encoder = train_target_encoder(target_encoder, train_split)

    return trained_pipeline, trained_target_encoder


def transform_data(
    preprocess_pipeline: Any, target_encoder: Any, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """
    Transform the data using the preprocessing pipeline and target encoder.

    Args:
        preprocess_pipeline (Any): The preprocessing pipeline.
        target_encoder (Any): The target encoder.
        data (pd.DataFrame): The input data as a DataFrame (targets may be included).

    Returns:
        Tuple[pd.DataFrame, Union[pd.Series, None]]: A tuple containing the transformed
            data and transformed targets;
            transformed targets are None if the data does not contain targets.
    """
    transformed_inputs = transform_inputs(preprocess_pipeline, data)
    transformed_targets = transform_targets(target_encoder, data)
    return transformed_inputs, transformed_targets


def save_pipeline_and_target_encoder(
    preprocess_pipeline: Any, target_encoder: Any, preprocessing_dir_path: str
) -> None:
    """
    Save the preprocessing pipeline and target encoder to files.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        target_encoder: The target encoder.
        preprocessing_dir_path (str): dir path where the pipeline and target encoder
            is to be saved

    """
    if not os.path.exists(preprocessing_dir_path):
        os.makedirs(preprocessing_dir_path)
    save_pipeline(
        pipeline=preprocess_pipeline,
        file_path_and_name=os.path.join(preprocessing_dir_path, PIPELINE_FILE_NAME),
    )
    save_target_encoder(
        target_encoder=target_encoder,
        file_path_and_name=os.path.join(
            preprocessing_dir_path, TARGET_ENCODER_FILE_NAME
        ),
    )


def load_pipeline_and_target_encoder(
    preprocessing_dir_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the preprocessing pipeline and target encoder

    Args:
        preprocessing_dir_path (str): dir path where the pipeline and target encoder
            are saved

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the preprocessing
            pipeline and target encoder.
    """
    preprocess_pipeline = load_pipeline(
        file_path_and_name=os.path.join(preprocessing_dir_path, PIPELINE_FILE_NAME)
    )
    target_encoder = load_target_encoder(
        file_path_and_name=os.path.join(
            preprocessing_dir_path, TARGET_ENCODER_FILE_NAME
        )
    )
    return preprocess_pipeline, target_encoder


def handle_class_imbalance(
    transformed_data: pd.DataFrame, transformed_labels: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance using SMOTE.

    Args:
        transformed_data (pd.DataFrame): The transformed data.
        transformed_labels (pd.Series): The transformed labels.
        random_state (int): The random state seed for reproducibility. Defaults to 0.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the balanced data and
            balanced labels.
    """
    # Adjust k_neighbors parameter for SMOTE
    # set k_neighbors to be the smaller of two values:
    #       1 and,
    #       the number of instances in the minority class minus one
    k_neighbors = min(
        1, sum(transformed_labels == min(transformed_labels.value_counts().index)) - 1
    )
    smote = SMOTE(k_neighbors=k_neighbors, random_state=0)
    balanced_data, balanced_labels = smote.fit_resample(
        transformed_data, transformed_labels
    )
    return balanced_data, balanced_labels


def insert_nulls_in_nullable_features(
    data: pd.DataFrame, data_schema: Any, preprocessing_config: Dict
) -> pd.DataFrame:
    """
    Inserts nulls into specified columns of a DataFrame if nulls are not
    already present.

    Args:
        data (pd.DataFrame): training data.
        schema: schema provider object.
        preprocess_config (dict): preprocessing configuration dictionary.

    Returns:
        pd.DataFrame: Transformed DataFrame with inserted nulls.
    """
    data_copy = data.copy()
    nullable_columns = data_schema.nullable_features
    perc_inserted_nulls = preprocessing_config.get("perc_inserted_nulls", 0.05)

    for column in nullable_columns:
        # Check if the column doesn't already contain any nulls
        if pd.isnull(data_copy[column]).sum() == 0:
            mask = np.random.rand(len(data_copy)) < perc_inserted_nulls
            # If no nulls were inserted due to probabilities, insert at least one
            if mask.sum() == 0:
                # Randomly select one index to insert a null
                null_idx = random.choice(data_copy.index)
                data_copy.loc[null_idx, column] = None
            else:
                data_copy.loc[mask, column] = None

    return data_copy
