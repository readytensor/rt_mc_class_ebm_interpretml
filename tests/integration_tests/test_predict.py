import os

import pandas as pd

from predict import run_batch_predictions
from train import run_training


def test_integration_run_batch_predictions_without_hpt(
    input_schema_dir,
    train_dir,
    config_file_paths_dict: dict,
    test_dir,
    sample_test_data,
    schema_provider,
    resources_paths_dict: dict,
):
    """
    Integration test for the run_batch_predictions function.

    This test simulates the full prediction pipeline, from reading the test data
    to saving the final predictions. The function is tested to ensure that it
    reads in the test data correctly, properly transforms the data, makes the
    predictions, and saves the final predictions in the correct format and location.

    The test also checks that the function handles various file and directory paths
    correctly and that it can handle variations in the input data.

    The test uses a temporary directory provided by the pytest's tmpdir fixture to
    avoid affecting the actual file system.

    Args:
        input_schema_dir (str): Directory path to the input data schema.
        train_dir (str): Directory path to the training data.
        config_file_paths_dict (dict): Dictionary containing the paths to the
            configuration files.
        test_dir (str): Directory path to the test data.
        sample_test_data (pd.DataFrame): Sample DataFrame for testing.
        schema_provider (Any): Loaded schema provider.
        resources_paths_dict (dict): Dictionary containing the paths to the
            resources files such as trained models, encoders, and explainers.
    """
    # extract paths to all config files
    model_config_file_path = config_file_paths_dict["model_config_file_path"]
    preprocessing_config_file_path = config_file_paths_dict[
        "preprocessing_config_file_path"
    ]
    default_hyperparameters_file_path = config_file_paths_dict[
        "default_hyperparameters_file_path"
    ]
    hpt_specs_file_path = config_file_paths_dict["hpt_specs_file_path"]
    explainer_config_file_path = config_file_paths_dict["explainer_config_file_path"]

    # Create temporary paths for all outputs/artifacts
    saved_schema_dir_path = resources_paths_dict["saved_schema_dir_path"]
    preprocessing_dir_path = resources_paths_dict["preprocessing_dir_path"]
    predictor_dir_path = resources_paths_dict["predictor_dir_path"]
    hpt_results_dir_path = resources_paths_dict["hpt_results_dir_path"]
    explainer_dir_path = resources_paths_dict["explainer_dir_path"]

    # Run the training process without hyperparameter tuning
    run_tuning = False
    run_training(
        input_schema_dir=input_schema_dir,
        saved_schema_dir_path=saved_schema_dir_path,
        model_config_file_path=model_config_file_path,
        train_dir=train_dir,
        preprocessing_config_file_path=preprocessing_config_file_path,
        preprocessing_dir_path=preprocessing_dir_path,
        predictor_dir_path=predictor_dir_path,
        default_hyperparameters_file_path=default_hyperparameters_file_path,
        run_tuning=run_tuning,
        hpt_specs_file_path=hpt_specs_file_path if run_tuning else None,
        hpt_results_dir_path=hpt_results_dir_path if run_tuning else None,
        explainer_config_file_path=explainer_config_file_path,
        explainer_dir_path=explainer_dir_path,
    )

    # Get temporary path for prediction
    predictions_file_path = resources_paths_dict["predictions_file_path"]

    # Run the prediction process
    run_batch_predictions(
        saved_schema_dir_path=saved_schema_dir_path,
        model_config_file_path=model_config_file_path,
        test_dir=test_dir,
        preprocessing_dir_path=preprocessing_dir_path,
        predictor_dir_path=predictor_dir_path,
        predictions_file_path=predictions_file_path,
    )

    # Assert that the predictions file is saved in the correct path
    assert os.path.isfile(predictions_file_path)

    # Load predictions and validate the format
    predictions_df = pd.read_csv(predictions_file_path)

    # Assert that predictions dataframe has the right columns
    assert schema_provider.id in predictions_df.columns
    for class_ in schema_provider.target_classes:
        assert class_ in predictions_df.columns

    # Assert that the number of rows in the predictions matches the number
    # of rows in the test data
    assert len(predictions_df) == len(sample_test_data)
