import json
from unittest.mock import patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serve import create_app
from src.serve_utils import get_model_resources
from src.train import run_training


@pytest.fixture
def app(
    input_schema_dir,
    train_dir,
    config_file_paths_dict: dict,
    resources_paths_dict: dict,
):
    """
    Define a fixture for the test app.

    Args:
        input_schema_dir (str): Directory path to the input data schema.
        train_dir (str): Directory path to the training data.
        config_file_paths_dict (dict): Dictionary containing the paths to the
            configuration files.
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

    # create model resources dictionary
    model_resources = get_model_resources(**resources_paths_dict)

    # create test app
    return TestClient(create_app(model_resources))


def test_ping(app):
    """Test the /ping endpoint."""
    response = app.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"message": "Pong!"}


@patch("src.serve.transform_req_data_and_make_predictions")
def test_infer_endpoint(mock_transform_and_predict, app, sample_request_data):
    """
    Test the infer endpoint.

    The function creates a mock request and sets the expected return value of the
    mock_transform_and_predict function.
    It then sends a POST request to the "/infer" endpoint with the mock request data.
    The function asserts that the response status code is 200 and the JSON response
    matches the expected output.
    Additionally, it checks if the mock_transform_and_predict function was called with
    the correct arguments.

    Args:
        mock_transform_and_predict (MagicMock): A mock of the
            transform_req_data_and_make_predictions function.
        app (TestClient): The TestClient fastapi app

    """
    # Define what your mock should return
    mock_transform_and_predict.return_value = pd.DataFrame(), {
        "status": "success",
        "predictions": [],
    }

    response = app.post("/infer", data=json.dumps(sample_request_data))

    print(response.json())
    assert response.status_code == 200
    assert response.json() == {"status": "success", "predictions": []}
    # You can add more assertions to check if the function was called with the
    # correct arguments
    mock_transform_and_predict.assert_called()
