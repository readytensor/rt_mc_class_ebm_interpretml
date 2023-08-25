import json
import os
import random
import string

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.schema.data_schema import MulticlassClassificationSchema
from src.serve import create_app
from src.serve_utils import get_model_resources
from src.train import run_training
from src.xai.explainer import ClassificationExplainer


@pytest.fixture
def schema_dict():
    """Fixture to create a sample schema for testing"""
    valid_schema = {
        "title": "test dataset",
        "description": "test dataset",
        "modelCategory": "multiclass_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "id", "description": "unique identifier."},
        "target": {
            "name": "target_field",
            "description": "some target desc.",
            "classes": ["A", "B"],
        },
        "features": [
            {
                "name": "numeric_feature_1",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 50,
                "nullable": True,
            },
            {
                "name": "numeric_feature_2",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 0.5,
                "nullable": False,
            },
            {
                "name": "categorical_feature_1",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["A", "B", "C"],
                "nullable": True,
            },
            {
                "name": "categorical_feature_2",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["P", "Q", "R", "S", "T"],
                "nullable": False,
            },
        ],
    }
    return valid_schema


@pytest.fixture
def schema_provider(schema_dict):
    """Fixture to create a sample schema for testing"""
    return MulticlassClassificationSchema(schema_dict)


@pytest.fixture
def config_dir_path():
    """Fixture to create a sample config_dir_path"""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir_path = os.path.join(cur_dir, "..", "src", "config")
    return config_dir_path


@pytest.fixture
def model_config(config_dir_path):
    """Fixture to create a sample model_config json"""
    model_config_file = os.path.join(config_dir_path, "model_config.json")
    with open(model_config_file, "r", encoding="utf-8") as file:
        model_config = json.load(file)
    return model_config


@pytest.fixture
def preprocessing_config(config_dir_path):
    """Fixture to create a preprocessing config"""
    preprocessing_config_file = os.path.join(config_dir_path, "preprocessing.json")
    with open(preprocessing_config_file, "r", encoding="utf-8") as file:
        pp_config = json.load(file)
    return pp_config


@pytest.fixture
def preprocessing_config_file_path(preprocessing_config, tmpdir):
    """Fixture to create and save a sample preprocessing_config json"""
    config_file_path = tmpdir.join("preprocessing.json")
    with open(config_file_path, "w") as file:
        json.dump(preprocessing_config, file)
    return str(config_file_path)


@pytest.fixture
def sample_data():
    """Fixture to create a larger sample DataFrame for testing"""
    np.random.seed(0)
    N = 100
    data = pd.DataFrame(
        {
            "id": range(1, N + 1),
            "numeric_feature_1": np.random.randint(1, 100, size=N),
            "numeric_feature_2": np.random.normal(0, 1, size=N),
            "categorical_feature_1": np.random.choice(["A", "B", "C"], size=N),
            "categorical_feature_2": np.random.choice(
                ["P", "Q", "R", "S", "T"], size=N
            ),
            "target_field": np.random.choice(["A", "B"], size=N),
        }
    )
    return data


@pytest.fixture
def sample_train_data(sample_data):
    """Fixture to create a larger sample DataFrame for testing"""
    N_train = int(len(sample_data) * 0.8)
    return sample_data.head(N_train)


@pytest.fixture
def sample_test_data(sample_data):
    """Fixture to create a larger sample DataFrame for testing"""
    N_test = int(len(sample_data) * 0.2)
    return sample_data.tail(N_test)


@pytest.fixture
def train_data_file_name():
    return "train.csv"


@pytest.fixture
def train_dir(sample_train_data, tmpdir, train_data_file_name):
    """Fixture to create and save a sample DataFrame for testing"""
    train_data_dir = tmpdir.mkdir("train")
    train_data_file_path = train_data_dir.join(train_data_file_name)
    sample_train_data.to_csv(train_data_file_path, index=False)
    return str(train_data_dir)


@pytest.fixture
def test_data_file_name():
    return "test.csv"


@pytest.fixture
def test_dir(sample_test_data, tmpdir, test_data_file_name):
    """Fixture to create and save a sample DataFrame for testing"""
    test_data_dir = tmpdir.mkdir("test")
    test_data_file_path = test_data_dir.join(test_data_file_name)
    sample_test_data.to_csv(test_data_file_path, index=False)
    return str(test_data_dir)


@pytest.fixture
def input_schema_file_name():
    return "schema.json"


@pytest.fixture
def input_schema_dir(schema_dict, tmpdir, input_schema_file_name):
    """Fixture to create and save a sample schema for testing"""
    schema_dir = tmpdir.mkdir("input_schema")
    schema_file_path = schema_dir.join(input_schema_file_name)
    with open(schema_file_path, "w") as file:
        json.dump(schema_dict, file)
    return str(schema_dir)


@pytest.fixture
def model_config_file_path(model_config, tmpdir):
    """Fixture to create and save a sample model_config json"""
    config_file_path = tmpdir.join("model_config.json")
    with open(config_file_path, "w") as file:
        json.dump(model_config, file)
    return str(config_file_path)


@pytest.fixture
def default_hyperparameters(config_dir_path):
    """Fixture to load and return default hyperparameters"""
    default_hps_file_path = os.path.join(
        config_dir_path, "default_hyperparameters.json"
    )
    with open(default_hps_file_path, "r", encoding="utf-8") as file:
        default_hps = json.load(file)
    return default_hps


@pytest.fixture
def default_hyperparameters_file_path(default_hyperparameters, tmpdir):
    """Fixture to return default_hyperparameters file path in tmpdir"""
    config_file_path = tmpdir.join("default_hyperparameters.json")
    with open(config_file_path, "w") as file:
        json.dump(default_hyperparameters, file)
    return str(config_file_path)


@pytest.fixture
def hpt_specs(config_dir_path):
    """Fixture to load and return hyperparameter tuning config"""
    json_file_path = os.path.join(config_dir_path, "hpt.json")
    with open(json_file_path, "r", encoding="utf-8") as file:
        hpt_config = json.load(file)
    return hpt_config


@pytest.fixture
def hpt_specs_file_path(hpt_specs, tmpdir):
    """Fixture to create and save a sample hpt_specs json"""
    config_file_path = tmpdir.join("hpt_specs.json")
    with open(config_file_path, "w") as file:
        json.dump(hpt_specs, file)
    return str(config_file_path)


@pytest.fixture
def predictions_df():
    """Fixture for creating a DataFrame representing predictions."""
    num_preds = 50
    # Create 5 random probabilities
    probabilities_A = [random.uniform(0, 1) for _ in range(num_preds)]
    # Subtract each probability from 1 to create a complementary probability
    probabilities_B = [1 - p for p in probabilities_A]

    # Create a DataFrame with an 'id' column and two class probability
    # columns 'A' and 'B'
    df = pd.DataFrame(
        {
            "id": [
                "".join(
                    random.choices(string.ascii_lowercase + string.digits, k=num_preds)
                )
                for _ in range(num_preds)
            ],
            "A": probabilities_A,
            "B": probabilities_B,
        }
    )
    return df


@pytest.fixture
def explainer_config(config_dir_path):
    json_file_path = os.path.join(config_dir_path, "explainer.json")
    with open(json_file_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config


@pytest.fixture
def explainer_config_file_path(explainer_config, tmpdir):
    """Fixture to create and save a sample explainer_config json"""
    config_file_path = tmpdir.join("explainer.json")
    with open(config_file_path, "w", encoding="utf-8") as file:
        json.dump(explainer_config, file)
    return str(config_file_path)


@pytest.fixture
def test_resources_dir_path(tmpdir):
    """Define a fixture for the path to the test_resources directory."""
    tmpdir.mkdir("test_resources")
    test_resources_path = os.path.join(tmpdir, "test_resources")
    return test_resources_path


@pytest.fixture
def config_file_paths_dict(
    default_hyperparameters_file_path,
    explainer_config_file_path,
    hpt_specs_file_path,
    model_config_file_path,
    preprocessing_config_file_path,
):
    """Define a fixture for the paths to all config files."""
    return {
        "default_hyperparameters_file_path": default_hyperparameters_file_path,
        "explainer_config_file_path": explainer_config_file_path,
        "hpt_specs_file_path": hpt_specs_file_path,
        "model_config_file_path": model_config_file_path,
        "preprocessing_config_file_path": preprocessing_config_file_path,
    }


@pytest.fixture
def resources_paths_dict(test_resources_dir_path, model_config_file_path):
    """Define a fixture for the paths to the test model resources."""
    return {
        "saved_schema_dir_path": os.path.join(test_resources_dir_path, "schema"),
        "predictor_dir_path": os.path.join(test_resources_dir_path, "predictor"),
        "preprocessing_dir_path": os.path.join(
            test_resources_dir_path, "preprocessing"
        ),
        "model_config_file_path": model_config_file_path,
        "explainer_dir_path": os.path.join(test_resources_dir_path, "explainer"),
        "hpt_results_dir_path": os.path.join(test_resources_dir_path, "hpt"),
        "predictions_file_path": os.path.join(
            test_resources_dir_path, "predictions.csv"
        ),
    }


@pytest.fixture
def sample_request_data(schema_dict):
    # Define a fixture for test request data
    sample_dict = {
        # made up id for this test
        schema_dict["id"]["name"]: "42",
    }
    for feature in schema_dict["features"]:
        if feature["dataType"] == "NUMERIC":
            sample_dict[feature["name"]] = feature["example"]
        elif feature["dataType"] == "CATEGORICAL":
            sample_dict[feature["name"]] = feature["categories"][0]
    return {"instances": [{**sample_dict}]}


@pytest.fixture
def sample_response_data(schema_dict):
    # Define a fixture for expected response
    return {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetClasses": schema_dict["target"]["classes"],
        "targetDescription": schema_dict["target"]["description"],
        "predictions": [
            {
                "sampleId": "42",
                # unknown because we don't know the predicted class
                "predictedClass": "unknown",
                # predicted probabilities are made up for this test
                "predictedProbabilities": [0.5, 0.5],
            }
        ],
    }


@pytest.fixture
def sample_explanation_response_data(schema_dict):
    # Define a fixture for expected response
    feature_scores = {
        # made up id for this test
        schema_dict["id"]["name"]: "42",
    }
    for feature in schema_dict["features"]:
        feature_scores[feature["name"]] = [42, 42]
    return {
        "status": "success",
        "message": "",
        # made up timestamp
        "timestamp": "2023-05-22T10:51:45.860800",
        "requestId": "made_up_id",
        "targetClasses": schema_dict["target"]["classes"],
        "targetDescription": schema_dict["target"]["description"],
        "predictions": [
            {
                "sampleId": "42",
                # unknown because we don't know the predicted class
                "predictedClass": "unknown",
                # predicted probabilities are made up for this test
                "predictedProbabilities": [0.5, 0.5],
                "explanation": {
                    # all values are made up
                    "baseline": [42, 42],
                    "featureScores": {**feature_scores},
                },
            }
        ],
        "explanationMethod": ClassificationExplainer.EXPLANATION_METHOD,
    }


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
