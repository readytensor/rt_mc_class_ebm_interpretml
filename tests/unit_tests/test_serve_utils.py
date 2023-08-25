import os
import pprint
from unittest.mock import MagicMock, patch

import pytest

from src.serve_utils import (
    combine_predictions_response_with_explanations,
    create_predictions_response,
    generate_unique_request_id,
    get_model_resources,
)


@pytest.fixture
def resources_paths():
    """Define a fixture for the paths to the test model resources."""
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_resources_path = os.path.join(cur_dir, "test_resources")
    return {
        "saved_schema_path": os.path.join(test_resources_path, "schema.joblib"),
        "predictor_file_path": os.path.join(test_resources_path, "predictor.joblib"),
        "pipeline_file_path": os.path.join(test_resources_path, "pipeline.joblib"),
        "target_encoder_file_path": os.path.join(
            test_resources_path, "target_encoder.joblib"
        ),
        "model_config_file_path": os.path.join(
            test_resources_path, "model_config.json"
        ),
        "explainer_file_path": os.path.join(test_resources_path, "explainer.joblib"),
    }


@pytest.fixture
def model_resources(resources_paths):
    """Define a fixture for the test ModelResources object."""
    return get_model_resources(**resources_paths)


@patch("serve_utils.uuid.uuid4")
def test_generate_unique_request_id(mock_uuid):
    """Test the generate_unique_request_id function."""
    mock_uuid.return_value = MagicMock(hex="1234567890abcdef1234567890abcdef")
    assert generate_unique_request_id() == "1234567890"


@pytest.fixture
def request_id():
    return generate_unique_request_id()


def test_create_predictions_response(predictions_df, schema_provider, request_id):
    """
    Test the `create_predictions_response` function.

    This test checks that the function returns a correctly structured dictionary,
    including the right keys and that the 'status' field is 'success'.
    It also checks that the 'predictions' field is a list, each element of which is a
    dictionary with the right keys.
    Additionally, it validates the 'predictedClass' is among the 'targetClasses', and
    the sum of 'predictedProbabilities' approximates to 1, allowing for a small
    numerical error.

    Args:
        predictions_df (pd.DataFrame): A fixture providing a DataFrame of model
            predictions.
        schema_provider (MulticlassClassificationSchema): A fixture providing an
            instance of the MulticlassClassificationSchema.

    Returns:
        None
    """
    response = create_predictions_response(predictions_df, schema_provider, request_id)

    # Check that the output is a dictionary
    assert isinstance(response, dict)

    # Check that the dictionary has the correct keys
    expected_keys = {
        "status",
        "message",
        "timestamp",
        "requestId",
        "targetClasses",
        "targetDescription",
        "predictions",
    }
    assert set(response.keys()) == expected_keys

    # Check that the 'status' field is 'success'
    assert response["status"] == "success"

    # Check that the 'predictions' field is a list
    assert isinstance(response["predictions"], list)

    # Check that each prediction has the correct keys
    prediction_keys = {
        "sampleId",
        "predictedClass",
        "predictedProbabilities",
    }
    for prediction in response["predictions"]:
        assert set(prediction.keys()) == prediction_keys

        # Check that 'predictedClass' is one of the 'targetClasses'
        assert prediction["predictedClass"] in response["targetClasses"]

        # Check that 'predictedProbabilities' sum to 1 (within a small tolerance)
        assert abs(sum(prediction["predictedProbabilities"]) - 1.0) < 1e-5


def test_combine_predictions_response_with_explanations():
    """
    Test the combine_predictions_response_with_explanations function.

    Ensures that the explanations are added to the corresponding predictions,
    and that the 'explanationMethod' field in the response is updated correctly.
    """
    # Define a sample predictions response
    predictions_response = {
        "status": "success",
        "message": "",
        "timestamp": "...varies...",
        "requestId": "...varies...",
        "targetClasses": ["0", "1"],
        "targetDescription": "some description",
        "predictions": [
            {
                "sampleId": "879",
                "predictedClass": "0",
                "predictedProbabilities": [0.97548, 0.02452],
            }
        ],
    }

    # Define a sample explanations dictionary
    explanations_response = {
        "explanations": [
            {
                "baseline": [0.57775, 0.42225],
                "featureScores": {
                    "Age_na": [0.05389, -0.05389],
                    "Age": [0.02582, -0.02582],
                    "SibSp": [-0.00469, 0.00469],
                    "Parch": [0.00706, -0.00706],
                    "Fare": [0.05561, -0.05561],
                    "Embarked_S": [0.01582, -0.01582],
                    "Embarked_C": [0.00393, -0.00393],
                    "Embarked_Q": [0.00657, -0.00657],
                    "Pclass_3": [0.0179, -0.0179],
                    "Pclass_1": [0.02394, -0.02394],
                    "Sex_male": [0.13747, -0.13747],
                },
            }
        ],
        "explanation_method": "shap",
    }

    # Run the function under test
    combined = combine_predictions_response_with_explanations(
        predictions_response, explanations_response
    )

    # Check the resulting dictionary
    expected = predictions_response.copy()
    pprint.pprint(combined)
    expected["predictions"][0]["explanation"] = explanations_response["explanations"][0]
    expected["explanationMethod"] = explanations_response["explanation_method"]

    assert combined == expected, f"Expected {expected}, but got {combined}"
