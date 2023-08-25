import pytest
from pydantic import ValidationError

from data_models.infer_request_model import (
    create_instance_model,
    get_inference_request_body_model,
)
from src.schema.data_schema import MulticlassClassificationSchema


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
            "positiveClass": "A",
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
def sample_request_data(sample_instance_data):
    # Define a fixture for test data
    return {"instances": [sample_instance_data]}


def test_can_create_instance_model(schema_provider):
    """
    Test creation of the instance model with valid schema provider.

    Ensures that a valid instance model is created without raising an exception.
    """
    try:
        _ = create_instance_model(schema_provider)
    except Exception as e:
        pytest.fail(f"Instance model creation failed with exception: {e}")


@pytest.fixture
def SampleInstanceModel(schema_provider):
    InstanceModel = create_instance_model(schema_provider)
    return InstanceModel


def test_valid_instance(SampleInstanceModel):
    """
    Test the instance model with valid instances.
    """
    # valid instance
    try:
        _ = SampleInstanceModel.parse_obj(
            {
                "id": "1232",
                "numeric_feature_1": 50,
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )
    except Exception as e:
        pytest.fail(f"Instance parsing failed with exception: {e}")

    # valid instance with extra feature (still valid)
    try:
        _ = SampleInstanceModel.parse_obj(
            {
                "id": "1232",
                "numeric_feature_1": 50,
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
                "extra_feature": 0.5,
            }
        )
    except Exception as e:
        pytest.fail(f"Instance parsing failed with exception: {e}")


def test_invalid_instance(SampleInstanceModel):
    """
    Test the instance model with invalid instances.

    Ensures that instance model validation raises an exception.
    """
    # empty instance
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj({})

    # missing feature_1
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "id": "1232",
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )

    # missing all features
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj({"id": "1232"})

    # missing id
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "numeric_feature_1": 50,
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )

    # id is None
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "id": None,
                "numeric_feature_1": 50,
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )

    # missing id and feature
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )

    # wrong data type for numeric_feature_1
    with pytest.raises(ValidationError):
        _ = SampleInstanceModel.parse_obj(
            {
                "id": "1232",
                "numeric_feature_1": "invalid",
                "numeric_feature_2": 0.5,
                "categorical_feature_1": "A",
                "categorical_feature_2": "P",
            }
        )


def test_get_inference_request_body_model(schema_provider):
    """
    Test creation of the instance model with valid schema provider.

    Ensures that a valid instance model is created without raising an exception.
    """
    try:
        _ = get_inference_request_body_model(schema_provider)
    except Exception as e:
        pytest.fail(f"Request Body model creation failed with exception: {e}")


@pytest.fixture
def SampleRequestBodyModel(schema_provider):
    InferenceRequestBody = get_inference_request_body_model(schema_provider)
    return InferenceRequestBody


def test_valid_inference_request_body(SampleRequestBodyModel):
    """
    Test the inference request body model with valid data.

    Ensures that a valid request body model is created without raising an exception.
    """
    # valid request with single instance
    try:
        # valid request body
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "1232",
                        "numeric_feature_1": 50,
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "P",
                    }
                ]
            }
        )
    except Exception as e:
        pytest.fail(f"Inference request body parsing failed with exception: {e}")

    # valid request with multiple instances
    try:
        # valid request body
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "123",
                        "numeric_feature_1": 50,
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "P",
                    },
                    {
                        "id": "456",
                        "numeric_feature_1": 60,
                        "numeric_feature_2": 1.5,
                        "categorical_feature_1": "B",
                        "categorical_feature_2": "Q",
                    },
                ]
            }
        )
    except Exception as e:
        pytest.fail(f"Inference request body parsing failed with exception: {e}")

    # valid request - extra key
    try:
        # valid request body
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "1232",
                        "numeric_feature_1": 50,
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "P",
                    }
                ],
                "extra": "key",
            }
        )
    except Exception as e:
        pytest.fail(f"Inference request body parsing failed with exception: {e}")


def test_invalid_inference_request_body(SampleRequestBodyModel):
    """
    Test the inference request body model with invalid instance(s).

    - Test the inference request body model with missing 'instances'.
    - Test the inference request body model with empty 'instances'.
    - Test the inference request body model with empty 'instances'.

    Ensures that request body model validation raises an exception.
    """
    # request is empty
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj({})

    # 'instances' is not a list
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj({"instances": "invalid"})

    # 'instances' is empty
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj({"instances": []})

    # 'instances' has sample instance with missing feature
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "1232",
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "P",
                    }
                ]
            }
        )

    # 'instances' has valid and invalid sample
    with pytest.raises(ValidationError):
        _ = SampleRequestBodyModel.parse_obj(
            {
                "instances": [
                    {
                        "id": "123",
                        "numeric_feature_1": 50,
                        "numeric_feature_2": 0.5,
                        "categorical_feature_1": "A",
                        "categorical_feature_2": "P",
                    },
                    {
                        "id": "456",
                        "numeric_feature_1": 60,
                        "categorical_feature_1": "B",
                        "categorical_feature_2": "Q",
                    },
                ]
            }
        )
