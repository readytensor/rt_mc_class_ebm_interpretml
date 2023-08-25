import pytest
from pydantic import ValidationError

from src.data_models.schema_validator import (
    ID,
    DataType,
    Feature,
    SchemaModel,
    Target,
    validate_schema_dict,
)


# Tests for ID
def test_valid_id():
    """
    Test the `ID` model with valid data.

    Ensures that a valid `ID` object is created without raising an exception.
    """
    valid_id = {"name": "ID", "description": "ID description"}
    id_obj = ID(**valid_id)
    assert id_obj.dict() == valid_id


# Tests for Target
def test_valid_target():
    """
    Test the `Target` model with valid data.

    Ensures that a valid `Target` object is created without raising an exception.
    """
    valid_target = {
        "name": "target",
        "description": "target description",
        "classes": ["class_0", "class_1"],
    }
    target_obj = Target(**valid_target)
    assert target_obj.dict() == valid_target


def test_invalid_target_classes_length():
    """
    Test the `Target` model with invalid data (length of classes != 2).

    Ensures that the model raises an exception for an invalid number of classes.
    """
    invalid_target = {
        "name": "target",
        "description": "target description",
        "classes": ["class_0"],
    }
    with pytest.raises(ValueError):
        Target(**invalid_target)


def test_invalid_target_classes_unique():
    """
    Test the `Target` model with invalid data (not unique classes).

    Ensures that the model raises an exception for non-unique classes.
    """
    invalid_target = {
        "name": "target",
        "description": "target description",
        "classes": ["class_0", "class_0"],
    }
    with pytest.raises(ValueError):
        Target(**invalid_target)


def test_invalid_target_classes_empty_string():
    """
    Test the `Target` model with invalid data (empty string class).

    Ensures that the model raises an exception for classes containing an empty string.
    """
    invalid_target = {
        "name": "target",
        "description": "target description",
        "classes": ["class_0", ""],
    }
    with pytest.raises(ValueError):
        Target(**invalid_target)


# Tests for Feature
def test_valid_numeric_feature():
    """
    Test the `Feature` model with valid data for a numeric feature.

    Ensures that a valid `Feature` object is created without raising an exception
    for numeric data. It also checks if all keys and values of the input dictionary
    are present in the output dictionary and match the input.

    Raises:
        AssertionError: If the created Feature object is not equivalent to the
            input dictionary.
    """
    valid_feature = {
        "name": "feature_1",
        "description": "feature_1 description",
        "dataType": "NUMERIC",
        "example": 1.0,
        "nullable": False,
    }
    feature_obj = Feature(**valid_feature)
    returned_feature = feature_obj.dict()
    # Checks if all keys and values of the input dictionary are present in the
    # output dictionary
    assert all(
        item in returned_feature.items() for item in valid_feature.items()
    ), "Some keys or values in the input dictionary are not present in the \
        output dictionary"


def test_valid_categorical_feature():
    """
    Test the `Feature` model with valid data for a categorical feature.

    Ensures that a valid `Feature` object is created without raising an exception
    for categorical data.
    """
    valid_feature = {
        "name": "feature_2",
        "description": "feature_2 description",
        "dataType": "CATEGORICAL",
        "categories": ["category_1", "category_2"],
        "nullable": False,
    }
    feature_obj = Feature(**valid_feature)
    returned_feature = feature_obj.dict()

    # Checks if all keys and values of the input dictionary are present in the
    # output dictionary
    assert all(
        item in returned_feature.items() for item in valid_feature.items()
    ), "Some keys or values in the input dictionary are not present in the \
        output dictionary"


def test_data_type_enum():
    """
    Tests the DataType enum in the Feature Pydantic model.

    This function creates Feature instances with valid DataType values (NUMERIC and
    CATEGORICAL) and checks that these values are correctly assigned. It also attempts
    to create a Feature instance with an invalid DataType value and checks that this
    raises a ValidationError.

    Raises:
        AssertionError: If the assigned DataType value in the Feature instance does not
            match the expected DataType value.
        ValidationError: If an invalid DataType value is provided during the creation of
            the Feature instance.

    """
    # Verify that a valid DataType value is accepted
    feature1 = Feature(
        name="feature1",
        description="A numeric feature",
        dataType=DataType.NUMERIC,
        nullable=False,
        example=1.23,
    )
    assert feature1.dataType == DataType.NUMERIC

    # Verify that a valid DataType value is accepted
    feature2 = Feature(
        name="feature2",
        description="A categorical feature",
        dataType=DataType.CATEGORICAL,
        nullable=False,
        categories=["cat1", "cat2"],
    )
    assert feature2.dataType == DataType.CATEGORICAL

    # Verify that an invalid DataType value raises a ValidationError
    with pytest.raises(ValidationError):
        Feature(
            name="feature3",
            description="An invalid feature",
            dataType="INVALID_DATA_TYPE",  # This is not a valid DataType
            nullable=False,
            example="invalid",
        )


def test_invalid_feature_data_type():
    """
    Test the `Feature` model with invalid data (invalid data type).

    Ensures that the model raises an exception for an invalid data type.
    """
    invalid_feature = {
        "name": "feature_1",
        "description": "feature_1 description",
        "dataType": "INVALID",
        "example": 1.0,
        "nullable": False,
    }
    with pytest.raises(ValueError):
        Feature(**invalid_feature)


def test_missing_example_in_numeric_feature():
    """
    Test the `Feature` model with invalid data (missing example for numeric feature).

    Ensures that the model raises an exception when the example is missing for a
    numeric feature.
    """
    invalid_feature = {
        "name": "feature_1",
        "description": "feature_1 description",
        "dataType": "NUMERIC",
        "nullable": False,
    }
    with pytest.raises(ValueError):
        Feature(**invalid_feature)


def test_missing_categories_in_categorical_feature():
    """
    Test the `Feature` model with invalid data (missing categories for categorical
    feature).

    Ensures that the model raises an exception when the categories are missing for
    a categorical feature.
    """
    invalid_feature = {
        "name": "feature_2",
        "description": "feature_2 description",
        "dataType": "CATEGORICAL",
        "nullable": False,
    }
    with pytest.raises(ValueError):
        Feature(**invalid_feature)


# Tests for SchemaModel
def test_valid_schema(schema_dict):
    """
    Test the `SchemaModel` with valid data.

    Ensures that a valid `SchemaModel` object is created without raising an exception.

    Args:
        schema_dict (dict): A dictionary representing a valid schema.

    Raises:
        AssertionError: If the parsed schema object does not have expected attributes.
    """
    # Attempt to parse the schema dictionary
    try:
        schema_obj = SchemaModel.parse_obj(schema_dict)
    except Exception as e:
        pytest.fail(f"Schema parsing failed with exception: {e}")

    # Check if the parsed schema object has expected attributes
    for key in schema_dict.keys():
        assert hasattr(
            schema_obj, key
        ), f"Parsed schema object does not have '{key}' attribute."

    assert isinstance(
        schema_obj.features, list
    ), "Parsed schema object attribute 'features' is not a list."
    assert len(schema_dict["features"]) == len(
        schema_obj.features
    ), "Parsed schema object attribute 'features' does not have the expected length."


def test_invalid_schema_model_category(schema_dict):
    """
    Test the `SchemaModel` with invalid data (invalid model category).

    Ensures that the model raises an exception for an invalid model category.
    """
    invalid_schema = schema_dict.copy()
    invalid_schema["modelCategory"] = "INVALID"
    with pytest.raises(ValueError):
        SchemaModel.parse_obj(invalid_schema)


def test_invalid_schema_version(schema_dict):
    """
    Test the `SchemaModel` with invalid data (invalid schema version).

    Ensures that the model raises an exception for an invalid schema version.
    """
    invalid_schema = schema_dict.copy()
    invalid_schema["schemaVersion"] = 2.0
    with pytest.raises(ValueError):
        SchemaModel.parse_obj(invalid_schema)


def test_schema_without_features(schema_dict):
    """
    Test the `SchemaModel` with invalid data (no features defined).

    Ensures that the model raises an exception when no features are defined.
    """
    invalid_schema = schema_dict.copy()
    invalid_schema["features"] = []
    with pytest.raises(ValueError):
        SchemaModel.parse_obj(invalid_schema)


# Tests for validate_schema_dict
def test_validate_schema_dict(schema_dict):
    """
    Test the `validate_schema_dict` function with valid data.

    Ensures that the function successfully validates a correct schema dictionary
    without raising an exception.
    """
    try:
        validate_schema_dict(schema_dict)
    except ValueError as e:
        pytest.fail(f"Validation failed with exception: {e}")


def test_validate_invalid_schema_dict(schema_dict):
    """
    Test the `validate_schema_dict` function with invalid data.

    Ensures that the function raises an exception for an invalid schema dictionary.
    """
    invalid_schema = schema_dict.copy()
    invalid_schema["schemaVersion"] = 2.0
    with pytest.raises(ValueError):
        validate_schema_dict(invalid_schema)


def test_duplicate_feature_names():
    """
    Test that the validation fails when duplicate feature names are present.
    """
    # Example of an invalid schema with duplicate feature names
    invalid_schema = {
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
                "name": "duplicated_feature_name",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 50,
                "nullable": True,
            },
            {
                "name": "duplicated_feature_name",
                "description": "some desc.",
                "dataType": "NUMERIC",
                "example": 0.5,
                "nullable": False,
            },
        ],
    }

    with pytest.raises(ValueError) as exc_info:
        validate_schema_dict(invalid_schema)

    assert "duplicated_feature_name" in str(exc_info.value)
