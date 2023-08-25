import os

import pytest

from src.schema.data_schema import (
    SCHEMA_FILE_NAME,
    MulticlassClassificationSchema,
    load_json_data_schema,
    load_saved_schema,
    save_schema,
)


def test_init():
    """
    Test the initialization of MulticlassClassificationSchema class with a valid schema
    dictionary.

    Asserts that the properties of the schema object match the input schema dictionary.
    """
    # Given
    schema_dict = {
        "modelCategory": "multiclass_classification",
        "title": "Test Title",
        "description": "Test description",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID"},
        "target": {"name": "Test Target", "classes": ["0", "1"], "nullable": False},
        "features": [{"name": "Test feature", "dataType": "NUMERIC", "nullable": True}],
    }

    # When
    schema = MulticlassClassificationSchema(schema_dict)

    # Then
    assert schema.model_category == "multiclass_classification"
    assert schema.title == "Test Title"
    assert schema.description == "Test description"
    assert schema.schema_version == 1.0
    assert schema.input_data_format == "CSV"
    assert schema.id == "Test ID"
    assert schema.target == "Test Target"
    assert schema.target_classes == ["0", "1"]
    assert schema.numeric_features == ["Test feature"]
    assert schema.categorical_features == []
    assert schema.features == ["Test feature"]
    assert schema.all_fields == ["Test ID", "Test Target", "Test feature"]


def test_get_allowed_values_for_categorical_feature():
    """
    Test the method to get allowed values for a categorical feature.
    Asserts that the allowed values match the input schema dictionary.
    Also tests for a ValueError when trying to get allowed values for a non-existent
    feature.
    """
    # Given
    schema_dict = {
        "modelCategory": "multiclass_classification",
        "title": "Test Title",
        "description": "Test description",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID"},
        "target": {"name": "Test Target", "classes": ["0", "1"]},
        "features": [
            {"name": "Test feature 1", "dataType": "NUMERIC", "nullable": True},
            {
                "name": "Test feature 2",
                "dataType": "CATEGORICAL",
                "categories": ["A", "B"],
                "nullable": True,
            },
        ],
    }
    schema = MulticlassClassificationSchema(schema_dict)

    # When
    allowed_values = schema.get_allowed_values_for_categorical_feature("Test feature 2")

    # Then
    assert allowed_values == ["A", "B"]

    # When / Then
    with pytest.raises(ValueError):
        schema.get_allowed_values_for_categorical_feature("Invalid feature")


def test_get_example_value_for_numeric_feature():
    """
    Test the method to get an example value for a numeric feature.
    Asserts that the example value matches the input schema dictionary.
    Also tests for a ValueError when trying to get an example value for a non-existent
    feature.
    """
    # Given
    schema_dict = {
        "modelCategory": "multiclass_classification",
        "title": "Test Title",
        "description": "Test description",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID"},
        "target": {"name": "Test Target", "classes": ["0", "1"]},
        "features": [
            {
                "name": "Test feature 1",
                "dataType": "NUMERIC",
                "example": 123.45,
                "nullable": True,
            },
            {
                "name": "Test feature 2",
                "dataType": "CATEGORICAL",
                "categories": ["A", "B"],
                "nullable": True,
            },
        ],
    }
    schema = MulticlassClassificationSchema(schema_dict)

    # When
    example_value = schema.get_example_value_for_feature("Test feature 1")

    # Then
    assert example_value == 123.45

    # When / Then
    with pytest.raises(ValueError):
        schema.get_example_value_for_feature("Invalid feature")


def test_get_description_for_id_target_and_features():
    """
    Test the methods to get descriptions for the id, target, and features.
    Asserts that the descriptions match the input schema dictionary.
    Also tests for a ValueError when trying to get a description for a non-existent
    feature.
    """
    # Given
    schema_dict = {
        "modelCategory": "multiclass_classification",
        "title": "Test Title",
        "description": "Test description",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID", "description": "ID field"},
        "target": {
            "name": "Test Target",
            "description": "Target field",
            "classes": ["0", "1"],
        },
        "features": [
            {
                "name": "Test feature 1",
                "dataType": "NUMERIC",
                "description": "Numeric feature",
                "nullable": True,
            },
            {
                "name": "Test feature 2",
                "dataType": "CATEGORICAL",
                "description": "Categorical feature",
                "categories": ["A", "B"],
                "nullable": True,
            },
        ],
    }
    schema = MulticlassClassificationSchema(schema_dict)

    # When
    id_description = schema.id_description
    target_description = schema.target_description
    feature_1_description = schema.get_description_for_feature("Test feature 1")
    feature_2_description = schema.get_description_for_feature("Test feature 2")

    # Then
    assert id_description == "ID field"
    assert target_description == "Target field"
    assert feature_1_description == "Numeric feature"
    assert feature_2_description == "Categorical feature"

    # When / Then
    with pytest.raises(ValueError):
        schema.get_description_for_feature("Invalid Feature")


def test_is_feature_nullable():
    """
    Test the method to check if a feature is nullable.
    Asserts that the nullable status matches the input schema dictionary.
    Also tests for a ValueError when trying to check the nullable status for a
    non-existent feature.
    """
    # Given
    schema_dict = {
        "modelCategory": "multiclass_classification",
        "title": "Test Title",
        "description": "Test description",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "Test ID", "nullable": False},
        "target": {"name": "Test Target", "classes": ["0", "1"], "nullable": False},
        "features": [
            {"name": "Test feature 1", "dataType": "NUMERIC", "nullable": True},
            {
                "name": "Test feature 2",
                "dataType": "CATEGORICAL",
                "categories": ["A", "B"],
                "nullable": False,
            },
        ],
    }
    schema = MulticlassClassificationSchema(schema_dict)

    # When
    is_nullable = schema.is_feature_nullable("Test feature 1")

    # Then
    assert is_nullable is True

    # When
    is_not_nullable = schema.is_feature_nullable("Test feature 2")

    # Then
    assert is_not_nullable is False

    # When / Then
    with pytest.raises(ValueError):
        schema.is_feature_nullable("Invalid feature")


def test_load_json_data_schema(input_schema_dir):
    """
    Test the method to load a schema from a JSON file.
    Asserts that the properties of the schema object match the input schema dictionary.
    """
    # Given input_schema_dir

    # When
    schema = load_json_data_schema(input_schema_dir)

    # Then
    assert isinstance(schema, MulticlassClassificationSchema)
    assert schema.model_category == "multiclass_classification"


def test_save_and_load_schema(tmpdir, schema_provider):

    # Save the schema using the save_schema function
    save_dir_path = str(tmpdir)
    save_schema(schema_provider, save_dir_path)

    # Check if file was saved correctly
    file_path = os.path.join(save_dir_path, SCHEMA_FILE_NAME)
    assert os.path.isfile(file_path)

    # Load the schema using the load_saved_schema function
    loaded_schema = load_saved_schema(save_dir_path)

    # Check if the loaded schema is an instance of MulticlassClassificationSchema
    assert isinstance(loaded_schema, MulticlassClassificationSchema)
    assert loaded_schema.model_category == "multiclass_classification"


def test_load_saved_schema_nonexistent_file(tmpdir):
    # Try to load the schema from a non-existent file
    save_dir_path = os.path.join(tmpdir, "non_existent")

    with pytest.raises(FileNotFoundError):
        returned = load_saved_schema(save_dir_path)
        print(returned)
