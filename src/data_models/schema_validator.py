from collections import Counter
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ValidationError, validator


class ID(BaseModel):
    """
    A model representing the ID field of the dataset.
    """

    name: str
    description: str


class Target(BaseModel):
    """
    A model representing the target field of a multiclass classification problem.
    """

    name: str
    description: str
    classes: List[str]

    @validator("classes", allow_reuse=True)
    def target_classes_are_two_and_unique_and_not_empty_str(cls, target_classes):
        if len(target_classes) < 2:
            raise ValueError(
                f"Target classes must be a list with at least two labels."
                f"Given `{target_classes}` with length {len(target_classes)}"
            )
        if len(set(target_classes)) != len(target_classes):
            raise ValueError(
                "Target classes must be unique. " f"Given `{target_classes}`"
            )
        if "" in target_classes:
            raise ValueError(
                "Target classes must not contain empty strings. "
                f"Given `{target_classes}`"
            )
        return target_classes


class DataType(str, Enum):
    """Enum for the data type of a feature"""

    NUMERIC = "NUMERIC"
    CATEGORICAL = "CATEGORICAL"


class Feature(BaseModel):
    """
    A model representing the predictor fields in the dataset. Validates the
    presence and type of the 'example' field based on the 'dataType' field
    for NUMERIC dataType and presence and contents of the 'categories' field
    for CATEGORICAL dataType.
    """

    name: str
    description: str
    dataType: DataType
    nullable: bool
    example: Optional[float]
    categories: Optional[List[str]]

    @validator("example", always=True, allow_reuse=True)
    def example_is_present_with_data_type_is_numeric(cls, v, values):
        data_type = values.get("dataType")
        if data_type == "NUMERIC" and v is None:
            raise ValueError(
                f"`example` must be present and a float or an integer "
                f"when dataType is NUMERIC. Check field: {values}"
            )
        return v

    @validator("categories", always=True, allow_reuse=True)
    def categories_are_present_with_data_type_is_categorical(cls, v, values):
        data_type = values.get("dataType")
        if data_type == "CATEGORICAL" and v is None:
            raise ValueError(
                "`categories` must be present when dataType is CATEGORICAL. "
                f"Check field: {values}"
            )
        return v

    @validator("categories", always=True, allow_reuse=True)
    def categories_are_non_empty_strings(cls, v, values):
        categories = values.get("categories")
        if categories is not None:
            if len(categories) == 0:
                raise ValueError(
                    f"`categories` must not be empty. Check field: {values}"
                )
            for category in categories:
                if str(category) == "" or not isinstance(category, str):
                    raise ValueError(
                        f"`categories` must be a list of strings. Check field: {values}"
                    )
        return v


class SchemaModel(BaseModel):
    """
    A schema validator for multiclass classification problems. Validates the
    problem category, version, and predictor fields of the input schema.
    """

    title: str
    description: str = None
    modelCategory: str
    schemaVersion: float
    inputDataFormat: str = None
    encoding: str = None
    id: ID
    target: Target
    features: List[Feature]

    @validator("modelCategory", allow_reuse=True)
    def valid_problem_category(cls, v):
        if v != "multiclass_classification":
            raise ValueError(
                f"modelCategory must be 'multiclass_classification'. Given {v}"
            )
        return v

    @validator("schemaVersion", allow_reuse=True)
    def valid_version(cls, v):
        if v != 1.0:
            raise ValueError(f"schemaVersion must be set to 1.0. Given {v}")
        return v

    @validator("features", allow_reuse=True)
    def at_least_one_predictor_field(cls, v):
        if len(v) < 1:
            raise ValueError(
                f"features must have at least one field defined. Given {v}"
            )
        return v

    @validator("features", allow_reuse=True)
    def unique_feature_names(cls, v):
        """
        Check that the feature names are unique.
        """
        feature_names = [feature.name for feature in v]
        duplicates = [
            item for item, count in Counter(feature_names).items() if count > 1
        ]

        if duplicates:
            raise ValueError(
                "Duplicate feature names found in schema: " f"`{', '.join(duplicates)}`"
            )

        return v


def validate_schema_dict(schema_dict: dict) -> dict:
    """
    Validate the schema
    Args:
        schema_dict: dict
            data schema as a python dictionary

    Raises:
        ValueError: if the schema is invalid

    Returns:
        dict: validated schema as a python dictionary
    """
    try:
        schema_dict = SchemaModel.parse_obj(schema_dict).dict()
        return schema_dict
    except ValidationError as exc:
        raise ValueError(f"Invalid schema: {exc}") from exc
