import pandas as pd
from pydantic import BaseModel, validator

from schema.data_schema import MulticlassClassificationSchema


def get_predictions_validator(schema: MulticlassClassificationSchema) -> BaseModel:
    """
    Returns a dynamic Pydantic data validator class based on the provided schema.

    The resulting validator checks the following:

    1. That the input DataFrame contains the ID field specified in the schema.
    2. That the input DataFrame contains two fields named as target classes.

    If any of these checks fail, the validator will raise a ValueError.

    Args:
        schema (MulticlassClassificationSchema): An instance of
                                                 MulticlassClassificationSchema.

    Returns:
        BaseModel: A dynamic Pydantic BaseModel class for data validation.
    """

    class DataValidator(BaseModel):
        data: pd.DataFrame

        class Config:
            arbitrary_types_allowed = True

        @validator("data", allow_reuse=True)
        def validate_dataframe(cls, data):

            # Check if DataFrame is empty
            if data.empty:
                raise ValueError(
                    "ValueError: The provided predictions file is empty. "
                    "No scores can be generated. "
                )

            if schema.id not in data.columns:
                raise ValueError(
                    "ValueError: Malformed predictions file. "
                    f"ID field '{schema.id}' is not present in predictions file."
                )

            missing_classes = set(schema.target_classes) - set(data.columns)
            if missing_classes:
                raise ValueError(
                    "ValueError: Malformed predictions file. Target field(s) "
                    f"{missing_classes} missing in predictions file.\n"
                    "Please ensure that the predictions file contains "
                    f"columns named {schema.target_classes} representing "
                    "predicted class probabilities."
                )

            # Check if probabilities are valid
            for class_ in schema.target_classes:
                if not data[class_].between(0, 1).all():
                    raise ValueError(
                        "ValueError: Invalid probabilities in predictions file. Some "
                        f"values in the '{class_}' column are not valid probabilities."
                        " All probabilities should be between 0 and 1, inclusive."
                    )
            return data

    return DataValidator


def validate_predictions(
    predictions: pd.DataFrame, data_schema: MulticlassClassificationSchema
) -> pd.DataFrame:
    """
    Validates the predictions using the provided schema.

    Args:
        predictions (pd.DataFrame): Predictions data to validate.
        data_schema (MulticlassClassificationSchema): An instance of
            inaryClassificationSchema.

    Returns:
        pd.DataFrame: The validated data.
    """
    DataValidator = get_predictions_validator(data_schema)
    try:
        validated_data = DataValidator(data=predictions)
        return validated_data.data
    except ValueError as exc:
        raise ValueError(f"Prediction data validation failed: {str(exc)}") from exc


if __name__ == "__main__":
    schema_dict = {
        "title": "test dataset",
        "description": "test dataset",
        "modelCategory": "multiclass_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {"name": "id", "description": "unique identifier."},
        "target": {
            "name": "target_field",
            "description": "some target desc.",
            "classes": ["A", "B", "C"],
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
                "name": "categorical_feature_1",
                "description": "some desc.",
                "dataType": "CATEGORICAL",
                "categories": ["X", "Y", "Z"],
                "nullable": True,
            },
        ],
    }
    schema_provider = MulticlassClassificationSchema(schema_dict)
    predictions = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "A": [0.9, 0.2, 0.8, 0.1, 0.85],
            "B": [0.05, 0.5, 0.1, 0.5, 0.05],
            "C": [0.05, 0.3, 0.1, 0.4, 0.1],
        }
    )

    validated_data = validate_predictions(predictions, schema_provider)
