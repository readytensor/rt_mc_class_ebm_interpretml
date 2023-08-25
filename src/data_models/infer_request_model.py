from typing import List, Union

from pydantic import BaseModel, Field, create_model, validator

from schema.data_schema import MulticlassClassificationSchema


def create_instance_model(schema: MulticlassClassificationSchema) -> BaseModel:
    """
    Creates a dynamic Pydantic model for instance validation based on the schema.

    Args:
        schema (MulticlassClassificationSchema): The multiclass classification schema.

    Returns:
        BaseModel: The dynamically created Pydantic model.
    """
    fields = {schema.id: (str, Field(..., example="some_id_123"))}

    for feature in schema.numeric_features:
        example_value = schema.get_example_value_for_feature(feature)
        if schema.is_feature_nullable(feature):
            field_type = Union[float, int, None]
        else:
            field_type = Union[float, int]
        fields[feature] = (field_type, Field(..., example=example_value))

    for feature in schema.categorical_features:
        example_value = schema.get_example_value_for_feature(feature)
        if schema.is_feature_nullable(feature):
            field_type = Union[str, None]
        else:
            field_type = str
        fields[feature] = (field_type, Field(..., example=example_value))

    return create_model("Instance", **fields)


def get_inference_request_body_model(
    schema: MulticlassClassificationSchema,
) -> BaseModel:
    """
    Creates a dynamic Pydantic model for the inference request body validation based
    on the schema.

    It ensures that the request body contains a list of instances, each of which is a
    dictionary representing a data instance with all the required numerical and
    categorical features as specified in the schema.

    Args:
        schema (MulticlassClassificationSchema): The multiclass classification schema.

    Returns:
        BaseModel: The dynamically created Pydantic model.
    """
    InstanceModel = create_instance_model(schema)

    class InferenceRequestBody(BaseModel):
        """
        InferenceRequestBody is a Pydantic model for validating the request body of an
            inference endpoint.

        The following validations are performed on the request data:
            - The request body contains a key 'instances' with a list of dictionaries
                as its value.
            - The list is not empty (i.e., at least one instance must be provided).
            - Each instance contains the ID field whose name is defined in the
                schema file.
            - Each instance contains all the required numerical and categorical
                features as defined in the schema file.
            - Values for each feature in each instance are of the correct data type.
              Values are allowed to be null (i.e., missing) if the feature is specified
                as nullable in the schema.
              Non-nullable features must have non-null values.
            - For categorical features, the given value must be one of the categories
                as defined in the schema file.

        Attributes:
            instances (List[Instance_Model]): A list of data instances to be validated.
        """

        instances: List[InstanceModel] = Field(..., min_items=1)

        @validator("instances", pre=True, each_item=True, allow_reuse=True)
        def validate_non_nullable_features(cls, instance):
            """
            Validates that non-nullable features must have non-null values.
            """
            for feature, value in instance.items():
                if (
                    feature in schema.features
                    and not schema.is_feature_nullable(feature)
                    and value is None
                ):
                    raise ValueError(
                        f"Feature `{feature}` is non-nullable. "
                        f"Given null value is not allowed."
                    )

            return instance

        @validator("instances", pre=True, each_item=True, allow_reuse=True)
        def validate_categorical_features(cls, instance):
            """
            Validates that the value of a categorical feature is one of the allowed
            values as defined in the schema file.
            """
            for feature, value in instance.items():
                if feature in schema.categorical_features:
                    categories = [
                        str(c)
                        for c in schema.get_allowed_values_for_categorical_feature(
                            feature
                        )
                    ]
                    if value is not None and str(value) not in categories:
                        raise ValueError(
                            f"Value '{value}' not allowed for '{feature}'."
                            f"Allowed values: {categories}"
                        )
            return instance

    return InferenceRequestBody
