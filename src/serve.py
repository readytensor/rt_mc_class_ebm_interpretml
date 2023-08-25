from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import paths
from data_models.infer_request_model import get_inference_request_body_model
from logger import log_error
from serve_utils import (
    ModelResources,
    combine_predictions_response_with_explanations,
    generate_unique_request_id,
    get_model_resources,
    logger,
    transform_req_data_and_make_predictions,
)
from prediction.predictor_model import explain_with_model

def create_app(model_resources):

    app = FastAPI()

    @app.get("/ping")
    async def ping() -> dict:
        """GET endpoint that returns a message indicating the service is running.

        Returns:
            dict: A dictionary with a "message" key and "Pong!" value.
        """
        logger.info("Received ping request. Service is healthy...")
        return {"message": "Pong!"}

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Any, exc: RequestValidationError
    ) -> JSONResponse:
        """
        Handle validation errors for FastAPI requests.

        Args:
            request (Any): The FastAPI request instance.
            exc (RequestValidationError): The RequestValidationError instance.
        Returns:
            JSONResponse: A JSON response with the error message and a 400 status code.
        """
        err_msg = "Validation error with request data."
        # Log the error
        logger.error(f"{err_msg} Error: {str(exc)}")
        # Log the error to the separate logging file
        log_error(message=err_msg, error=exc, error_fpath=paths.SERVE_ERROR_FILE_PATH)
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc), "predictions": None},
        )

    InferenceRequestBodyModel = get_inference_request_body_model(
        model_resources.data_schema
    )

    @app.post("/infer", tags=["inference"], response_class=JSONResponse)
    async def infer(request: InferenceRequestBodyModel) -> dict:
        """POST endpoint that takes input data as a JSON object and returns
        predicted class probabilities.

        Args:
            request (InferenceRequestBodyModel): The request body containing the
                input data.

        Raises:
            HTTPException: If there is an error during inference.

        Returns:
            dict: A dictionary with "status", "message", and "predictions" keys.
        """
        try:
            request_id = generate_unique_request_id()
            logger.info(f"Responding to inference request. Request id: {request_id}")
            logger.info("Starting predictions...")
            data = pd.DataFrame.from_records(request.dict()["instances"])
            _, predictions_response = await transform_req_data_and_make_predictions(
                data, model_resources, request_id
            )
            logger.info("Returning predictions...")
            return predictions_response
        except Exception as exc:
            err_msg = f"Error occurred during inference. Request id: {request_id}"
            # Log the error to the general logging file 'serve.log'
            logger.error(f"{err_msg} Error: {str(exc)}")
            # Log the error to the separate logging file 'serve-error.log'
            log_error(
                message=err_msg, error=exc, error_fpath=paths.SERVE_ERROR_FILE_PATH
            )
            raise HTTPException(
                status_code=500, detail=f"{err_msg} Error: {str(exc)}"
            ) from exc

    @app.post("/explain", tags=["explanations", "XAI"], response_class=JSONResponse)
    async def explain(request: InferenceRequestBodyModel) -> dict:
        """POST endpoint that takes input data as a JSON object and returns
        the predicted class probabilities with explanations.

        Args:
            request (InferenceRequestBodyModel): The request body containing
                the input data.

        Raises:
            HTTPException: If there is an error during inference.

        Returns:
            dict: A dictionary with "status", "message", "timestamp", "requestId",
                "targetClasses", "targetDescription", "predictions",
                and "explanationMethod" keys.
        """
        try:
            request_id = generate_unique_request_id()
            logger.info(f"Responding to explanation request. Request id: {request_id}")
            logger.info("Starting prediction...")
            data = pd.DataFrame.from_records(request.dict()["instances"])
            (
                transformed_data,
                predictions_response,
            ) = await transform_req_data_and_make_predictions(
                data, model_resources, request_id
            )
            logger.info("Generating explanations...")
            explanations = explain_with_model(
                explainer=model_resources.predictor_and_explainer,
                data=transformed_data,
                class_names=model_resources.data_schema.target_classes,
            )
            # print(explanations)
            logger.info("Combining predictions and explanations...")
            predictions_response = combine_predictions_response_with_explanations(
                predictions_response=predictions_response, explanations=explanations
            )
            logger.info("Returning explanations response...")
            return predictions_response
        except Exception as exc:
            err_msg = f"Error occurred during explanations. Request id: {request_id}"
            # Log the error to the general logging file 'serve.log'
            logger.error(f"{err_msg} Error: {str(exc)}")
            # Log the error to the separate logging file 'serve-error.log'
            log_error(
                message=err_msg, error=exc, error_fpath=paths.SERVE_ERROR_FILE_PATH
            )
            raise HTTPException(
                status_code=500, detail=f"{err_msg} Error: {str(exc)}"
            ) from exc

    return app


def create_and_run_app(model_resources: ModelResources):
    """Create and run Fastapi app for inference service

    Args:
        model (ModelResources, optional): The model resources instance.
            Defaults to load model resources from paths defined in paths.py.
    """
    app = create_app(model_resources)
    print("Starting service. Listening on port 8080.")
    uvicorn.run(app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    create_and_run_app(model_resources=get_model_resources())
