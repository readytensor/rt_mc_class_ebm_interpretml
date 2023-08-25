import os
from time import perf_counter

import pytest

from tests.performance_tests.performance_test_helpers import (
    delete_file_if_exists,
    store_results_to_csv,
)

# This global variable will be True if no tests have been run yet, and False otherwise
FIRST_TEST = True

ENDPOINTS = ["/infer", "/explain"]


@pytest.mark.slow
@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_api_endpoint_performance(
    app, sample_request_data, inference_apis_perf_results_path, endpoint
):
    """
    Performance test for the FastAPI application endpoints.
    """
    global FIRST_TEST

    # If this is the first test and the results file already exists, delete it
    if FIRST_TEST and os.path.isfile(inference_apis_perf_results_path):
        delete_file_if_exists(inference_apis_perf_results_path)
        FIRST_TEST = False

    # Run the API and record metrics
    num_requests, response_time, throughput = run_inference_api_and_record_metrics(
        app, endpoint, sample_request_data, 5
    )
    # update num_requests to 50 or higher for more rigorous testing

    # Store API performance metrics
    store_results_to_csv(
        inference_apis_perf_results_path,
        (
            "endpoint",
            "num_requests_tested",
            "avg_response_time_secs",
            "throughput_reqs_per_sec",
        ),
        (
            endpoint.lstrip("/"),
            num_requests,
            round(response_time, 4),
            round(throughput, 4),
        ),
    )


def run_inference_api_and_record_metrics(
    app, endpoint, sample_request_data, num_requests=50
):
    """Runs inference API and records the response time and throughput metrics.

    Args:
        app (TestClient): The FastAPI TestClient instance.
        endpoint (str): The API endpoint to test.
        sample_request_data (dict): The sample request data for the API.
        num_requests (int): The number of requests to send for throughput measurement.

    Returns:
        tuple: The average response time in seconds and the throughput in requests
            per second.
    """
    # For computing both response time and throughput, send multiple requests and
    # measure the time
    start_time = perf_counter()
    for _ in range(num_requests):
        response = app.post(endpoint, json=sample_request_data)
        assert response.status_code == 200
    total_time = perf_counter() - start_time

    # Compute the average response time and throughput
    avg_response_time = total_time / num_requests
    throughput = num_requests / total_time

    return num_requests, avg_response_time, throughput
