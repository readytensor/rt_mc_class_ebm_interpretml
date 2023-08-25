import os

import pytest


@pytest.fixture
def performance_test_results_dir_path():
    tests_dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir_path = os.path.join(tests_dir_path, "test_results", "performance_tests")
    return results_dir_path


@pytest.fixture
def train_predict_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "train_predict_performance_results.csv"
    )
    return str(file_path)


@pytest.fixture
def inference_apis_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "inference_api_performance_results.csv"
    )
    return str(file_path)


@pytest.fixture
def docker_img_build_perf_results_path(performance_test_results_dir_path):
    file_path = os.path.join(
        performance_test_results_dir_path, "docker_img_build_performance_results.csv"
    )
    return str(file_path)
