import json
import os

import numpy as np
import pandas as pd
import pytest

from src.hyperparameter_tuning.tuner import HyperParameterTuner, tune_hyperparameters


@pytest.fixture
def default_hyperparameters():
    """Create a default-hyperparameters fixture"""
    return {
        "hp_int": 1,
        "hp_float": 2.0,
        "hp_log_int": 5,
        "hp_log_float": 0.5,
        "hp_categorical": "a",
    }


@pytest.fixture
def default_hyperparameters_file_path(default_hyperparameters, tmpdir):
    """Fixture to create and save a sample default hyperparameters file for testing"""
    default_hyperparameters_fpath = tmpdir.join("default_hyperparameters.json")
    with open(default_hyperparameters_fpath, "w", encoding="utf-8") as file:
        json.dump(default_hyperparameters, file)
    return default_hyperparameters_fpath


@pytest.fixture
def hpt_specs():
    """Create a hpt-specs fixture"""
    return {
        "num_trials": 20,
        "hyperparameters": [
            {
                "type": "int",
                "search_type": "uniform",
                "name": "hp_int",
                "range_low": 0,
                "range_high": 10,
            },
            {
                "type": "real",
                "search_type": "uniform",
                "name": "hp_float",
                "range_low": 0.0,
                "range_high": 10.0,
            },
            {
                "type": "int",
                "search_type": "log-uniform",
                "name": "hp_log_int",
                "range_low": 1,
                "range_high": 10,
            },
            {
                "type": "real",
                "search_type": "log-uniform",
                "name": "hp_log_float",
                "range_low": 0.1,
                "range_high": 10.0,
            },
            {
                "type": "categorical",
                "search_type": None,
                "name": "hp_categorical",
                "categories": ["a", "b", "c"],
            },
        ],
    }


@pytest.fixture
def hpt_specs_file_path(hpt_specs, tmpdir):
    """Fixture to create and save a sample hyperparameters specs file for testing"""
    hpt_specs_fpath = tmpdir.join("hpt_specs.json")
    with open(hpt_specs_fpath, "w") as file:
        json.dump(hpt_specs, file)
    return hpt_specs_fpath


@pytest.fixture
def hpt_results_dir_path(tmpdir):
    """Create a hpt-results-file-path fixture"""
    return os.path.join(str(tmpdir), "hpt_outputs")


@pytest.fixture
def tuner(default_hyperparameters, hpt_specs, hpt_results_dir_path):
    """Create a tuner fixture"""
    return HyperParameterTuner(
        default_hyperparameters=default_hyperparameters,
        hpt_specs=hpt_specs,
        hpt_results_dir_path=hpt_results_dir_path,
    )


@pytest.fixture
def mock_data():
    train_X = np.random.rand(100, 1)
    train_y = np.random.randint(0, 2, 100)
    valid_X = np.random.rand(20, 1)
    valid_y = np.random.randint(0, 2, 20)
    return train_X, train_y, valid_X, valid_y


def test_init(default_hyperparameters, hpt_specs, hpt_results_dir_path):
    """Tests the `__init__` method of the `HyperParameterTuner` class.

    This test verifies that the `__init__` method correctly initializes the
    hyperparameter tuner object with the provided parameters.
    """
    tuner = HyperParameterTuner(
        default_hyperparameters, hpt_specs, hpt_results_dir_path
    )
    assert tuner.default_hyperparameters == default_hyperparameters
    assert tuner.hpt_specs == hpt_specs
    assert tuner.hpt_results_dir_path == hpt_results_dir_path
    assert tuner.is_minimize is True
    assert tuner.num_trials == hpt_specs.get("num_trials", 20)
    assert tuner.hyperparameter_names == [
        "hp_int",
        "hp_float",
        "hp_log_int",
        "hp_log_float",
        "hp_categorical",
    ]
    assert tuner.default_hyperparameter_vals == [1, 2.0, 5, 0.5, "a"]


def test_get_objective_func(mocker, tuner, mock_data, default_hyperparameters):
    """Tests the `_get_objective_func` method of the `HyperParameterTuner` class.

    This test verifies that the `_get_objective_func` method correctly returns
    a callable objective function for hyperparameter tuning.
    """
    mock_train_X, mock_train_y, mock_valid_X, mock_valid_y = mock_data

    mock_train = mocker.patch(
        "src.hyperparameter_tuning.tuner.train_predictor_model",
        return_value="mock_classifier",
    )
    mock_evaluate = mocker.patch(
        "src.hyperparameter_tuning.tuner.evaluate_predictor_model", return_value=0.8
    )

    objective_func = tuner._get_objective_func(
        mock_train_X, mock_train_y, mock_valid_X, mock_valid_y
    )
    result = objective_func([1, 2.0, 5, 0.5, "a"])
    mock_train.assert_called_once_with(
        mock_train_X, mock_train_y, default_hyperparameters
    )
    mock_evaluate.assert_called_once_with("mock_classifier", mock_valid_X, mock_valid_y)
    assert result == 0.8


def test_get_hpt_space(tuner):
    """Tests the `get_hpt_space` method of the `HyperParameterTuner` class.

    This test verifies that the `get_hpt_space` method correctly returns
    a list of hyperparameter space objects.
    """
    hpt_space = tuner._get_hpt_space()

    assert hpt_space[0].name == "hp_int"
    assert hpt_space[0].prior == "uniform"
    assert hpt_space[2].name == "hp_log_int"
    assert hpt_space[2].prior == "log-uniform"
    assert hpt_space[4].name == "hp_categorical"


def test_run_hyperparameter_tuning(mocker, tuner, mock_data):
    """
    Tests the `run_hyperparameter_tuning` method of the `HyperParameterTuner`
    class.

    This test verifies that the `run_hyperparameter_tuning` method correctly performs
    the hyperparameter tuning process and returns the best hyperparameters and score.
    """
    mock_train_X, mock_train_y, mock_valid_X, mock_valid_y = mock_data
    mock_gp_minimize = mocker.patch(
        "src.hyperparameter_tuning.tuner.gp_minimize",
        return_value="mock_optimizer_results",
    )
    mock_save_hpt_results = mocker.patch.object(tuner, "save_hpt_summary_results")
    mock_get_best_hps = mocker.patch.object(
        tuner, "get_best_hyperparameters", return_value="mock_best_hps"
    )

    result = tuner.run_hyperparameter_tuning(
        mock_train_X, mock_train_y, mock_valid_X, mock_valid_y
    )
    mock_gp_minimize.assert_called_once()
    mock_save_hpt_results.assert_called_once_with("mock_optimizer_results")
    mock_get_best_hps.assert_called_once_with("mock_optimizer_results")
    assert result == "mock_best_hps"


def test_get_best_hyperparameters(mocker, tuner):
    """
    Tests the `get_best_hyperparameters` method of the `HyperParameterTuner`
    class.

    This test verifies that the `get_best_hyperparameters` method correctly returns
    the best hyperparameters from the optimization results.
    """
    mock_optimizer_results = mocker.MagicMock()
    mock_optimizer_results.func_vals = [0.3, 0.2, 0.4]
    mock_optimizer_results.x_iters = [
        [1, 1.0, 1, 0.1, "a"],
        [2, 2.0, 2, 0.2, "b"],
        [3, 3.0, 3, 0.3, "c"],
    ]
    result = tuner.get_best_hyperparameters(mock_optimizer_results)
    assert result == {
        "hp_int": 2,
        "hp_float": 2.0,
        "hp_log_int": 2,
        "hp_log_float": 0.2,
        "hp_categorical": "b",
    }


def test_save_hpt_summary_results(mocker, tuner, hpt_results_dir_path):
    """
    Tests the `save_hpt_summary_results` method of the `HyperParameterTuner`
    class.

    This test verifies that the `save_hpt_summary_results` method correctly saves
    the hyperparameter tuning results to a file.
    """
    # Mock necessary parts
    mock_optimizer_results = mocker.MagicMock()
    mock_optimizer_results.func_vals = [0.3, 0.2, 0.4]
    mock_optimizer_results.x_iters = [
        [1, 1.0, 1, 0.1, "a"],
        [2, 2.0, 2, 0.2, "b"],
        [3, 3.0, 3, 0.3, "c"],
    ]

    # Run the function under test
    tuner.save_hpt_summary_results(mock_optimizer_results)

    # Check if the CSV file is created in the temporary directory
    assert len(os.listdir(hpt_results_dir_path)) == 1

    # Check the contents of the CSV file
    df = pd.read_csv(os.path.join(hpt_results_dir_path, "hpt_results.csv"))
    assert df.shape == (3, 7)
    assert (
        df.columns
        == [
            "trial_num",
            "hp_int",
            "hp_float",
            "hp_log_int",
            "hp_log_float",
            "hp_categorical",
            "metric_value",
        ]
    ).all()


def test_tune_hyperparameters(
    mocker,
    mock_data,
    hpt_results_dir_path,
    default_hyperparameters_file_path,
    default_hyperparameters,
    hpt_specs_file_path,
    hpt_specs,
):
    """Tests the `tune_hyperparameters` function.

    This test verifies that the `tune_hyperparameters` function correctly
    instantiates the `HyperParameterTuner` class with the right parameters
    and that the `run_hyperparameter_tuning` method is called with the correct
    arguments.
    """

    mock_train_X, mock_train_y, mock_valid_X, mock_valid_y = mock_data

    # Mock HyperParameterTuner
    mock_tuner = mocker.patch("src.hyperparameter_tuning.tuner.HyperParameterTuner")
    # Mock return value of run_hyperparameter_tuning method
    mock_tuner.return_value.run_hyperparameter_tuning.return_value = {
        "hp1": 1,
        "hp2": 2,
    }

    is_minimize = False

    # Call the function
    result = tune_hyperparameters(
        mock_train_X,
        mock_train_y,
        mock_valid_X,
        mock_valid_y,
        hpt_results_dir_path,
        is_minimize,
        default_hyperparameters_file_path,
        hpt_specs_file_path,
    )

    # Check the calls
    mock_tuner.assert_called_once_with(
        default_hyperparameters=default_hyperparameters,
        hpt_specs=hpt_specs,
        hpt_results_dir_path=hpt_results_dir_path,
        is_minimize=is_minimize,
    )
    mock_tuner.return_value.run_hyperparameter_tuning.assert_called_once_with(
        mock_train_X, mock_train_y, mock_valid_X, mock_valid_y
    )

    # Check the result
    assert result == {"hp1": 1, "hp2": 2}
