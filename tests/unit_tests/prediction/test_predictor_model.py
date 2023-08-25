import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from src.prediction.predictor_model import (
    Classifier,
    evaluate_predictor_model,
    load_predictor_model,
    predict_with_model,
    save_predictor_model,
    train_predictor_model,
)


# Define the hyperparameters fixture
@pytest.fixture
def hyperparameters(default_hyperparameters):
    return default_hyperparameters


@pytest.fixture
def classifier(hyperparameters):
    """Define the classifier fixture"""
    return Classifier(**hyperparameters)


@pytest.fixture
def synthetic_data():
    """Define the synthetic dataset fixture"""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(X.shape[1])])
    y = pd.Series(y, name="target")
    train_X, train_y = X[:80], y[:80]
    test_X, test_y = X[80:], y[80:]
    return train_X, train_y, test_X, test_y


def test_build_model(default_hyperparameters):
    """
    Test if the classifier is created with the specified hyperparameters.
    """
    modified_hyperparameters = default_hyperparameters.copy()
    for key, value in modified_hyperparameters.items():
        value_type = type(value)
        if value_type == str:
            modified_hyperparameters[key] = f"{key}_test"
        elif value_type in [int, float]:
            modified_hyperparameters[key] = 42
    new_classifier = Classifier(**modified_hyperparameters)
    for param, value in modified_hyperparameters.items():
        assert getattr(new_classifier, param) == value


def test_build_model_without_hyperparameters(default_hyperparameters):
    """
    Test if the classifier is created with default hyperparameters when
    none are provided.
    """
    default_classifier = Classifier()

    # Check if the model has default hyperparameters
    for param, value in default_hyperparameters.items():
        assert getattr(default_classifier, param) == value


def test_fit_predict_evaluate(classifier, synthetic_data):
    """
    Test if the fit method trains the model correctly and if the predict and evaluate
    methods work as expected.
    """
    train_X, train_y, test_X, test_y = synthetic_data
    classifier.fit(train_X, train_y)
    predictions = classifier.predict(test_X)
    assert predictions.shape == test_y.shape
    assert np.array_equal(predictions, predictions.astype(bool))

    proba_predictions = classifier.predict_proba(test_X)
    assert proba_predictions.shape == (test_y.shape[0], 2)

    accuracy = classifier.evaluate(test_X, test_y)
    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1


def test_save_load(tmpdir, classifier, synthetic_data, hyperparameters):
    """
    Test if the save and load methods work correctly and if the loaded model has the
    same hyperparameters and predictions as the original.
    """

    train_X, train_y, test_X, test_y = synthetic_data
    classifier.fit(train_X, train_y)

    # Specify the file path
    model_dir_path = tmpdir.mkdir("model")

    # Save the model
    classifier.save(model_dir_path)

    # Load the model
    loaded_clf = Classifier.load(model_dir_path)

    # Check the loaded model has the same hyperparameters as the original classifier
    for param, value in hyperparameters.items():
        assert getattr(loaded_clf, param) == value

    # Test predictions
    predictions = loaded_clf.predict(test_X)
    assert np.array_equal(predictions, classifier.predict(test_X))

    proba_predictions = loaded_clf.predict_proba(test_X)
    assert np.array_equal(proba_predictions, classifier.predict_proba(test_X))

    # Test evaluation
    accuracy = loaded_clf.evaluate(test_X, test_y)
    assert accuracy == classifier.evaluate(test_X, test_y)


def test_accuracy_compared_to_logistic_regression(classifier, synthetic_data):
    """
    Test if the accuracy of the classifier is close enough to the accuracy of a
    baseline model like logistic regression.
    """
    train_X, train_y, test_X, test_y = synthetic_data

    # Fit and evaluate the classifier
    classifier.fit(train_X, train_y)
    classifier_accuracy = classifier.evaluate(test_X, test_y)

    # Fit and evaluate the logistic regression model
    baseline_model = LogisticRegression()
    baseline_model.fit(train_X, train_y)
    baseline_accuracy = baseline_model.score(test_X, test_y)

    # Set an acceptable difference in accuracy
    accuracy_threshold = -0.05

    # Check if the classifier's accuracy is close enough to the logistic
    # regression accuracy
    assert classifier_accuracy - baseline_accuracy > accuracy_threshold


def test_classifier_str_representation(classifier, hyperparameters):
    """
    Test the `__str__` method of the `Classifier` class.

    The test asserts that the string representation of a `Classifier` instance is
    correctly formatted and includes the model name and the correct hyperparameters.

    Args:
        classifier (Classifier): An instance of the `Classifier` class,
            created using the `hyperparameters` fixture.
        hyperparameters (dict): A dictionary of the hyperparameters used to initialize
            the `classifier`.

    Raises:
        AssertionError: If the string representation of `classifier` does not
            match the expected format or if it does not include the correct
            hyperparameters.
    """
    classifier_str = str(classifier)

    assert classifier.model_name in classifier_str
    for param in hyperparameters.keys():
        assert param in classifier_str


def test_train_predictor_model(synthetic_data, hyperparameters):
    """
    Test that the 'train_predictor_model' function returns a Classifier instance with
    correct hyperparameters.
    """
    train_X, train_y, _, _ = synthetic_data
    classifier = train_predictor_model(train_X, train_y, hyperparameters)

    assert isinstance(classifier, Classifier)
    for param, value in hyperparameters.items():
        assert getattr(classifier, param) == value


def test_predict_with_model(synthetic_data, hyperparameters):
    """
    Test that the 'predict_with_model' function returns predictions of correct size
    and type.
    """
    train_X, train_y, test_X, _ = synthetic_data
    classifier = train_predictor_model(train_X, train_y, hyperparameters)
    predictions = predict_with_model(classifier, test_X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[0] == test_X.shape[0]


def test_save_predictor_model(tmpdir, synthetic_data, hyperparameters):
    """
    Test that the 'save_predictor_model' function correctly saves a Classifierinstance
    to disk.
    """
    train_X, train_y, _, _ = synthetic_data
    model_dir_path = os.path.join(tmpdir, "model")
    classifier = train_predictor_model(train_X, train_y, hyperparameters)
    save_predictor_model(classifier, model_dir_path)
    assert os.path.exists(model_dir_path)
    assert len(os.listdir(model_dir_path)) >= 1


def test_untrained_save_predictor_model_fails(tmpdir, classifier):
    """
    Test that the 'save_predictor_model' function correctly raises  NotFittedError
    when saving an untrained classifier to disk.
    """
    with pytest.raises(NotFittedError):
        model_dir_path = os.path.join(tmpdir, "model")
        save_predictor_model(classifier, model_dir_path)


def test_load_predictor_model(tmpdir, synthetic_data, classifier, hyperparameters):
    """
    Test that the 'load_predictor_model' function correctly loads a Classifier
    instance from disk and that the loaded instance has the correct hyperparameters.
    """
    train_X, train_y, _, _ = synthetic_data
    classifier = train_predictor_model(train_X, train_y, hyperparameters)

    model_dir_path = os.path.join(tmpdir, "model")
    save_predictor_model(classifier, model_dir_path)

    loaded_clf = load_predictor_model(model_dir_path)
    assert isinstance(loaded_clf, Classifier)
    for param, value in hyperparameters.items():
        assert getattr(loaded_clf, param) == value


def test_evaluate_predictor_model(synthetic_data, hyperparameters):
    """
    Test that the 'evaluate_predictor_model' function returns an accuracy score of
    correct type and within valid range.
    """
    train_X, train_y, test_X, test_y = synthetic_data
    classifier = train_predictor_model(train_X, train_y, hyperparameters)
    accuracy = evaluate_predictor_model(classifier, test_X, test_y)

    assert isinstance(accuracy, float)
    assert 0 <= accuracy <= 1
