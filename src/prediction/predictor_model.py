import os
import warnings
from typing import Optional, List

import joblib
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier as EBC
from sklearn.exceptions import NotFittedError
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

PREDICTOR_FILE_NAME = "predictor.joblib"
GLOBAL_EXPLANATIONS_FILE_NAME = "global_explanations.csv"
GLOBAL_EXPLANATIONS_CHART_FILE_NAME = "global_explanations.png"


class Classifier:
    """A wrapper class for the Explainable Boosting Machine (EBM) classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "Explainable Boosting Machine (EBM) Classifier"
    explanation_method = "Glassbox Model: Explainable Boosting Machine (Interpret)"

    def __init__(
        self,
        min_samples_leaf: Optional[int] = 2,
        learning_rate: Optional[float] = 1e-2,
        **kwargs,
    ):
        """Construct a new Explainable Boosting Machine (EBM) classifier.

        Args:
            min_samples_leaf (int, optional): The minimum number of samples required
                to split an internal node.
                Defaults to 2.
            learning_rate (int, optional): The minimum number of samples required
                to be at a leaf node.
                Defaults to 0.01.
        """
        self.min_samples_leaf = int(min_samples_leaf)
        self.learning_rate = float(learning_rate)
        # build model later in `fit` because we need feature names to instantiate
        self.feature_names = None
        self.model = None
        self._is_trained = False

    def build_model(self) -> EBC:
        """Build a new classifier."""
        model = EBC(
            min_samples_leaf=self.min_samples_leaf,
            learning_rate=self.learning_rate,
            feature_names=self.feature_names,
            early_stopping_rounds=20,
            # interactions=0,
            random_state=0,
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.feature_names = train_inputs.columns.tolist()
        self.model = self.build_model()
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict_proba(inputs)

    def explain_local(self, X, class_names):
        local_explanations = self.model.explain_local(X=X, y=None)
        explanations=[]
        for i in range(len(X)):
            sample_exp = local_explanations.data(i)
            sample_expl_dict = {
                "baseline": np.round(sample_exp["extra"]["scores"][0], 5),
                "feature_scores": {
                    f: np.round(s, 5)
                    for f, s in zip(sample_exp["names"], sample_exp["scores"])
                }
            }
            explanations.append(sample_expl_dict)
        return {
            "explanation_method": self.explanation_method,
            "explanations": explanations,
        }

    def explain_global(self,):
        return self.model.explain_global(name=self.model_name)

    def _save_global_explanations(self, model_dir_path):
        global_explanations = self.model.explain_global()
        data = global_explanations.data()
        df = pd.DataFrame()
        # create list of feature names - "extra" contains the intercept
        # read documentation of interpret ml global_explanations for details
        df["feature"] = data["names"] + [data["names"][0]]
        df["score"] = data["scores"] + [data["scores"][0]]
        df.sort_values(by=["score"], inplace=True, ascending=False)
        df.to_csv(
            os.path.join(model_dir_path, GLOBAL_EXPLANATIONS_FILE_NAME),
            index=False,
            float_format="%.4f",
        )
        self._save_plot_of_explanations(df["score"], df["feature"], model_dir_path)

    def _save_plot_of_explanations(self, vals, labels, model_dir_path):
        height = 2.0 + len(vals) * 0.3
        colors = ["rosybrown" if x < 0 else "steelblue" for x in vals]
        plt.figure(figsize=(18, height), dpi=80)
        plt.barh(labels, vals, color=colors)
        plt.xlabel("score", fontsize=16)
        plt.ylabel("feature", fontsize=16)
        plt.yticks(labels, fontsize=14)
        plt.xticks(fontsize=14)
        plt.title("Global Feature Impact", fontdict={"size": 20})
        plt.grid(linestyle="--", alpha=0.5)
        plt.savefig(os.path.join(model_dir_path, GLOBAL_EXPLANATIONS_CHART_FILE_NAME))

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        self._save_global_explanations(model_dir_path=model_dir_path)

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"min_samples_leaf: {self.min_samples_leaf}, "
            f"learning_rate: {self.learning_rate})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def explain_with_model(
    explainer: Classifier, data: pd.DataFrame, class_names: List[str]
) -> np.ndarray:
    """
    Make predictions and explain them for the given data.

    Args:
        classifier (Classifier): The classifier model.
                                Note this classifier is also an explainer.
        data (pd.DataFrame): The input data.
        class_names List[str]: List of class names as strings

    Returns:
        np.ndarray: Explanations.
    """
    return explainer.explain_local(data, class_names)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
