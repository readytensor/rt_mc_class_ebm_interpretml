import os
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from shap import Explainer

from prediction.predictor_model import predict_with_model
from utils import read_json_as_dict

EXPLAINER_FILE_NAME = "explainer.joblib"


class ClassificationExplainer:
    """Shap Explainer class for classification models"""

    EXPLANATION_METHOD = "Shap"

    def __init__(
        self, max_local_explanations: int = 5, max_saved_train_data_length: int = 10000
    ):
        self.max_local_explanations = max_local_explanations
        self.max_saved_train_data_length = max_saved_train_data_length
        self._shap_explainer = None
        self._explainer_data = None

    def fit(self, train_data: pd.DataFrame, **kwargs) -> None:
        """Fit the explainer to the training data.

        Args:
            train_data (pd.DataFrame): Training data to use for explainer.
        """
        # We only save the data, but we dont fit explainer
        # until we need it to get explanations.
        if train_data.shape[0] > self.max_saved_train_data_length:
            data = train_data.sample(self.max_saved_train_data_length, replace=False)
        else:
            data = train_data.copy()
        self._explainer_data = data

    def _build_explainer(self, predictor_model, class_names):
        """Build shap explainer

        Args:
            predictor_model (Any): A trained predictor model
            class_names List[str]: List of class names

        Returns:
            'Explainer': instance of shap Explainer from shap library
        """
        return Explainer(
            model=lambda instances: predict_with_model(
                predictor_model, instances, return_probs=True
            ),
            masker=self._explainer_data,
            algorithm="auto",
            output_names=class_names,
            seed=0,
        )

    def get_explanations(
        self,
        instances_df: pd.DataFrame,
        predictor_model: Any,
        class_names: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get local explanations for the given instances.

        Args:
            instances_df (pd.DataFrame): Instances to explain predictions
            predictor_model (Any): A trained predictor model
            class_names List[str]: List of class names

        Returns:
            Dict[str, Any]: Explanations returned in a dictionary
        """
        # limit explanations to at most self.max_local_explanations
        instances_df = instances_df.head(self.max_local_explanations)

        if self._shap_explainer is None:
            self._shap_explainer = self._build_explainer(predictor_model, class_names)

        explanations = []
        shap_values = self._shap_explainer(instances_df)
        for row_num in range(len(instances_df)):
            feature_scores = {}
            for f_num, feature in enumerate(shap_values.feature_names):
                feature_scores[feature] = list(
                    np.round(shap_values.values[row_num][f_num], 5)
                )

            explanations.append(
                {
                    "baseline": list(np.round(shap_values.base_values[row_num], 5)),
                    "featureScores": feature_scores,
                }
            )

        return {
            "explanation_method": self.EXPLANATION_METHOD,
            "explanations": explanations,
        }

    def save(self, file_path: Path) -> None:
        """Save the explainer to a file."""
        with open(file_path, "wb") as file:
            joblib.dump(self, file)

    @classmethod
    def load(cls, file_path: Path) -> "ClassificationExplainer":
        """Load the explainer from a file."""
        with open(file_path, "rb") as file:
            loaded_explainer = joblib.load(file)
        return loaded_explainer


def fit_and_save_explainer(
    train_data: pd.DataFrame, explainer_config_file_path: str, save_dir_path: str
) -> None:
    """
    Fit the explainer to the training data and save it to a file.

    Args:
        train_data (pd.DataFrame): pandas DataFrame of training data
        explainer_config_file_path (str): Path to the explainer configuration file.
        save_dir_path (str): Dir path where explainer should be saved.

    Returns:
        ClassificationExplainer: Instance of ClassificationExplainer
    """
    explainer_config = read_json_as_dict(explainer_config_file_path)
    explainer = ClassificationExplainer(
        max_local_explanations=explainer_config["max_local_explanations"],
        max_saved_train_data_length=explainer_config["max_saved_train_data_length"],
    )
    explainer.fit(train_data)
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    explainer.save(os.path.join(save_dir_path, EXPLAINER_FILE_NAME))
    return explainer


def load_explainer(save_dir_path: str) -> Any:
    """
    Load the explainer from a file.

    Args:
        save_dir_path (str): Dir path where explainer is saved.

    Returns:
        ClassificationExplainer: Instance of ClassificationExplainer
    """
    return ClassificationExplainer.load(
        os.path.join(save_dir_path, EXPLAINER_FILE_NAME)
    )


def get_explanations_from_explainer(
    instances_df: pd.DataFrame,
    explainer: ClassificationExplainer,
    predictor_model: Any,
    class_names: List[str],
) -> Dict[str, Any]:
    """Get explanations for the given instances_df.

    Args:
        instances_df (pd.DataFrame): instances to explain predictions
        explainer (ClassificationExplainer): Instance of
                    ClassificationExplainer
        predictor_model (Any): A trained predictor model
        class_names List[str]: List of class names as strings

    Returns:
        Dict[str, Any]: Explanations returned in a dictionary
    """
    return explainer.get_explanations(
        instances_df, predictor_model, class_names=class_names
    )
