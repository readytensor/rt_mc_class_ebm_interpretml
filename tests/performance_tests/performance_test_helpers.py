import csv
import os
import random
import shutil
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def set_seeds_for_data_gen():
    """Set random seeds for reproducibility."""
    random.seed(42)
    np.random.seed(42)


def delete_dir_if_exists(path: str):
    """Removes a directory if it exists.

    Args:
        path (str): Path to the directory to be removed.
    """
    if os.path.exists(path) and os.path.isdir(path):
        shutil.rmtree(path)


def delete_file_if_exists(path: str):
    """Removes a file if it exists.

    Args:
        path (str): Path to the file to be removed.
    """
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)


def generate_schema_and_data(rows: int, columns: int) -> Tuple[Dict, pd.DataFrame]:
    """Generates a schema and data for testing.

    Args:
        rows (int): Number of rows for the data.
        columns (int): Number of columns for the data.

    Returns:
        Tuple[Dict, pd.DataFrame]: The generated schema and data.
    """
    # define schema
    schema = {
        "title": "Generated dataset",
        "description": "ID",
        "modelCategory": "multiclass_classification",
        "schemaVersion": 1.0,
        "inputDataFormat": "CSV",
        "id": {
            "name": "id",
            "description": "A unique identifier for each record in the dataset.",
        },
        "target": {
            "name": "target",
            "description": "A multiclass variable indicating the class \
                (class_0 = No, class_1 = Yes).",
            "classes": ["class_0", "class_1"],
        },
        "features": [],
    }

    # create features in schema
    for i in range(1, columns + 1):
        feature = {
            "name": f"numeric_feature_{i}",
            "description": f"Numeric feature {i}",
            "dataType": "NUMERIC",
            "example": round(random.uniform(0, 100), 2),
            "nullable": bool(random.getrandbits(1)),
        }
        schema["features"].append(feature)

    # create dataframe
    data_dict = {}
    for feature in schema["features"]:
        # if feature is nullable, create an array with some nulls
        if feature["nullable"]:
            data = np.where(np.random.rand(rows) > 0.05, np.random.rand(rows), np.nan)
        else:
            data = np.random.rand(rows)

        data_dict[feature["name"]] = data

    # add id and target
    data_dict["id"] = np.arange(rows)
    data_dict["target"] = np.random.choice(["class_0", "class_1"], rows)

    df = pd.DataFrame(data_dict)

    return schema, df


def store_results_to_csv(output_file_path: str, headers: Tuple, results: Tuple):
    """Writes the test results to a CSV file.

    Args:
        output_file_path (str): Path to the output CSV file.
        headers (Tuple): Tuple containing the headers to be written.
        results (Tuple): Tuple containing the results to be written.
                         This must be same length as headers
    """
    file_exists = os.path.isfile(output_file_path)
    with open(output_file_path, "a", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile, delimiter=",", lineterminator="\n", fieldnames=headers
        )

        # Write header if file didn't exist at the start
        if not file_exists:
            writer.writeheader()

        # Write the performance metrics for this test
        print("results", results)
        results_row = {k: v for k, v in zip(headers, results)}
        writer.writerow(results_row)
