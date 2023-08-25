# Explainable Boosting Machine (EBM) Multiclass Classifier with Shapley Explanations

## Project Description

This repository is a dockerized implementation of the re-usable multiclass classifier model. It is implemented in flexible way so that it can be used with any multiclass classification dataset with the use of CSV-formatted data, and a JSON-formatted data schema file. The main purpose of this repository is to provide a complete example of a machine learning model implementation that is ready for deployment.
The following are the requirements for using your data with this model:

- The data must be in CSV format.
- The number of rows must not exceed 20,000. Number of columns must not exceed 200. The model may function with larger datasets, but it has not been performance tested on larger datasets.
- The number of target classes (labels) must be between 3 and 20.
- Features must be one of the following two types: NUMERIC or CATEGORICAL. Other data types are not supported. Note that CATEGORICAL type includes boolean type.
- The train and test (or prediction) files must contain an ID field. The train data must also contain a target field.
- The data need not be preprocessed because the implementation already contains logic to handle missing values, categorical features, outliers, and scaling.

---

Here are the highlights of this implementation: <br/>

- A flexible preprocessing pipeline built using **SciKit-Learn** and **feature-engine**. Transformations include missing value imputation, categorical encoding, outlier removal, feature selection, and feature scaling. <br/>
- A **Explainable Boosting Machine (EBM) Multiclass Classifier** algorithm built using **InterpretML** library.
- Hyperparameter-tuning using **scikit-optimize**
- **FASTAPI** inference service for online inferences.
  Additionally, the implementation contains the following features:
- **Data Validation**: Pydantic data validation is used for the schema, training and test files, as well as the inference request data.
- **Error handling and logging**: Python's logging module is used for logging and key functions include exception handling.
- **Testing**: Comprehensive set of unit, integration, coverage and performance tests using **pytest** and **pytest-cov**.
- **Static code analysis**: Code quality tests are implemented using **flake8**, **black**, **isort**, **safety**, and **radon**.
- **Test automation**: Tox is used for test automation.

## Project Structure

The following is the directory structure of the project:

- **`examples/`**: This directory contains example files for a smoke_test dataset. This is a simple (toy) dataset. Three files are included: `smoke_test_schema.json`, `smoke_test_train.csv` and `smoke_test_test.csv`. You can place these files in the `inputs/schema`, `inputs/data/training` and `inputs/data/testing` folders, respectively.
- **`model_inputs_outputs/`**: This directory contains files that are either inputs to, or outputs from, the model. When running the model locally (i.e. without using docker), this directory is used for model inputs and outputs. This directory is further divided into:
  - **`/inputs/`**: This directory contains all the input files for this project, including the `data` and `schema` files. The `data` is further divided into `testing` and `training` subsets.
  - **`/model/artifacts/`**: This directory is used to store the model artifacts, such as trained models and their parameters.
  - **`/outputs/`**: The outputs directory contains sub-directories for error logs, and hyperparameter tuning outputs, and prediction results.
- **`requirements/`**: This directory contains the requirements files. We have multiple requirements files for different purposes:
  - `requirements.txt` for the main code in the `src` directory
  - `requirements_quality.txt` for dependencies related to code quality, safety, formatting and style checks.
  - `requirements_text.txt` for dependencies required to run tests in the `tests` directory.
- **`src/`**: This directory holds the source code for the project. It is further divided into various subdirectories:
  - **`config/`**: for configuration files for data preprocessing, model hyperparameters, hyperparameter tuning-configuration specs, paths, etc.
  - **`data_models/`**: for data models for input validation including the schema, training and test files, and the inference request data. It also contains the data model for the batch prediction results.
  - **`schema/`**: for schema handler script. This script contains the class that provides helper getters/methods for the data schema.
  - **`preprocessing/`**: for data preprocessing scripts including the feature and target encoding/transformations. We use **Scikit-Learn** and **feature-engine** for preprocessing.
  - **`prediction/`**: Scripts for the Explainable Boosting Machine (EBM) Multiclass Classifier model implemented using **InterpretML** library.
  - **`hyperparameter_tuning/`**: for hyperparameter-tuning (HPT) functionality implemented using **Scikit-Optimize** for the model.
  - **`serve.py`**: This script is used to serve the model as a REST API using **FastAPI**. It loads the artifacts and creates a FastAPI server to serve the model. It provides 3 endpoints: `/ping`, `/infer`, and `/explain`. The `/ping` endpoint is used to check if the server is running. The `/infer` endpoint is used to make predictions. The `/explain` endpoint is used to get local explanations for the predictions.
  - **`serve_utils.py`**: This script contains utility functions used by the `serve.py` script.
  - **`logger.py`**: This script contains the logger configuration using **logging** module.
  - **`train.py`**: This script is used to train the model. It loads the data, preprocesses it, trains the model, and saves the artifacts in the path `./model_inputs_outputs/model/artifacts/`. It also saves a SHAP explainer object in the path `./model/artifacts/`. When the train task is run with a flag to perform hyperparameter tuning, it also saves the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.
  - **`predict.py`**: This script is used to run batch predictions using the trained model. It loads the artifacts and creates and saves the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
  - **`utils.py`**: This script contains utility functions used by the other scripts.
- **`tests/`**: This directory contains all the tests for the project and associated resources and results.
  - **`integration_tests/`**: This directory contains the integration tests for the project. We cover four main workflows: data preprocessing, training, prediction, and inference service.
  - **`performance_tests/`**: This directory contains performance tests for the training and batch prediction workflows in the script `test_train_predict.py`. It also contains performance tests for the inference service workflow in the script `test_inference_apis.py`. Helper functions are defined in the script `performance_test_helpers.py`. Fixtures and other setup are contained in the script `conftest.py`.
  - **`test_results/`**: This folder contains the results for the performance tests. These are persisted to disk for later analysis.
  - **`unit_tests/`**: This folder contains all the unit tests for the project. It is further divided into subdirectories mirroring the structure of the `src` folder. Each subdirectory contains unit tests for the corresponding script in the `src` folder.
- **`tmp/`**: This directory is used for storing temporary files which are not necessary to commit to the repository.
- **`.dockerignore`**: This file specifies the files and folders that should be ignored by Docker.
- **`.gitignore`**: This file specifies the files and folders that should be ignored by Git.
- **`docker-compose.yaml`**: This file is used to define the services that make up the application. It is used by Docker Compose to run the application.
- **`Dockerfile`**: This file is used to build the Docker image for the application.
- **`entry_point.sh`**: This file is used as the entry point for the Docker container. It is used to run the application. When the container is run using one of the commands `train`, `predict` or `serve`, this script runs the corresponding script in the `src` folder to execute the task.
- **`fix_line_endings.sh`**: This script is used to fix line endings in the project files. It is used to ensure that the project files have the correct line endings when the project is run on Windows.
- **`LICENSE`**: This file contains the license for the project.
- **`pytest.ini`**: This is the configuration file for pytest, the testing framework used in this project.
- **`README.md`**: This file (this particular document) contains the documentation for the project, explaining how to set it up and use it.
- **`tox.ini`**: This is the configuration file for tox, the primary test runner used in this project.

## Usage

In this section we cover the following:

- How to prepare your data for training and inference
- How to run the model implementation locally (without Docker)
- How to run the model implementation with Docker
- How to use the inference service (with or without Docker)

### Preparing your data

- If you plan to run this model implementation on your own multiclass classification dataset, you will need your training and testing data in a CSV format. Also, you will need to create a schema file as per the Ready Tensor specifications. The schema is in JSON format, and it's easy to create. You can use the example schema file provided in the `examples` directory as a template.

### To run locally (without Docker)

- Create your virtual environment and install dependencies listed in `requirements.txt` which is inside the `requirements` directory.
- Move the three example files (`smoke_test_schema.json`, `smoke_test_train.csv` and `smoke_test_test.csv`) in the `examples` directory into the `./model_inputs_outputs/inputs/schema`, `./model_inputs_outputs/inputs/data/training` and `./model_inputs_outputs/inputs/data/testing` folders, respectively (or alternatively, place your custom dataset files in the same locations).
- Run the script `src/train.py` to train the random forest classifier model. This will save the model artifacts, including the preprocessing pipeline and label encoder, in the path `./model_inputs_outputs/model/artifacts/`. If you want to run with hyperparameter tuning then include the `-t` flag. This will also save the hyperparameter tuning results in the path `./model_inputs_outputs/outputs/hpt_outputs/`.
- Run the script `src/predict.py` to run batch predictions using the trained model. This script will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `./model_inputs_outputs/outputs/predictions/`.
- Run the script `src/serve.py` to start the inference service, which can be queried using the `/ping`, `/infer` and `/explain` endpoints. The service runs on port 8080.

### To run with Docker

1. Set up a bind mount on host machine: It needs to mirror the structure of the `model_inputs_outputs` directory. Place the train data file in the `model_inputs_outputs/inputs/data/training` directory, the test data file in the `model_inputs_outputs/inputs/data/testing` directory, and the schema file in the `model_inputs_outputs/inputs/schema` directory.
2. Build the image. You can use the following command: <br/>
   `docker build -t classifier_img .` <br/>
   Here `classifier_img` is the name given to the container (you can choose any name).
3. Note the following before running the container for train, batch prediction or inference service:
   - The train, batch predictions tasks and inference service tasks require a bind mount to be mounted to the path `/opt/model_inputs_outputs/` inside the container. You can use the `-v` flag to specify the bind mount.
   - When you run the train or batch prediction tasks, the container will exit by itself after the task is complete. When the inference service task is run, the container will keep running until you stop or kill it.
   - When you run training task on the container, the container will save the trained model artifacts in the specified path in the bind mount. This persists the artifacts even after the container is stopped or killed.
   - When you run the batch prediction or inference service tasks, the container will load the trained model artifacts from the same location in the bind mount. If the artifacts are not present, the container will exit with an error.
   - The inference service runs on the container's port **8080**. Use the `-p` flag to map a port on local host to the port 8080 in the container.
   - Container runs as user 1000. Provide appropriate read-write permissions to user 1000 for the bind mount. Please follow the principle of least privilege when setting permissions. The following permissions are required:
     - Read access to the `inputs` directory in the bind mount. Write or execute access is not required.
     - Read-write access to the `outputs` directory and `model` directories. Execute access is not required.
4. You can run training with or without hyperparameter tuning:
   - To run training without hyperparameter tuning (i.e. using default hyperparameters), run the container with the following command container: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train` <br/>
     where `classifier_img` is the name of the container. This will train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount.
   - To run training with hyperparameter tuning, issue the command: <br/>
     `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img train -t` <br/>
     This will tune hyperparameters,and used the tuned hyperparameters to train the model and save the artifacts in the `model_inputs_outputs/model/artifacts` directory in the bind mount. It will also save the hyperparameter tuning results in the `model_inputs_outputs/outputs/hpt_outputs` directory in the bind mount.
5. To run batch predictions, place the prediction data file in the `model_inputs_outputs/inputs/data/testing` directory in the bind mount. Then issue the command: <br/>
   `docker run -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img predict` <br/>
   This will load the artifacts and create and save the predictions in a file called `predictions.csv` in the path `model_inputs_outputs/outputs/predictions/` in the bind mount.
6. To run the inference service, issue the following command on the running container: <br/>
   `docker run -p 8080:8080 -v <path_to_mount_on_host>/model_inputs_outputs:/opt/model_inputs_outputs classifier_img serve` <br/>
   This starts the service on port 8080. You can query the service using the `/ping`, `/infer` and `/explain` endpoints. More information on the requests/responses on the endpoints is provided below.

### Using the Inference Service

#### Getting Predictions

To get predictions for a single sample, use the following command:

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
          "id": "DFFXKT",
          "color": null,
          "number": null
        }
    ]
}' http://localhost:8080/infer
```

The key `instances` contains a list of objects, each of which is a sample for which the prediction is requested. The server will respond with a JSON object containing the predicted probabilities for each input record:

```json
{
  "status": "success",
  "message": "",
  "timestamp": "2023-08-06T17:40:47.765863",
  "requestId": "19ebe8d6ce",
  "targetClasses": ["0", "1", "2"],
  "targetDescription": "Target class: one of {0, 1, 2}",
  "predictions": [
    {
      "sampleId": "DFFXKT",
      "predictedClass": "2",
      "predictedProbabilities": [0.00143, 0.0, 0.99857]
    }
  ]
}
```

#### Getting predictions and local explanations

To get predictions and explanations for a single sample, use the following request to send to `/explain` endpoint (same structure as data for the `/infer` endpoint):

```bash
curl -X POST -H "Content-Type: application/json" -d '{
  {
    "instances": [
        {
          "id": "DFFXKT",
          "color": null,
          "number": null
        }
    ]
}' http://localhost:8080/explain
```

The server will respond with a JSON object containing the predicted probabilities and locations for each input record:

```json
{
  "status": "success",
  "message": "",
  "timestamp": "2023-08-06T17:41:29.368119",
  "requestId": "2ff0a73f36",
  "targetClasses": ["0", "1", "2"],
  "targetDescription": "Target class: one of {0, 1, 2}",
  "predictions": [
    {
      "sampleId": "DFFXKT",
      "predictedClass": "2",
      "predictedProbabilities": [0.00143, 0.0, 0.99857],
      "explanation": {
        "baseline": [0.66315, 0.17233, 0.16452],
        "featureScores": {
          "color_Blue": [-0.12895, 0.02178, 0.10716],
          "color_Green": [-0.06613, -0.07633, 0.14246],
          "color_Red": [-0.06385, -0.06317, 0.12702],
          "number": [-0.0476, 0.00185, 0.04575],
          "number_na": [-0.35519, -0.05645, 0.41165]
        }
      }
    }
  ],
  "explanationMethod": "Shap"
}
```

#### OpenAPI

Since the service is implemented using FastAPI, we get automatic documentation of the APIs offered by the service. Visit the docs at `http://localhost:8080/docs`.

## Testing

### Running through Tox

This project uses Tox for running tests. For this, you will need tox installed on your system. You can install tox using pip:

```bash
pip install tox
```

Once you have tox installed, you can run all tests by simply running the following command from the root of your project directory:

```bash
tox
```

This will run the tests as well as formatters `black` and `isort` and linter `flake8`. You can run tests corresponding to specific environment, or specific markers. Please check `tox.ini` file for configuration details.

### Running through Pytest

To run tests using pytest, first create a virtual environment and install the dependencies listed in the following three files located in the `requirements` directory`:

- `requirements.txt`: for main dependencies
- `requirements_test.txt`: for test dependencies
- `requirements_quality.txt`: for dependencies related to code quality (formatting, linting, complexity, etc.)
  Once you have the dependencies installed, you can run the tests using the following command from the root of your project directory:

```bash
# Run all tests
pytest
# or, to run tests in a specific directory
pytest <path_to_directory>
# or, to run tests in a specific file
pytest <path_to_file>
# or, to run tests with a specific marker (such as `slow`, or `not slow`)
pytest -m <marker_name>
```

## Requirements

The requirements files are placed in the folder `requirements`.
Dependencies for the main model implementation in `src` are listed in the file `requirements.txt`.
For testing, dependencies are listed in the file `requirements_test.txt`.
Dependencies for quality-tests are listed in the file `requirements_quality.txt`. You can install these packages by running the following command from the root of your project directory:

```python
pip install -r requirements/requirements.txt
pip install -r requirements/requirements_test.txt
pip install -r requirements/requirements_quality.txt
```

Alternatively, you can let tox handle the installation of test dependencies for you for testing purposes. To do this, simply run the command `tox` from the root directory of the repository. This will create the environments, install dependencies, and run the tests as well as quality checks on the code.

## LICENSE

This project is provided under the MIT License. Please see the [LICENSE](LICENSE) file for more information.

## Contact Information

Repository created by Ready Tensor, Inc. (https://www.readytensor.ai/)
