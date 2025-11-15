# 🚀 Day 7 of 30-Day MLOps Challenge

## Model Experiment Tracking with MLflow -- "Log It or Lose It!"

### 📚 Key Learnings

-   Importance of experiment tracking: **reproducibility, comparability,
    collaboration**\
-   MLflow architecture: **Tracking, Projects, Models, Registry**\
-   Tracking parameters, metrics, artifacts with MLflow\
-   Logging & registering models for deployment

------------------------------------------------------------------------

## 🧠 What is ML Experiment Tracking?

ML Experiment Tracking is the process of logging, organizing, and
comparing ML runs to understand what changes improve results.

### Why It's Needed

You often try different:\
- Model architectures\
- Hyperparameters\
- Preprocessing methods\
- Datasets

Without tracking, you lose reproducibility and comparability.

### What It Tracks

  Tracked Item   Description
  -------------- -------------------------------------
  Parameters     Hyperparameters like LR, batch size
  Metrics        Accuracy, loss, precision, recall
  Artifacts      Models, plots, datasets
  Code Version   Git commit hash
  Environment    Python & library versions

------------------------------------------------------------------------

## 🔧 Tools for ML Experiment Tracking

-   **MLflow Tracking** (Most popular)\
-   Weights & Biases\
-   Neptune.ai\
-   Comet.ml\
-   DVC + Git

------------------------------------------------------------------------

# Why Experiment Tracking is Critical in MLOps

## 🔁 Reproducibility

Ensures same results across code, data & hyperparameters.

## 📊 Comparability

Helps compare many runs to find best-performing models.

## 🤝 Collaboration

Teams can share experiments, artifacts & results easily.

------------------------------------------------------------------------

# 🔥 MLflow Overview

MLflow is an open-source ML lifecycle platform covering:\
- **Tracking**\
- **Projects**\
- **Models**\
- **Model Registry**

------------------------------------------------------------------------

# 🏗️ MLflow Architecture

  Component   Purpose
  ----------- -----------------------------------
  Tracking    Log metrics, params, artifacts
  Projects    Standardize reproducible code
  Models      Unified model packaging
  Registry    Versioning & lifecycle management

------------------------------------------------------------------------

# 🔍 MLflow Tracking API -- Code Examples

## Install

``` bash
pip install mlflow
```

## Basic Tracking Example

``` python
import mlflow

mlflow.set_experiment("my-experiment")

with mlflow.start_run(run_name="my-first-run") as run:
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("optimizer", "adam")

    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)

    with open("model.txt", "w") as f:
        f.write("This is a dummy model file.")
    mlflow.log_artifact("model.txt")

    print("Run ID:", run.info.run_id)
```

## Multiple Metrics

``` python
for step in range(5):
    mlflow.log_metric("accuracy", 0.9 + step*0.01, step=step)
```

## Access Runs

``` python
from mlflow.tracking import MlflowClient
client = MlflowClient()
runs = client.search_runs(experiment_ids=["0"])
```

------------------------------------------------------------------------

# 🖥️ MLflow UI

Start UI:

``` bash
mlflow ui
```

Production use:

``` bash
mlflow server --host 127.0.0.1 --port 8080
```

------------------------------------------------------------------------

# 📦 MLflow: Log, Register, and Deploy Models

## 1. Log Model

``` python
mlflow.sklearn.log_model(model, artifact_path="rf_model", registered_model_name="IrisRFModel")
```

## 2. Register Model

``` python
client.create_registered_model("IrisRFModel")
```

## 3. Load Model

``` python
model = mlflow.sklearn.load_model("models:/IrisRFModel/Production")
preds = model.predict(X_test)
```

## 4. Serve Model

``` bash
mlflow models serve -m models:/IrisRFModel/Production -p 5001
```

------------------------------------------------------------------------

# 🔥 Challenges

-   Add MLflow tracking to any ML script\
-   Log 3 parameters & 2 metrics\
-   Save model artifact\
-   Compare runs in UI\
-   Configure custom backend store\
-   Explore mlflow.projects

------------------------------------------------------------------------

# 🙌 How to Participate

-   Complete tasks\
-   Document progress on GitHub/Medium/Hashnode\
-   Follow for more updates

------------------------------------------------------------------------

# Keep Learning 🚀
