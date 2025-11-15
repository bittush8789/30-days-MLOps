# Day 9 – ML Pipelines with Kubeflow Pipelines

## Automate & Orchestrate ML Workflows

### 📚 Key Learnings

* Why ML pipelines are essential in production ML workflows
* What is orchestration and what are Directed Acyclic Graphs (DAGs)
* Kubeflow Pipelines for building and managing ML workflows
* Step-based (Airflow) vs component-based (Kubeflow) workflows
* Install Kubeflow & run a pipeline on Kubeflow

---

## 🔍 What Are ML Pipelines?

ML pipelines automate the workflow of developing, deploying, and maintaining ML models.

### **Key Pipeline Stages**

| Stage                      | Description                                 |
| -------------------------- | ------------------------------------------- |
| **1. Data Ingestion**      | Load data from sources like APIs, DBs, CSVs |
| **2. Data Preprocessing**  | Clean, normalize, transform data            |
| **3. Feature Engineering** | Build features to improve models            |
| **4. Model Training**      | Train ML models                             |
| **5. Model Evaluation**    | Validate model performance                  |
| **6. Model Deployment**    | Deploy model to production                  |
| **7. Monitoring**          | Track performance, detect drift             |
| **8. Retraining**          | Retrain model based on new data             |

### Why Use ML Pipelines?

* ✔ Automation
* ✔ Reproducibility
* ✔ Scalability
* ✔ Versioning
* ✔ Collaboration

### Common Tools

| Tool                      | Purpose                          |
| ------------------------- | -------------------------------- |
| Scikit-learn Pipelines    | Simple ML pipelines              |
| Kubeflow Pipelines        | Scalable pipelines on Kubernetes |
| Apache Airflow            | Orchestration engine             |
| MLflow                    | Experiment tracking              |
| Tecton/Feast              | Feature store integration        |
| TensorFlow Extended (TFX) | Production ML pipelines          |

---

## 🔄 What is Orchestration?

Orchestration automates & manages workflow execution.

**Ensures:**

* Correct execution order
* Dependency management
* Efficient resource usage
* Retry & failure handling

### Common Orchestration Tools

* Apache Airflow
* Kubeflow Pipelines
* Argo Workflows
* Prefect
* Luigi

---

## 🔗 What is a DAG?

A Directed Acyclic Graph (DAG):

* Nodes represent tasks
* Edges represent dependencies
* No cycles → ensures forward-only execution

Benefits:

* Order of execution
* Dependency control
* Parallel execution
* Failure isolation
* Reusability
* Visualization

---

# 🚀 Kubeflow Overview

Kubeflow = **Kubernetes + Machine Learning Workflows**

### Core Components

| Component          | Description                 |
| ------------------ | --------------------------- |
| Kubeflow Pipelines | Build & manage ML workflows |
| Katib              | Hyperparameter tuning       |
| KServe             | Model serving               |
| Notebooks          | Jupyter notebooks on K8s    |
| TFJob / PyTorchJob | Distributed training        |
| Central Dashboard  | Main UI                     |

Includes: Argo, Istio, MinIO, Dex, etc.

---

# 🧩 Kubeflow Concepts

* **Jupyter Notebooks** → interactive dev
* **Volumes** → Persistent storage
* **Pipeline Components** → Reusable ML steps
* **Experiments / Runs** → Track executions
* **Training Jobs** → Distributed ML
* **KServe** → Model serving
* **MinIO / S3** → Artifact storage
* **ML Metadata** → Lineage, tracking
* **Profiles & RBAC** → Multi-tenancy

---

# ⚙️ How Kubeflow Works

1. User interacts via UI / SDK
2. Notebook servers for development
3. Pipelines UI for DAG workflows
4. Argo Workflows runs pipeline tasks
5. Katib handles HPO
6. Training operators execute distributed jobs
7. KServe deploys trained models
8. Prometheus/Grafana monitor metrics

---

# 🔁 Airflow vs Kubeflow Pipelines

| Feature        | Airflow          | Kubeflow                 |
| -------------- | ---------------- | ------------------------ |
| Workflow Type  | Step-based       | Component-based          |
| Execution Unit | Python operators | Containerized components |
| Best For       | Data engineering | ML workflows             |
| K8s Native     | Partial          | Full native              |
| Metadata       | Limited          | Full ML metadata         |
| Reusability    | Low              | High                     |

---

# 🛠️ Install Kubeflow (KIND Setup)

### Prerequisites

* 16GB RAM recommended
* Docker or Podman
* kind v0.27+
* kustomize installed

### Step 1 – Create KIND Cluster

```
cat <<EOF | kind create cluster --name=kubeflow --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  image: kindest/node:v1.32.0@sha256:c48c62eac5da28cdadcf560d1d8616cfa6783b58f0d94cf63ad1bf49600cb027
  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    apiServer:
      extraArgs:
        "service-account-issuer": "https://kubernetes.default.svc"
        "service-account-signing-key-file": "/etc/kubernetes/pki/sa.key"
EOF
```

### Step 2 – Export Kubeconfig

```
kind get kubeconfig --name kubeflow > /tmp/kubeflow-config
export KUBECONFIG=/tmp/kubeflow-config
```

### Step 3 – Docker Credentials

```
kubectl create secret generic regcred \
  --from-file=.dockerconfigjson=$HOME/.docker/config.json \
  --type=kubernetes.io/dockerconfigjson
```

### Step 4 – Install Kubeflow

```
git clone https://github.com/kubeflow/manifests.git
cd manifests

while ! kustomize build example | kubectl apply --server-side --force-conflicts -f -; do
  echo "Retrying to apply resources";
  sleep 20;
done
```

### Verify Pods

```
kubectl get pods -n kubeflow
```

### Access Dashboard

```
kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80
```

Open: **[http://localhost:8080](http://localhost:8080)**

Login:

* Email: [user@example.com](mailto:user@example.com)
* Password: 12341234

---

# 📦 Build a Kubeflow Pipeline (Iris Example)

Install packages:

```
pip install kfp scikit-learn pandas joblib
```

### Pipeline Steps

1. Preprocess
2. Train
3. Evaluate

### Full Pipeline Code with Artifact Passing

```
from kfp import dsl
from kfp.components import create_component_from_func, InputPath, OutputPath

# Step 1: Preprocess data and output CSV

def preprocess(output_data_path: OutputPath(str), output_target_names: OutputPath(str)):
    from sklearn.datasets import load_iris
    import pandas as pd
    import joblib
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv(output_data_path, index=False)
    joblib.dump(iris.target_names, output_target_names)

preprocess_op = create_component_from_func(
    preprocess,
    base_image='python:3.8-slim',
    packages_to_install=['scikit-learn', 'pandas', 'joblib']
)

# Step 2: Train model

def train(input_data_path: InputPath(str), output_model_path: OutputPath(str)):
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier
    import joblib
    df = pd.read_csv(input_data_path)
    X = df.drop(columns='target')
    y = df['target']
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, output_model_path)

train_op = create_component_from_func(
    train,
    base_image='python:3.8-slim',
    packages_to_install=['scikit-learn', 'pandas', 'joblib']
)

# Step 3: Evaluate model

def evaluate(input_data_path: InputPath(str), input_model_path: InputPath(str)):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score
    df = pd.read_csv(input_data_path)
    X = df.drop(columns='target')
    y = df['target']
    model = joblib.load(input_model_path)
    acc = accuracy_score(y, model.predict(X))
    print(f"Accuracy: {acc}")

evaluate_op = create_component_from_func(
    evaluate,
    base_image='python:3.8-slim',
    packages_to_install=['scikit-learn', 'pandas', 'joblib']
)

# Pipeline Definition
@dsl.pipeline(
    name='Scikit-learn Iris Pipeline with Artifacts',
    description='Pipeline demonstrating artifact passing'
)
def iris_pipeline():
    preprocess_task = preprocess_op()
    train_task = train_op(input_data_path=preprocess_task.outputs['output_data_path'])
    evaluate_task = evaluate_op(
        input_data_path=preprocess_task.outputs['output_data_path'],
        input_model_path=train_task.outputs['output_model_path']
    )

from kfp.compiler import Compiler
Compiler().compile(iris_pipeline, 'iris_pipeline_artifact.yaml')
```

---

# 🔥 Challenges

* Build a 3-step ML pipeline using Kubeflow
* Convert steps into reusable components
* Visualize DAG in Kubeflow UI
* Add retry logic & conditional steps
* Store artifacts in MinIO / S3
* Upload compiled YAML & run it

---

# 🙌 Follow Me

* **LinkedIn:** Bittu Sharma
* **GitHub:** github.com

Keep Learning… 🚀
