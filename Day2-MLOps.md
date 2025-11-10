# 🧠 30 Days of MLOps Challenge: Day 2  
## 🚀 MLOps Tools Landscape – Explore the Ecosystem  

### 📚 Key Learnings
- Understand the different categories of tools used across the ML lifecycle  
- Compare open-source tools vs managed services (e.g., MLflow vs SageMaker)  
- Identify how each tool fits into MLOps stages: versioning, training, orchestration, deployment, monitoring  
- Learn in-depth about MLflow, DVC, Kubeflow, Airflow, SageMaker, and Vertex AI – their roles and strengths  

---

## 🧩 Categories of Tools Across the ML Lifecycle
The Machine Learning (ML) lifecycle spans multiple stages from data acquisition to model monitoring.  
Various categories of tools are used to support and automate these stages effectively:

### 1. Data Engineering & Preparation
**Tools:**  
- Data Collection: Apache Nifi, Kafka, Web Scrapers, APIs  
- Data Cleaning: OpenRefine, Pandas, DataWrangler  
- Data Transformation: Spark, dbt, Airbyte  
- Data Storage: PostgreSQL, S3, Delta Lake, BigQuery  

### 2. Experimentation & Development
**Tools:**  
- Notebooks & IDEs: Jupyter, Colab, VS Code  
- Experiment Tracking: MLflow, Weights & Biases, Neptune.ai  
- Data Versioning: DVC, LakeFS  
- Feature Stores: Feast, Tecton  

### 3. Model Training & Optimization
**Tools:**  
- Frameworks: TensorFlow, PyTorch, Scikit-learn, XGBoost  
- Hyperparameter Tuning: Optuna, Ray Tune, Hyperopt  
- Distributed Training: Horovod, SageMaker, Vertex AI  

### 4. Model Packaging & Deployment
**Tools:**  
- Packaging: ONNX, TorchScript, BentoML  
- Deployment: KFServing, Seldon Core, SageMaker, Vertex AI  
- Containerization: Docker, Podman  
- CI/CD: Jenkins, GitHub Actions, GitLab CI  

### 5. Model Monitoring & Observability
**Tools:**  
- Monitoring: Prometheus, Grafana, Evidently AI  
- Drift Detection: WhyLabs, Fiddler, Arize  
- Logging & Tracing: ELK Stack, Jaeger, OpenTelemetry  

### 6. Governance & Compliance
**Tools:**  
- Explainability: SHAP, LIME  
- Fairness: AI Fairness 360, Fairlearn  
- Security & Access: Vault, OPA, IAM tools  

### 7. Workflow Orchestration
**Tools:** Apache Airflow, Kubeflow Pipelines, Prefect, Dagster  

---

## ⚖️ Open-Source Tools vs Managed Services in MLOps

| Feature/Aspect | Open-Source Tools (MLflow, DVC, Kubeflow) | Managed Services (SageMaker, Vertex AI, Azure ML) |
|----------------|--------------------------------------------|--------------------------------------------------|
| Ease of Setup | Manual setup & configuration | Out-of-the-box setup |
| Customization | Fully customizable | Limited customization |
| Scalability | Manual scaling | Auto-scaled by provider |
| Integration | Needs glue code | Native ecosystem integration |
| Cost | Low upfront, infra cost grows | Pay-as-you-go |
| Data Security | Full control | Provider-dependent |
| Maintenance | Manual | Managed by provider |
| Learning Curve | Steep | Easier for beginners |
| Community Support | Large OSS communities | Official support |
| Vendor Lock-In | None | High risk |

**Summary:**  
- Open-Source Tools → Best for flexibility, control, and infrastructure-savvy teams.  
- Managed Services → Ideal for fast deployment and low ops overhead.  

---

## 🔁 MLOps Tools Categorized by ML Lifecycle Stages

### 1. Versioning
- Git – Code versioning  
- DVC – Data & model versioning  
- MLflow – Model tracking  
- Weights & Biases – Experiment tracking  

### 2. Training
- Frameworks: TensorFlow, PyTorch, Scikit-learn  
- MLflow & W&B – Experiment tracking, tuning  
- Optuna, Keras Tuner – Optimization  

### 3. Orchestration
- Apache Airflow, Kubeflow Pipelines, Argo Workflows, Metaflow  

### 4. Deployment
- Seldon Core, KServe, TensorFlow Serving, SageMaker, BentoML  

### 5. Monitoring
- Prometheus, Grafana, Evidently AI, WhyLabs, SageMaker Model Monitor, Arize AI  

---

## 🧩 MLOps Tools: In-Depth Guide

### 1. MLflow – Experiment Tracking, Model Registry, and Deployment

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_metric("rmse", 0.78)
    mlflow.sklearn.log_model(model, "model")
```

### 2. DVC (Data Version Control)

```bash
dvc init
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Add training data"
dvc remote add -d myremote s3://ml-data-store
dvc push
```

### 3. Kubeflow – End-to-End ML on Kubernetes

```python
@dsl.pipeline(name='Basic pipeline', description='An example pipeline.')
def basic_pipeline():
    op = dsl.ContainerOp(
        name='echo',
        image='alpine',
        command=['echo', 'Hello Kubeflow!']
    )
```

### 4. Apache Airflow – Workflow Orchestration

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def train_model():
    print("Training model...")

def evaluate_model():
    print("Evaluating model...")

dag = DAG('ml_pipeline', start_date=datetime(2023, 1, 1), schedule_interval='@daily')

train = PythonOperator(task_id='train', python_callable=train_model, dag=dag)
evaluate = PythonOperator(task_id='evaluate', python_callable=evaluate_model, dag=dag)

train >> evaluate
```

### 5. Amazon SageMaker – Managed ML Platform

```python
from sagemaker.sklearn.estimator import SKLearn

sklearn = SKLearn(entry_point='train.py',
                  role='SageMakerRole',
                  instance_type='ml.m5.large')
sklearn.fit()

predictor = sklearn.deploy(instance_type='ml.m5.large', initial_instance_count=1)
```

### 6. Google Vertex AI – Unified ML Platform

```python
from google.cloud import aiplatform

aiplatform.init(project='my-project', location='us-central1')

job = aiplatform.CustomTrainingJob(
    display_name='my-training-job',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/tf-cpu.2-2:latest',
    requirements=['pandas']
)

model = job.run(replica_count=1, model_display_name='my-model')
```

---

## 🔥 Challenges
- Create a visual diagram showing how each MLOps tool fits into the ML lifecycle  
- Write a short post: “MLflow vs SageMaker – Which one to start with and why?”  
- Install MLflow locally and log a dummy experiment  
- Create a DVC pipeline to version a small dataset (CSV is fine)  

---

## 🤷🏻 How to Participate?
✅ Complete the above challenges  
✅ Document your progress and learnings on GitHub ReadMe, Medium, or Hashnode  

---

## ✍️ Conclusion
MLOps brings together the power of DevOps, data engineering, and machine learning.  
By mastering tools like MLflow, DVC, Kubeflow, Airflow, SageMaker, and Vertex AI, you can automate every stage of your ML lifecycle — from data to deployment — with scalability, reproducibility, and observability.
