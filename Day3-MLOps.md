# 🚀 Day 3 of 30-Day MLOps Challenge: Mastering Data Versioning with DVC

**Author:** [Bittu Sharma](https://www.linkedin.com/in/bittusharma)  
**Date:** Nov 12, 2025  

---

## 📚 Key Learnings

- Why versioning datasets is as important as versioning code in ML workflows  
- How **DVC (Data Version Control)** integrates with Git for full pipeline reproducibility  
- How to use DVC to track datasets, models, and pipelines  
- Basics of setting up a DVC project, connecting remote storage, and managing large files  
- How DVC enables collaboration across ML teams by standardizing data + code versioning  

---

## 🧠 Learn Here — What is Data Versioning?

**Data Versioning** in ML is the practice of tracking, managing, and controlling changes to datasets used throughout the machine learning lifecycle — similar to how Git tracks code versions.

---

## 🧰 Tools Used for Data Versioning

| Tool | Description |
|------|--------------|
| **DVC (Data Version Control)** | Git-like version control for data and models |
| **LakeFS** | Git-style versioning for object stores (e.g., S3) |
| **Pachyderm** | Data lineage and versioning built into pipelines |
| **Weights & Biases / MLflow** | Can log and track dataset artifacts and metadata |

---

## ⚖️ Why Versioning Datasets is as Important as Versioning Code

### 1. 🧬 Reproducibility
Just like code, the training dataset determines the behavior of the ML model.  
Without dataset versioning, reproducing results becomes impossible since even small data changes can alter model outcomes.  
Essential for debugging, validation, audits, and regulated environments.

### 2. 📈 Experiment Tracking
Tracking which dataset version was used in each experiment is key to evaluating model performance over time.  
Helps compare results across dataset iterations.  

### 3. 🤝 Collaboration
Ensures team members work with the same, consistent data.  
Prevents confusion from ad-hoc data changes and supports parallel experimentation.

### 4. 📊 Model Performance Monitoring
Tracks how dataset changes affect model performance.  
Allows rollback to previous versions in case of performance degradation.

### 5. 🚀 Production Consistency
Ensures that production models use the exact dataset they were trained and tested on.  
Prevents **data drift** caused by unnoticed dataset changes.

### 6. 🛡️ Compliance and Auditing
Regulated industries require full traceability of datasets used in models.  
Dataset versioning supports audit trails and compliance documentation.

---

## 💡 What is DVC?

**DVC (Data Version Control)** is an open-source tool that helps track, version, and manage data, models, and experiments in ML workflows — similar to how Git tracks code.

---

## ⚙️ Why DVC?

ML projects often involve:

- Large datasets and models (too big for Git)
- Reproducibility issues from dynamic data
- Collaboration needs across data + code

---

## 🔍 What Does DVC Do?

| Feature | Description |
|----------|--------------|
| 🔄 **Data Versioning** | Track large files (datasets, models) via lightweight metadata in Git |
| ⚙️ **Pipelines** | Define data processing and model training workflows |
| 💾 **Remote Storage** | Sync data/models to S3, GCS, Azure, SSH, etc. |
| 🔬 **Experiment Tracking** | Track hyperparameters, code, data, and results |
| 🔗 **Git Integration** | Works alongside Git for complete project versioning |

---

## 🧰 Installing DVC

### 🖥 macOS
```bash
brew install dvc
# or
pip install dvc
# With S3 support
pip install "dvc[s3]"
🐧 Linux
bash
Copy code
pip install dvc
# With GDrive or SSH
pip install "dvc[gdrive,ssh]"
# Using Snap
sudo snap install dvc --classic
# Using Conda
conda install -c conda-forge dvc
🪟 Windows
bash
Copy code
pip install dvc
# or
choco install dvc
# Using Conda
conda install -c conda-forge dvc
✅ Verify installation:

bash
Copy code
dvc --version
🔧 How DVC Works
🧠 Core Concept
Git tracks code and metadata.

DVC manages large data files, model artifacts, and pipeline stages.

DVC creates .dvc, dvc.yaml, and dvc.lock files — all tracked by Git.

⚙️ Workflow Integration
🗂️ Version Control Everything
Git stores pipeline definitions, DVC stores large data remotely.

👥 Collaborate
Team members clone repo via Git.

Run dvc pull to fetch datasets/models.

Run dvc repro to reproduce full pipeline.

🚀 Step-by-Step Workflow
1️⃣ Initialize Git & DVC
bash
Copy code
git init
dvc init
git commit -m "Initialize Git and DVC"
2️⃣ Track Data and Models
bash
Copy code
dvc add data/raw_data.csv
git add data/raw_data.csv.dvc .gitignore
git commit -m "Track raw data with DVC"
3️⃣ Configure Remote Storage
bash
Copy code
dvc remote add -d myremote s3://mybucket/dvcstore
dvc push
4️⃣ Track ML Models
bash
Copy code
mv model.pkl models/model.pkl
dvc add models/model.pkl
git add models/model.pkl.dvc models/.gitignore
git commit -m "Track ML model with DVC"
dvc push
5️⃣ Define & Track ML Pipeline
bash
Copy code
dvc run -n preprocess \
  -d data/raw_data.csv -o data/processed \
  python scripts/preprocess.py

git add dvc.yaml dvc.lock
git commit -m "Add preprocess stage to pipeline"

dvc run -n train_model \
  -d src/train.py -d data/raw-dataset.csv \
  -o models/model.pkl \
  python src/train.py data/raw-dataset.csv models/model.pkl
6️⃣ Reproduce the Pipeline
bash
Copy code
dvc repro
7️⃣ Visualize the Pipeline
bash
Copy code
dvc dag
8️⃣ Collaborate via Remotes
bash
Copy code
git pull
dvc pull
🎯 Benefits for Pipeline Reproducibility
Data + Code Coupling: Git for code, DVC for data alignment

Reproducibility: dvc.lock captures exact inputs/outputs

Collaboration: Teams reproduce results reliably

Modularity: Pipelines built with multiple stages

📖 Learning Resources
📘 Official DVC Documentation

📘 DVC Get Started Guide

📘 Why Use DVC?

📘 DVC + Git Workflow Explained

🔥 Challenges
✅ Set up DVC in a new or existing Git-based ML project
✅ Add and track a dataset (data.csv) using dvc add
✅ Commit and push changes to GitHub and DVC remote
✅ Clone project and reproduce dataset with dvc pull
✅ Write a README on “How to use DVC for data versioning in this project”
✅ Set up S3 or GCS as remote and sync data

🤷🏻 How to Participate?
✅ Complete the tasks and challenges

✅ Document your progress and learnings

✅ Share your journey on GitHub, Medium, or Hashnode

🔗 Connect with Me
LinkedIn

GitHub

💬 Keep Learning...
"Data Versioning is the foundation of reproducible, reliable, and scalable ML systems."