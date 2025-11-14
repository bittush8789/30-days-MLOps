# 🚀 Day 4 of 30-Day MLOps Challenge: Reproducible ML Environments using Conda & Docker

## 📚 Key Learnings

* Importance of environment reproducibility in ML (avoid *"works on my machine"* issue)
* How to use Conda to manage Python environments and dependencies
* How to create portable and consistent ML environments using Docker
* Differences and synergies between Conda and Docker

---

## 🧩 Environment Reproducibility in ML

Environment reproducibility refers to recreating the **exact same setup** — software versions, libraries, system dependencies, and hardware — to ensure consistent model performance across machines and time.

### 💡 Why It Matters

* **Consistent Results** across training, testing, and production
* **Reliable Experimentation** for comparative studies
* **Team Collaboration** without setup issues
* **No "Works on My Machine" Problems**
* **Simplified CI/CD & Debugging**

### 🧰 Tools for Reproducibility

* **Conda / Virtualenv** – Python environment and dependency management
* **Docker** – Portable system-level environment packaging
* **Pip + requirements.txt** – Python package tracking
* **MLflow / DVC** – Track models, data, and environments

Reproducible environments are essential for **trustworthy and scalable ML systems**.

---

## 🐍 Conda for Managing Python Environments in ML

Conda helps manage isolated, reproducible environments with smooth dependency handling.

### ⚙️ Why Use Conda?

* Avoid dependency conflicts
* Supports Python + non-Python packages
* Ideal for ML libraries (PyTorch, TensorFlow, Sklearn)
* Export/import complete environments easily

### 🪜 Steps to Use Conda

#### 1️⃣ Install Conda

**Linux / Mac:**

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

**Mac OS ARM:**

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```

**Windows (PowerShell):**

```
wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe" -outfile "./Downloads/Miniconda3-latest-Windows-x86_64.exe"
```

#### 2️⃣ Create Environment

```
conda create -n ml-env python=3.10
```

Or with packages:

```
conda create -n ml-env python=3.10 numpy pandas scikit-learn jupyter
```

#### 3️⃣ Activate Environment

```
conda activate ml-env
```

#### 4️⃣ Install Dependencies

```
conda install matplotlib seaborn jupyterlab
conda install -c conda-forge xgboost
```

Deep learning:

```
conda install -c pytorch pytorch torchvision torchaudio
conda install -c conda-forge tensorflow
```

#### 5️⃣ Add Jupyter Kernel

```
pip install ipykernel
python -m ipykernel install --user --name ml-env --display-name "Python (ml-env)"
```

#### 6️⃣ Export Environment

```
conda env export > environment.yml
```

#### 7️⃣ Recreate Environment

```
conda env create -f environment.yml
```

#### 8️⃣ Remove Environment

```
conda remove -n ml-env --all
```

### 💡 Conda vs pip Tips

* Use **conda** for binary packages
* Use **pip** only when not available in conda
* Install pip packages **last**

### 📁 ML Project Structure

```
my-ml-project/
├── data/
├── notebooks/
├── src/
├── environment.yml
└── README.md
```

### 🧠 Best Practices

* Version your `environment.yml`
* Use **conda-lock** or Docker
* Prefer **conda-forge** channel

---

## 🐳 Creating Portable and Consistent ML Environments Using Docker

### 💪 Why Use Docker for ML?

* Full environment portability
* Avoids "works on my machine" issues
* Ensures reproducible deployment

### 🪜 Steps

#### 1️⃣ Create a Dockerfile

```
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "train.py"]
```

#### 2️⃣ requirements.txt

```
numpy
pandas
scikit-learn
matplotlib
jupyterlab
tensorflow
```

#### 3️⃣ Build Image

```
docker build -t ml-env:latest .
```

#### 4️⃣ Run Container

Development mode:

```
docker run -it --rm -v $(pwd):/app ml-env:latest
```

Jupyter Lab:

```
docker run -it -p 8888:8888 -v $(pwd):/app ml-env:latest jupyter lab --ip=0.0.0.0 --allow-root
```

#### 5️⃣ Add .dockerignore

```
__pycache__/
*.pyc
.env
data/
models/
```

#### 6️⃣ docker-compose (Optional)

```
version: '3'
services:
  ml:
    build: .
    volumes:
      - .:/app
    ports:
      - "8888:8888"
  mongo:
    image: mongo:latest
    ports:
      - "27017:27017"
```

### 💡 Pro Tips

* Pin dependency versions
* Use lightweight base images
* Store data separately (volumes)
* Use .env for secrets

---

## ⚖️ Conda vs Docker Comparison

| Feature     | Conda                     | Docker              |
| ----------- | ------------------------- | ------------------- |
| Scope       | Python/R environments     | Full OS environment |
| Speed       | Faster local setup        | Slower image builds |
| Isolation   | Package-level             | System-level        |
| Portability | Medium                    | Very High           |
| Use Case    | ML notebooks, prototyping | Deployment, CI/CD   |

---

## 🧩 Docker + Conda: Best of Both Worlds

### Why Use Together?

* Package-level reproducibility (Conda)
* System-level reproducibility (Docker)
* Zero environment inconsistency

### ⚙️ Setup

**Dockerfile**

```
FROM continuumio/miniconda3
COPY environment.yml .
RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "mlenv", "/bin/bash", "-c"]
WORKDIR /app
COPY . .
CMD ["python", "train.py"]
```

**environment.yml**

```
name: mlenv
channels:
  - defaults
dependencies:
  - python=3.9
  - pandas
  - numpy
  - scikit-learn
```

### 🌟 Benefits

* Fully reproducible training environments
* Easier collaboration
* Scalable ML workflows

### 📂 Example Project Structure

```
ml-project/
├── Dockerfile
├── environment.yml
├── train.py
└── README.md
```

---

## 🔥 Challenges

* Create a Conda environment with 3 ML packages and export it
* Install a Jupyter kernel for your environment
* Build a Dockerfile containing Pandas + Scikit-learn
* Run a Docker container to verify dependencies
* Combine Conda + Docker using `environment.yml`
* Document everything in README.md

---

## 🤷🏻 How to Participate?

* Complete tasks
* Document on GitHub ReadMe, Medium, or Hashnode

Follow on **LinkedIn** and **GitHub**.

---

### Keep Learning… 🚀
