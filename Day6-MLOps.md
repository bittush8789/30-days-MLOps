# 🚀 Day 6 of 30-Day MLOps Challenge

## Training ML Models with Scikit-learn & TensorFlow – Build & Save Your Models Like a Pro

### 📚 Key Learnings

* How to train ML models using Scikit-learn and TensorFlow
* Differences between traditional ML (Scikit-learn) vs deep learning (TensorFlow/Keras) workflows
* Importance of saving models (joblib, pickle, SavedModel) for deployment
* Introduction to training pipelines: modular code, data loading, preprocessing, training, evaluation, and saving


## 🧠 ML Model Basics

An ML model is a mathematical representation trained to recognize patterns or make decisions based on data.

**In simple terms:**
A model learns from data (e.g., cat vs dog images) and predicts outcomes for new unseen data.

### ML Workflow

1. **Raw Data** → Collected from images, text, numbers, logs
2. **Feature Extraction** → Identify useful patterns
3. **Train Algorithm** → Learn from data
4. **Trained Model** → Ready for predictions
5. **New Data** → Feed unseen inputs
6. **Prediction** → Output from the model

### ML Model Components

* **Features** (input)
* **Weights/Parameters** (learned values)
* **Algorithm** (mapping logic)
* **Output** (prediction)

---

## 📦 Scikit-learn Overview

Scikit-learn is a Python library for:

* Classification
* Regression
* Clustering
* Dimensionality Reduction
* Model selection + preprocessing

### Key Concepts

* **Estimator**: has `fit()` method
* **Predictor**: has `predict()`
* **Transformer**: has `transform()`
* **Pipeline**: chaining steps
* **Model Persistence**: save using `joblib` or `pickle`

---

## 🧪 Practical Project: Predicting Student Performance Using Scikit-Learn

### 📁 Project Structure

```
├── data_generation.py
├── train_model.py
├── predict.py
├── serve_model.py
├── students.csv
├── student_model.pkl
└── README.md
```

### Step 1 — Generate Dataset (`data_generation.py`)

```python
import pandas as pd
import random

random.seed(42)
schools = ['Greenwood High', 'Sunrise Public', 'Hillview School']
data = []
for _ in range(100):
    school = random.choice(schools)
    study_hours = round(random.uniform(1, 10), 1)
    absences = random.randint(0, 10)
    grade = round(random.uniform(40, 100), 1)
    passed = 1 if grade >= 50 else 0
    data.append([school, study_hours, absences, grade, passed])

df = pd.DataFrame(data, columns=['school', 'study_hours', 'absences', 'grade', 'passed'])
df.to_csv("students.csv", index=False)
print("✅ Sample student data saved to students.csv")
```

### Step 2 — Train Model (`train_model.py`)

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

df = pd.read_csv("students.csv")
X = df.drop("passed", axis=1)
y = df["passed"]

preprocessor = ColumnTransformer([
    ("school_encoder", OneHotEncoder(), ["school"])
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "student_model.pkl")
print("✅ Model trained and saved as student_model.pkl")
```

### Step 3 — Prediction (`predict.py`)

```python
import joblib
import pandas as pd

model = joblib.load("student_model.pkl")

sample = pd.DataFrame([{
    "school": "Greenwood High",
    "study_hours": 6.5,
    "absences": 2,
    "grade": 78.0
}])

pred = model.predict(sample)
print(f"🎯 Predicted: {'Pass' if pred[0] == 1 else 'Fail'}")
```

### Step 4 — Serving with FastAPI (`serve_model.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()
model = joblib.load("student_model.pkl")

class StudentData(BaseModel):
    school: str
    study_hours: float
    absences: int
    grade: float

@app.post("/predict")
def predict(data: StudentData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": "Pass" if prediction == 1 else "Fail"}
```

---

# 🧠 TensorFlow Overview

TensorFlow is Google's deep learning framework for ML and DL tasks.

### Features

* Supports CPU/GPU/TPU
* Keras high-level API
* TensorBoard for visualization
* TensorFlow Serving for deployment

---

## 📈 Practical Example: Time Series Forecasting (TensorFlow)

### Project Structure

```
.
├── data/
│   └── generate_dataset.py
├── models/
│   └── saved_model/
├── src/
│   ├── train_model.py
│   └── serve_model.py
├── sample_data.csv
└── README.md
```

### Dataset Generation

```python
# data/generate_dataset.py
import numpy as np
import pandas as pd

days = np.arange(365)
temperature = 10 + 0.02 * days + np.sin(0.1 * days) + np.random.normal(0, 0.5, size=(365,))
df = pd.DataFrame({'day': days, 'temperature': temperature})
df.to_csv("sample_data.csv", index=False)
```

### Train LSTM Model

```python
# src/train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("sample_data.csv")
data = df['temperature'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 10
X, y = create_sequences(data_scaled, window_size)

split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

model = Sequential([
    LSTM(64, input_shape=(window_size, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model.save("models/saved_model")
```

### Load & Predict

```python
# src/serve_model.py
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("sample_data.csv")
data = df['temperature'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
model = load_model("models/saved_model")

window_size = 10
recent_sequence = data_scaled[-window_size:].reshape(1, window_size, 1)
future_prediction = model.predict(recent_sequence)
future_temperature = scaler.inverse_transform(future_prediction)
print(f"Predicted next day's temperature: {future_temperature[0][0]:.2f}")
```

---

# 📦 Importance of Saving Models

| Purpose         | Why It Matters          |
| --------------- | ----------------------- |
| Deployment      | Serve via API or app    |
| Reproducibility | Same results every time |
| Performance     | Avoid retraining        |
| Portability     | Shareable across teams  |
| Versioning      | Track model history     |

### Best Tools

| Tool       | Best For            | Notes                        |
| ---------- | ------------------- | ---------------------------- |
| joblib     | Scikit-learn models | Best for numpy-heavy objects |
| pickle     | Python objects      | Flexible but less optimized  |
| SavedModel | TensorFlow models   | Best for production serving  |

---

# 🧱 Basic Training Pipeline Structure

```
data_loader.py
preprocessing.py
model.py
train.py
evaluate.py
save_load.py
pipeline.py
data/data.csv
models/model.pkl
```

### Example Modules

#### `data_loader.py`

```python
import pandas as pd

def load_data(path):
    return pd.read_csv(path)
```

#### `preprocessing.py`

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
```

#### `model.py`

```python
from sklearn.ensemble import RandomForestClassifier

def build_model():
    return RandomForestClassifier(n_estimators=100, random_state=42)
```

#### `train.py`

```python
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
```

#### `evaluate.py`

```python
from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    print(f"Accuracy: {acc:.2f}")
    return {"accuracy": acc}
```

#### `save_load.py`

```python
import joblib

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)
```

---

# 🔥 Challenges

* Feature engineering using Pandas
* Train + save a Scikit-learn model (Iris/Titanic)
* Train + save a TensorFlow model (MNIST/Fashion-MNIST)
* Log evaluation metrics
* Modularize training pipeline
* Load saved models and run inference
* Document workflow in README.md

---

# 🤝 How to Participate?

✔ Complete tasks and challenges
✔ Document progress
✔ Post updates on GitHub, Medium, Hashnode

**Keep Learning… 🚀**
