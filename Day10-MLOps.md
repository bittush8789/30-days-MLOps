🚀 Day 10 of 30 Days of MLOps Challenge
Serving ML Models with FastAPI & Flask

Author: Bittu Sharma
Date: Nov 16, 2025

📚 Key Learnings

How to convert an ML model into a REST API using Flask or FastAPI

Differences between Flask vs FastAPI for serving ML models

Handling JSON input/output, error messages, and input validation

Setting up logging and CORS for production-ready APIs

🧠 Convert Your ML Model into a REST API

Example: Iris Classifier using Scikit-learn.

🎯 Step 1: Train and Save the Model
# save_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

iris = load_iris()
X, y = iris.data, iris.target
model = RandomForestClassifier()
model.fit(X, y)

with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

🅾️ Option 1: Serve with Flask
📦 Install Dependencies
pip install flask scikit-learn

🧪 Flask App
# app_flask.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("iris_model.pkl", "rb"))

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)

🧪 Test Flask API
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

🅾️ Option 2: Serve with FastAPI
📦 Install Dependencies
pip install fastapi uvicorn scikit-learn

🧪 FastAPI App
# app_fastapi.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

class Features(BaseModel):
    features: list

app = FastAPI()
model = pickle.load(open("iris_model.pkl", "rb"))

@app.post("/predict")
def predict(data: Features):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

@app.get("/health")
def health():
    return {"status": "ok"}

🧪 Run FastAPI
uvicorn app_fastapi:app --reload

🧪 Test FastAPI
curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

⚖️ Flask vs FastAPI
Feature	Flask	FastAPI
Release Year	2010	2018
ASGI Support	❌ No	✅ Yes
Performance	Slower	Faster
Type Checking	❌ Manual	✅ Pydantic
API Docs	❌ Manual	✅ Auto (Swagger)
Async Support	⚠️ Limited	✅ Native
Learning Curve	Easy	Moderate
Community	Mature	Growing
Data Validation	❌ Manual	✅ Built-in
🔍 Final Verdict

Flask → simple, beginner-friendly, best for prototypes

FastAPI → modern, fastest, auto validation, ideal for production

📥 JSON Input/Output Example
Input (POST /predict)
{
  "features": [5.1, 3.5, 1.4, 0.2]
}

Output
{
  "prediction": 0
}

🛡️ Validation & Error Handling
FastAPI Validation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
import pickle, numpy as np

app = FastAPI()
model = pickle.load(open("iris_model.pkl", "rb"))

class Features(BaseModel):
    features: conlist(float, min_items=4, max_items=4)

@app.post("/predict")
def predict(data: Features):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

Auto Error Example

Input:

{ "features": [1, 2] }


FastAPI Response:

{
  "detail": [
    {
      "msg": "ensure this value has at least 4 items",
      "type": "value_error.list.min_items"
    }
  ]
}

Flask Manual Validation
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = data.get("features", [])

        if not isinstance(features, list) or len(features) != 4:
            return jsonify({"error": "features must be a list of 4 numbers"}), 400

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

📝 Logging & CORS (Production Ready)
FastAPI Logging + CORS
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import logging, pickle, numpy as np

logging.basicConfig(level=logging.INFO)
model = pickle.load(open("iris_model.pkl", "rb"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logging.info(f"Incoming {request.method} {request.url}")
    response = await call_next(request)
    return response

Flask Logging + CORS
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging, pickle, numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ml-api")

app = Flask(__name__)
CORS(app)
model = pickle.load(open("iris_model.pkl", "rb"))

@app.before_request
def log_request_info():
    logger.info(f"Incoming {request.method} request to {request.path}")

🔥 Challenges

Serve a Scikit-learn or TensorFlow model with Flask/FastAPI

Create /predict route

Add input validation

Add batch prediction support

Add logging to file

🤝 How to Participate?

✔️ Complete daily tasks
✔️ Document progress on GitHub, Medium, or Hashnode
✔️ Share your learnings

🌐 Follow Me

LinkedIn

GitHub