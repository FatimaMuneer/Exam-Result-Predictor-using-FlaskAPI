from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel

# ■■ Load model when the server starts ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
ml_model = {}  # we store the model in a dict so we can access it in routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs ONCE when server starts
    ml_model["predictor"] = joblib.load("model.pkl")
    print("Model loaded successfully")
    yield
    # runs ONCE when server shuts down (cleanup)
    ml_model.clear()


app = FastAPI(title="Student Pass/Fail Predictor", lifespan=lifespan)

from pydantic import BaseModel


# What the caller sends
class StudentFeatures(BaseModel):
    study_hours: float  # hours studied per day (e.g. 4.5)
    prev_score: float  # previous exam score (e.g. 65.0)


# What we send back
class PredictionResult(BaseModel):
    prediction: int  # 0 = Fail, 1 = Pass
    result_label: str  # "Pass" or "Fail" (human-readable)
    confidence: float  # probability of the predicted class
    
@app.post("/predict", response_model=PredictionResult)
def predict(features: StudentFeatures):
    model = ml_model.get("predictor")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build the input array the model expects: [[study_hours, prev_score]]
    input_data = np.array([[features.study_hours, features.prev_score]])

    # Get prediction (0 or 1)
    prediction = int(model.predict(input_data)[0])

    # Get confidence — probability of each class, take the predicted one
    probabilities = model.predict_proba(input_data)[0]   # logistic regression
    confidence = round(float(probabilities[prediction]), 3)

    return PredictionResult(
        prediction=prediction,
        result_label="Pass" if prediction == 1 else "Fail",
        confidence=confidence,
    )


# Health check — useful to confirm the server is running
@app.get("/")
def health():
    return {"status": "running", "model_loaded": "predictor" in ml_model}