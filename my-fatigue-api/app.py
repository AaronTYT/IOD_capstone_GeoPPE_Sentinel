from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load saved model (SVM Pipeline)
model = joblib.load("svm_clean.pkl")

app = FastAPI(
    title="Fatigue Prediction API",
    description="Predict worker fatigue level (Low / Moderate / High / Severe)",
    version="1.0"
)

class WorkerData(BaseModel):
    sleep_hours: float
    shift_hours: float
    opf_minutes: float
    ppe_violations: int
    high_risk_events: int
    break_compliance: float
    movement_score: float

@app.get("/")
def root():
    return {"status": "Fatigue API is running!"}

@app.post("/predict")
def predict(data: WorkerData):

    # Convert input into model format
    X = np.array([[
        data.sleep_hours,
        data.shift_hours,
        data.opf_minutes,
        data.ppe_violations,
        data.high_risk_events,
        data.break_compliance,
        data.movement_score
    ]])

    # Probability for each fatigue class
    probs = model.predict_proba(X)[0]

    # Predicted class (string)
    pred_class = model.classes_[np.argmax(probs)]

    return {
        "fatigue_label": pred_class,
        "probabilities": {
            cls: float(p)
            for cls, p in zip(model.classes_, probs)
        }
    }
