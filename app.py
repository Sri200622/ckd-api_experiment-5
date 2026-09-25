from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import time
import pandas as pd

app = FastAPI(title="CKD Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = joblib.load("model/ckd_pipeline.joblib")
pipeline = bundle["pipeline"]
FEATURES = bundle["features"]
METRICS = bundle["metrics"]

stats = {"total_requests": 0, "successful_predictions": 0, "failed_requests": 0, "response_times": []}

@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/predict":
        stats["total_requests"] += 1
        if response.status_code != 200:
            stats["failed_requests"] += 1
    return response

class PatientInput(BaseModel):
    age: float = Field(..., ge=0)
    bp: float
    sg: float
    al: float
    bgr: float
    bu: float
    sc: float
    sod: float
    hemo: float
    wbcc: float

@app.get("/")
def root():
    return {"service": "CKD Prediction API", "version": "1.0", "status": "running"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ready")
def ready():
    return {"ready": pipeline is not None}

@app.get("/model-info")
def model_info():
    return {
        "model_name": bundle["model_name"],
        "version": bundle["version"],
        "algorithm": "RandomForestClassifier",
        "num_features": len(FEATURES),
        "features": FEATURES,
        "evaluation_metrics": METRICS,
    }

@app.post("/predict")
def predict(data: PatientInput):
    start = time.time()
    row = pd.DataFrame([data.dict()])[FEATURES]
    try:
        pred = pipeline.predict(row)[0]
        prob = pipeline.predict_proba(row)[0].tolist()
        stats["successful_predictions"] += 1
        stats["response_times"].append(time.time() - start)
        return {
            "prediction": "CKD" if pred == 1 else "Not CKD",
            "model_output_probability": prob,
            "note": "This is a model output, not a medical diagnosis."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/metrics")
def metrics():
    avg_time = sum(stats["response_times"]) / len(stats["response_times"]) if stats["response_times"] else 0
    return {
        "total_requests": stats["total_requests"],
        "successful_predictions": stats["successful_predictions"],
        "failed_requests": stats["failed_requests"],
        "avg_response_time_sec": round(avg_time, 5),
    }