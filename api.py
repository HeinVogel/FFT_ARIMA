# api.py — FastAPI JSON endpoints
from fastapi import FastAPI
import json, glob, pandas as pd
app = FastAPI(title="Market Cycles API")

@app.get("/metrics")
def get_metrics():
    with open(glob.glob("logs/metrics_*.json")[0]) as f:
        return json.load(f)

@app.get("/forecast")
def get_forecast():
    df = pd.read_csv(glob.glob("forecast_*.csv")[0])
    return df.to_dict(orient="records")