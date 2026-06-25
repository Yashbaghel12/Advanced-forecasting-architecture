import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# Humne jo script pehle likhi thi, yeh function wahan se use hoga

app = FastAPI(title="Team Ailurophile DataCo Demand Forecasting API")

# Input Schema validation define karein
class DemandPredictionRequest(BaseModel):
    context_features: list  # [[rolling_std_7, rolling_std_30, cv]]
    chronos_forecast: list  # [dl_predictions]

@app.post("/predict")
def predict_endpoint(request: DemandPredictionRequest):
    try:
        # Array dimensions parse karein
        ctx_array = np.array(request.context_features)
        chronos_array = np.array(request.chronos_forecast)
        
        # Hamare finalized inference layer ko call karein
        predictions, alphas = predict_realtime_demand(ctx_array, chronos_array)
        
        # Response return karein JSON format mein
        return {
            "status": "success",
            "blended_demand_forecast": predictions.tolist(),
            "gating_allocation_alpha": alphas.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
