# app/api/predict.py
import pandas as pd
from fastapi import APIRouter

from app.model_api.pipeline_adapter import predict_from_saved
from app.schemas.property import PropertyData

router = APIRouter()


@router.post("/predict")
def predict(data: PropertyData):
    df = pd.DataFrame([data.model_dump()])  # pydantic v2
    preds = predict_from_saved(df)
    return preds.iloc[0].to_dict()


@router.get("/predict/schema")
def predict_schema():
    return PropertyData.model_json_schema()
