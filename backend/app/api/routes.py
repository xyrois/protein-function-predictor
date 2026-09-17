from fastapi import APIRouter, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse
from app.schemas.health import HealthResponse
from app.ml.inference import predict, DEVICE, model

router = APIRouter()

@router.get("/health", response_model=HealthResponse, tags=["Status"])
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        device=str(DEVICE)
    )

@router.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_protein_function(payload: PredictionRequest):
    try:
        raw_result = predict(payload.sequence)
        return PredictionResponse(
            prediction=raw_result["prediction"],
            confidence=raw_result["confidence"],
            top_predictions=raw_result["top_predictions"],
            sequence_length=len(payload.sequence)
        )
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")