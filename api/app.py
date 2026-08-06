from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference import SVHNPredictor
from src.preprocessing import InvalidImageError


predictor: SVHNPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = SVHNPredictor()
    yield
    predictor = None


app = FastAPI(
    title="SVHN Digit Recognition API",
    description=(
        "Classifies cropped digit images using a convolutional "
        "neural network trained on SVHN data."
    ),
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "SVHN Digit Recognition API",
        "documentation": "/docs"
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict")
async def predict_digit(
    file: UploadFile = File(...)
) -> dict:
    allowed_content_types = {
        "image/png",
        "image/jpeg"
    }

    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail="Only PNG and JPEG images are supported."
        )

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is unavailable."
        )

    try:
        image_bytes = await file.read()
        result = predictor.predict(image_bytes)
        return result

    except InvalidImageError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        ) from exc