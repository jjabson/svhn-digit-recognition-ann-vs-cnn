from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from src.inference import SVHNPredictor
from src.preprocessing import InvalidImageError
from src.schemas.prediction import PredictionResponse

from src.schemas.evaluation import (
    ClassPerformanceResponse,
    ConfusionMatrixResponse,
    EvaluationInsightsResponse,
    EvaluationSummaryResponse,
    class_metrics_list_to_response,
    class_metrics_to_response,
    confusion_matrix_to_response,
    evaluation_insights_to_response,
    evaluation_summary_to_response,
)

from tools.evaluation.evaluation_service import (
    get_class_metrics,
    get_evaluation_insights,
    get_primary_evaluation,
)

from src.schemas.model import (
    ModelSummaryResponse,
    model_architecture_to_response,
)

from tools.model.model_service import get_model_architecture

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
        "Production-oriented API for SVHN digit classification, "
        "model evaluation insights, and ML engineering diagnostics."
    ),
    version="1.2.0",
    lifespan=lifespan,
    contact={
        "name": "SVHN ML Engineering Project",
    },
    openapi_tags=[
        {
            "name": "System",
            "description": (
                "Service health and API information."
            ),
        },
        {
            "name": "Prediction",
            "description": (
                "Run CNN inference on uploaded digit images."
            ),
        },
        {
            "name": "Evaluation",
            "description": (
                "Inspect persisted model-evaluation metrics "
                "and derived evaluation intelligence."
            ),
        },
    ],
)

@app.get(
    "/",
    tags=["System"],
    summary="API information",
)
def root():
    return {
        "message": "SVHN Digit Recognition API",
        "docs": "/docs",
    }

@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
)
def health()-> dict[str, str]:
    return {"status": "healthy"}


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict a digit",
    description=(
        "Classifies an uploaded SVHN-style digit image using the "
        "trained CNN model and returns the predicted digit, confidence, "
        "and probability distribution across all digit classes."
    ),
)

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


@app.get(
    "/evaluation/insights",
    response_model=EvaluationInsightsResponse,
    tags=["Evaluation"],
    summary="Get evaluation insights",
    description=(
        "Returns high-level insights derived from the primary "
        "independent model evaluation."
    ),
)

def evaluation_insights() -> EvaluationInsightsResponse:
    """
    Return high-level insights from the primary model evaluation.
    """
    evaluation = get_primary_evaluation()

    insights = get_evaluation_insights(
        evaluation
    )

    return evaluation_insights_to_response(
        insights
    )

@app.get(
    "/evaluation/summary",
    response_model=EvaluationSummaryResponse,
    tags=["Evaluation"],
    summary="Get evaluation summary",
    description=(
        "Returns headline performance metrics from the primary "
        "independent model evaluation."
    ),
)
def evaluation_summary() -> EvaluationSummaryResponse:
    evaluation = get_primary_evaluation()

    return evaluation_summary_to_response(
        evaluation
    )

@app.get(
    "/evaluation/classes",
    response_model=list[ClassPerformanceResponse],
    tags=["Evaluation"],
    summary="Get per-class evaluation metrics",
    description=(
            "Returns precision, recall, F1 score, and sample count "
            "for each digit class in the primary independent evaluation."
    ),
)
def evaluation_classes() -> list[ClassPerformanceResponse]:
    evaluation = get_primary_evaluation()

    return class_metrics_list_to_response(
        evaluation.class_metrics
    )

@app.get(
    "/evaluation/classes/{digit}",
    response_model=ClassPerformanceResponse,
    responses={
        404: {
            "description": "Digit class not found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "No evaluation metrics found for digit 12."
                    }
                }
            },
        }
    },
    tags=["Evaluation"],
    summary="Get metrics for a digit",
    description=(
        "Returns precision, recall, F1 score, and sample count "
        "for a specific digit from the primary independent evaluation."
    ),
)
def evaluation_class(
    digit: int,
) -> ClassPerformanceResponse:
    evaluation = get_primary_evaluation()

    metrics = get_class_metrics(
        evaluation,
        digit,
    )

    if metrics is None:
        raise HTTPException(
            status_code=404,
            detail=f"No evaluation metrics found for digit {digit}.",
        )

    return class_metrics_to_response(metrics)



@app.get(
    "/evaluation/confusion-matrix",
    response_model=ConfusionMatrixResponse,
    tags=["Evaluation"],
    summary="Get confusion matrix",
    description=(
        "Returns the confusion matrix from the primary independent "
        "model evaluation. Rows represent true digit classes and "
        "columns represent predicted digit classes."
    ),
)
def evaluation_confusion_matrix() -> ConfusionMatrixResponse:
    evaluation = get_primary_evaluation()

    return confusion_matrix_to_response(evaluation)

@app.get(
    "/model/summary",
    response_model=ModelSummaryResponse,
    tags=["Model"],
    summary="Get model summary",
    description=(
        "Returns key architecture and capacity information "
        "for the trained SVHN CNN model."
    ),
)
def model_summary() -> ModelSummaryResponse:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available.",
        )

    architecture = get_model_architecture(
        predictor.model
    )

    return model_architecture_to_response(
        architecture
    )