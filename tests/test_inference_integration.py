from pathlib import Path

from src.inference import SVHNPredictor
from src.orchestration.adapters import create_svhn_predict_fn
from src.orchestration.service import run_orchestrated_inference
from src.schemas.orchestration import (
    DecisionStatus,
    InferencePolicy,
)


def test_real_cnn_prediction_flows_through_orchestration():
    image_path = Path(
        "sample_images/digit_7_true7.png"
    )

    image_bytes = image_path.read_bytes()

    predictor = SVHNPredictor()

    predict_fn = create_svhn_predict_fn(
        predictor=predictor,
        image_bytes=image_bytes,
    )

    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    decision = run_orchestrated_inference(
        primary_model_name="cnn",
        primary_predict_fn=predict_fn,
        policy=policy,
    )

    assert decision.selected_model == "cnn"
    assert decision.predicted_digit == 7
    assert decision.confidence is not None
    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.fallback_used is False
    assert decision.review_required is False