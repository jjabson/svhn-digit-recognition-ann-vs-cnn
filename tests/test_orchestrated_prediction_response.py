from src.schemas.orchestration import DecisionStatus, InferenceDecision
from src.schemas.prediction import (
    OrchestratedPredictionResponse,
    inference_decision_to_response,
)

def test_inference_decision_converts_to_orchestrated_response():
    decision = InferenceDecision(
        selected_model="cnn",
        predicted_digit=8,
        confidence=0.818678081035614,
        status=DecisionStatus.UNCERTAIN,
        decision_reason="low_confidence",
        fallback_used=False,
        fallback_reason=None,
        review_required=True,
    )

    response = inference_decision_to_response(decision)

    assert response.predicted_digit == 8
    assert response.confidence == 0.818678081035614
    assert response.confidence_percent == "81.87%"
    assert response.selected_model == "cnn"
    assert response.status == "uncertain"
    assert response.decision_reason == "low_confidence"
    assert response.fallback_used is False
    assert response.fallback_reason is None
    assert response.review_required is True

def test_orchestrated_prediction_response_accepts_uncertain_decision():
    response = OrchestratedPredictionResponse(
        predicted_digit=8,
        confidence=0.818678081035614,
        confidence_percent="81.87%",
        selected_model="cnn",
        status="uncertain",
        decision_reason="low_confidence",
        fallback_used=False,
        fallback_reason=None,
        review_required=True,
    )

    assert response.predicted_digit == 8
    assert response.confidence == 0.818678081035614
    assert response.confidence_percent == "81.87%"
    assert response.selected_model == "cnn"
    assert response.status == "uncertain"
    assert response.decision_reason == "low_confidence"
    assert response.fallback_used is False
    assert response.fallback_reason is None
    assert response.review_required is True