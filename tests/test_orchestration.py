from src.orchestration.service import orchestrate_inference

from src.orchestration.service import run_orchestrated_inference

from src.schemas.orchestration import (
    DecisionStatus,
    InferencePolicy,
    InferenceAttempt,
)


def test_accepts_high_confidence_primary():
    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    primary_attempt = InferenceAttempt(
        model_name="cnn",
        predicted_digit=7,
        confidence=0.982,
        succeeded=True,
        error_message=None,
    )

    decision = orchestrate_inference(
        primary_attempt,
        policy,
    )

    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.selected_model == "cnn"
    assert decision.predicted_digit == 7
    assert decision.confidence == 0.982
    assert decision.fallback_used is False
    assert decision.fallback_reason is None
    assert decision.review_required is False

def test_returns_uncertain_when_primary_is_low_confidence_without_fallback():
    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    primary_attempt = InferenceAttempt(
        model_name="ann",
        predicted_digit=7,
        confidence=0.61,
        succeeded=True,
        error_message=None,
    )

    decision = orchestrate_inference(
        primary_attempt,
        policy,
    )

    assert decision.status == DecisionStatus.UNCERTAIN
    assert decision.selected_model == "ann"
    assert decision.predicted_digit == 7
    assert decision.confidence == 0.61
    assert decision.decision_reason == "low_confidence"
    assert decision.fallback_used is False
    assert decision.review_required is True

def test_uses_fallback_when_primary_is_low_confidence():
    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    primary_attempt = InferenceAttempt(
        model_name="ann",
        predicted_digit=7,
        confidence=0.61,
        succeeded=True,
        error_message=None,
    )

    fallback_attempt = InferenceAttempt(
        model_name="cnn",
        predicted_digit=7,
        confidence=0.971,
        succeeded=True,
        error_message=None,
    )

    decision = orchestrate_inference(
        primary_attempt,
        policy,
        fallback_attempt=fallback_attempt,
    )

    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.selected_model == "cnn"
    assert decision.predicted_digit == 7
    assert decision.confidence == 0.971
    assert decision.fallback_used is True
    assert decision.fallback_reason == "primary_low_confidence"
    assert decision.review_required is False

def test_uses_fallback_when_primary_fails():
    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    primary_attempt = InferenceAttempt(
        model_name="ann",
        predicted_digit=None,
        confidence=None,
        succeeded=False,
        error_message="Primary model unavailable",
    )

    fallback_attempt = InferenceAttempt(
        model_name="cnn",
        predicted_digit=4,
        confidence=0.971,
        succeeded=True,
        error_message=None,
    )

    decision = orchestrate_inference(
        primary_attempt,
        policy,
        fallback_attempt=fallback_attempt,
    )

    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.selected_model == "cnn"
    assert decision.predicted_digit == 4
    assert decision.confidence == 0.971
    assert decision.fallback_used is True
    assert decision.fallback_reason == "primary_model_failed"
    assert decision.review_required is False

def test_returns_uncertain_when_fallback_is_low_confidence():
    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    primary_attempt = InferenceAttempt(
        model_name="ann",
        predicted_digit=7,
        confidence=0.61,
        succeeded=True,
        error_message=None,
    )

    fallback_attempt = InferenceAttempt(
        model_name="cnn",
        predicted_digit=3,
        confidence=0.72,
        succeeded=True,
        error_message=None,
    )

    decision = orchestrate_inference(
        primary_attempt,
        policy,
        fallback_attempt=fallback_attempt,
    )

    assert decision.status == DecisionStatus.UNCERTAIN
    assert decision.selected_model == "cnn"
    assert decision.predicted_digit == 3
    assert decision.confidence == 0.72
    assert decision.decision_reason == "fallback_low_confidence"
    assert decision.fallback_used is True
    assert decision.fallback_reason == "primary_low_confidence"
    assert decision.review_required is True

def test_returns_failed_when_primary_and_fallback_both_fail():
    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    primary_attempt = InferenceAttempt(
        model_name="ann",
        predicted_digit=None,
        confidence=None,
        succeeded=False,
        error_message="Primary model unavailable",
    )

    fallback_attempt = InferenceAttempt(
        model_name="cnn",
        predicted_digit=None,
        confidence=None,
        succeeded=False,
        error_message="Fallback model unavailable",
    )

    decision = orchestrate_inference(
        primary_attempt,
        policy,
        fallback_attempt=fallback_attempt,
    )

    assert decision.status == DecisionStatus.FAILED
    assert decision.selected_model is None
    assert decision.predicted_digit is None
    assert decision.confidence is None
    assert decision.decision_reason == "fallback_model_failed"
    assert decision.fallback_used is True
    assert decision.fallback_reason == "primary_model_failed"
    assert decision.review_required is True

def test_does_not_execute_fallback_when_primary_is_accepted():
    fallback_call_count = 0

    def primary_prediction():
        return 7, 0.982

    def fallback_prediction():
        nonlocal fallback_call_count
        fallback_call_count += 1
        return 3, 0.991

    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    decision = run_orchestrated_inference(
        primary_model_name="cnn",
        primary_predict_fn=primary_prediction,
        policy=policy,
        fallback_model_name="ann",
        fallback_predict_fn=fallback_prediction,
    )

    assert decision.status == DecisionStatus.ACCEPTED
    assert decision.selected_model == "cnn"
    assert decision.predicted_digit == 7
    assert decision.confidence == 0.982
    assert decision.fallback_used is False
    assert decision.fallback_reason is None

    assert fallback_call_count == 0