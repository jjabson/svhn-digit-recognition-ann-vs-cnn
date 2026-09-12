from src.schemas.orchestration import (
    DecisionStatus,
    InferenceDecision,
)
from tools.reliability.failure_analysis import (
    FailureScenarioResult,
    evaluate_failure_scenario,
    failure_scenario_from_decision,
    summarize_failure_reliability,
    run_failure_reliability_scenarios,
)

from src.orchestration.service import run_orchestrated_inference
from src.schemas.orchestration import InferencePolicy

def test_failure_scenario_result_stores_observed_outcome():
    result = FailureScenarioResult(
        scenario_name="primary model failure",
        status=DecisionStatus.FAILED,
        fallback_used=False,
        review_required=True,
        decision_reason="Primary inference failed.",
        fallback_reason=None,
    )

    assert result.scenario_name == "primary model failure"
    assert result.status == DecisionStatus.FAILED
    assert result.fallback_used is False
    assert result.review_required is True
    assert result.decision_reason == "Primary inference failed."

def test_failure_scenario_from_decision():
    decision = InferenceDecision(
        selected_model=None,
        predicted_digit=None,
        confidence=None,
        status=DecisionStatus.FAILED,
        decision_reason="Primary inference failed.",
        fallback_used=False,
        fallback_reason=None,
        review_required=True,
    )

    result = failure_scenario_from_decision(
        scenario_name="primary model failure",
        decision=decision,
    )

    assert result.scenario_name == "primary model failure"
    assert result.status == DecisionStatus.FAILED
    assert result.fallback_used is False
    assert result.review_required is True
    assert result.decision_reason == "Primary inference failed."
    assert result.fallback_reason is None

def test_evaluate_primary_accepted_scenario():
    decision = InferenceDecision(
        selected_model="svhn_cnn",
        predicted_digit=7,
        confidence=0.98,
        status=DecisionStatus.ACCEPTED,
        decision_reason="Primary prediction accepted.",
        fallback_used=False,
        fallback_reason=None,
        review_required=False,
    )

    result = evaluate_failure_scenario(
        scenario_name="primary accepted",
        decision=decision,
    )

    assert result.scenario_name == "primary accepted"
    assert result.status == DecisionStatus.ACCEPTED
    assert result.fallback_used is False
    assert result.review_required is False
    assert result.decision_reason == "Primary prediction accepted."

def test_summarize_failure_reliability():
    results = [
        FailureScenarioResult(
            scenario_name="primary accepted",
            status=DecisionStatus.ACCEPTED,
            fallback_used=False,
            review_required=False,
            decision_reason="Primary prediction accepted.",
            fallback_reason=None,
        ),
        FailureScenarioResult(
            scenario_name="primary failure",
            status=DecisionStatus.FAILED,
            fallback_used=False,
            review_required=True,
            decision_reason="Primary inference failed.",
            fallback_reason=None,
        ),
        FailureScenarioResult(
            scenario_name="fallback accepted",
            status=DecisionStatus.ACCEPTED,
            fallback_used=True,
            review_required=False,
            decision_reason="Fallback prediction accepted.",
            fallback_reason="primary_model_failed",
        ),
    ]

    summary = summarize_failure_reliability(
        results
    )

    assert summary.total_scenarios == 3
    assert summary.failed_scenarios == 1
    assert summary.fallback_scenarios == 1
    assert summary.review_required_scenarios == 1

    assert summary.failure_rate == 1 / 3
    assert summary.fallback_usage_rate == 1 / 3
    assert summary.review_required_rate == 1 / 3

def test_summarize_failure_reliability_handles_empty_results():
    summary = summarize_failure_reliability([])

    assert summary.total_scenarios == 0
    assert summary.failed_scenarios == 0
    assert summary.fallback_scenarios == 0
    assert summary.review_required_scenarios == 0

    assert summary.failure_rate == 0.0
    assert summary.fallback_usage_rate == 0.0
    assert summary.review_required_rate == 0.0

def test_primary_failure_recovers_with_fallback():
    def primary_prediction():
        raise RuntimeError("Primary model unavailable")

    def fallback_prediction():
        return 4, 0.971

    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    decision = run_orchestrated_inference(
        primary_model_name="ann",
        primary_predict_fn=primary_prediction,
        policy=policy,
        fallback_model_name="cnn",
        fallback_predict_fn=fallback_prediction,
    )

    result = failure_scenario_from_decision(
        scenario_name="primary failure recovered by fallback",
        decision=decision,
    )

    assert result.status == DecisionStatus.ACCEPTED
    assert result.fallback_used is True
    assert result.fallback_reason == "primary_model_failed"
    assert result.review_required is False

def test_primary_and_fallback_failure_requires_review():
    def primary_prediction():
        raise RuntimeError("Primary model unavailable")

    def fallback_prediction():
        raise RuntimeError("Fallback model unavailable")

    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    decision = run_orchestrated_inference(
        primary_model_name="ann",
        primary_predict_fn=primary_prediction,
        policy=policy,
        fallback_model_name="cnn",
        fallback_predict_fn=fallback_prediction,
    )

    result = failure_scenario_from_decision(
        scenario_name="primary and fallback failure",
        decision=decision,
    )

    assert result.status == DecisionStatus.FAILED
    assert result.fallback_used is True
    assert result.fallback_reason == "primary_model_failed"
    assert result.review_required is True
    assert result.decision_reason == "fallback_model_failed"

def test_low_confidence_primary_recovers_with_fallback():
    def primary_prediction():
        return 7, 0.61

    def fallback_prediction():
        return 7, 0.971

    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    decision = run_orchestrated_inference(
        primary_model_name="ann",
        primary_predict_fn=primary_prediction,
        policy=policy,
        fallback_model_name="cnn",
        fallback_predict_fn=fallback_prediction,
    )

    result = failure_scenario_from_decision(
        scenario_name="low confidence primary recovered by fallback",
        decision=decision,
    )

    assert result.status == DecisionStatus.ACCEPTED
    assert result.fallback_used is True
    assert result.fallback_reason == "primary_low_confidence"
    assert result.review_required is False

def test_low_confidence_primary_and_fallback_requires_review():
    def primary_prediction():
        return 7, 0.61

    def fallback_prediction():
        return 3, 0.72

    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    decision = run_orchestrated_inference(
        primary_model_name="ann",
        primary_predict_fn=primary_prediction,
        policy=policy,
        fallback_model_name="cnn",
        fallback_predict_fn=fallback_prediction,
    )

    result = failure_scenario_from_decision(
        scenario_name="low confidence primary and fallback",
        decision=decision,
    )

    assert result.status == DecisionStatus.UNCERTAIN
    assert result.fallback_used is True
    assert result.fallback_reason == "primary_low_confidence"
    assert result.review_required is True
    assert result.decision_reason == "fallback_low_confidence"

def test_run_failure_reliability_scenarios():
    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    results = run_failure_reliability_scenarios(
        policy=policy,
    )

    summary = summarize_failure_reliability(
        results
    )

    assert len(results) == 5

    assert summary.total_scenarios == 5
    assert summary.failed_scenarios == 1
    assert summary.fallback_scenarios == 4
    assert summary.review_required_scenarios == 2

    assert summary.failure_rate == 1 / 5
    assert summary.fallback_usage_rate == 4 / 5
    assert summary.review_required_rate == 2 / 5