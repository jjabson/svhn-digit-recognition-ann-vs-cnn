from dataclasses import dataclass

from src.schemas.orchestration import (
    DecisionStatus,
    InferenceDecision,
)

from src.orchestration.service import run_orchestrated_inference
from src.schemas.orchestration import InferencePolicy

@dataclass(frozen=True)
class FailureScenarioResult:
    scenario_name: str
    status: DecisionStatus
    fallback_used: bool
    fallback_reason: str | None
    review_required: bool
    decision_reason: str | None


def failure_scenario_from_decision(
    scenario_name: str,
    decision: InferenceDecision,
) -> FailureScenarioResult:
    return FailureScenarioResult(
        scenario_name=scenario_name,
        status=decision.status,
        fallback_used=decision.fallback_used,
        fallback_reason=decision.fallback_reason,
        review_required=decision.review_required,
        decision_reason=decision.decision_reason,
    )

def evaluate_failure_scenario(
    scenario_name: str,
    decision: InferenceDecision,
) -> FailureScenarioResult:
    return failure_scenario_from_decision(
        scenario_name=scenario_name,
        decision=decision,
    )

@dataclass(frozen=True)
class FailureReliabilitySummary:
    total_scenarios: int
    failed_scenarios: int
    fallback_scenarios: int
    review_required_scenarios: int

    @property
    def failure_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.failed_scenarios / self.total_scenarios

    @property
    def fallback_usage_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return self.fallback_scenarios / self.total_scenarios

    @property
    def review_required_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return (
            self.review_required_scenarios
            / self.total_scenarios
        )

def summarize_failure_reliability(
    results: list[FailureScenarioResult],
) -> FailureReliabilitySummary:
    total_scenarios = len(results)

    failed_scenarios = sum(
        result.status == DecisionStatus.FAILED
        for result in results
    )

    fallback_scenarios = sum(
        result.fallback_used
        for result in results
    )

    review_required_scenarios = sum(
        result.review_required
        for result in results
    )

    return FailureReliabilitySummary(
        total_scenarios=total_scenarios,
        failed_scenarios=failed_scenarios,
        fallback_scenarios=fallback_scenarios,
        review_required_scenarios=review_required_scenarios,
    )

def run_failure_reliability_scenarios(
    policy: InferencePolicy,
) -> list[FailureScenarioResult]:
    scenarios: list[FailureScenarioResult] = []

    accepted_primary = run_orchestrated_inference(
        primary_model_name="cnn",
        primary_predict_fn=lambda: (7, 0.98),
        policy=policy,
        fallback_model_name="ann",
        fallback_predict_fn=lambda: (3, 0.99),
    )

    scenarios.append(
        failure_scenario_from_decision(
            scenario_name="primary accepted",
            decision=accepted_primary,
        )
    )

    def failed_primary():
        raise RuntimeError("Primary model unavailable")

    recovered_failure = run_orchestrated_inference(
        primary_model_name="ann",
        primary_predict_fn=failed_primary,
        policy=policy,
        fallback_model_name="cnn",
        fallback_predict_fn=lambda: (4, 0.97),
    )

    scenarios.append(
        failure_scenario_from_decision(
            scenario_name="primary failure recovered by fallback",
            decision=recovered_failure,
        )
    )

    def failed_fallback():
        raise RuntimeError("Fallback model unavailable")

    unrecovered_failure = run_orchestrated_inference(
        primary_model_name="ann",
        primary_predict_fn=failed_primary,
        policy=policy,
        fallback_model_name="cnn",
        fallback_predict_fn=failed_fallback,
    )

    scenarios.append(
        failure_scenario_from_decision(
            scenario_name="primary and fallback failure",
            decision=unrecovered_failure,
        )
    )

    low_confidence_recovery = run_orchestrated_inference(
        primary_model_name="ann",
        primary_predict_fn=lambda: (7, 0.61),
        policy=policy,
        fallback_model_name="cnn",
        fallback_predict_fn=lambda: (7, 0.97),
    )

    scenarios.append(
        failure_scenario_from_decision(
            scenario_name="low confidence primary recovered by fallback",
            decision=low_confidence_recovery,
        )
    )

    low_confidence_uncertain = run_orchestrated_inference(
        primary_model_name="ann",
        primary_predict_fn=lambda: (7, 0.61),
        policy=policy,
        fallback_model_name="cnn",
        fallback_predict_fn=lambda: (3, 0.72),
    )

    scenarios.append(
        failure_scenario_from_decision(
            scenario_name="low confidence primary and fallback",
            decision=low_confidence_uncertain,
        )
    )

    return scenarios
