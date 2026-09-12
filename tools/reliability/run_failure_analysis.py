from src.schemas.orchestration import InferencePolicy
from tools.reliability.failure_analysis import (
    run_failure_reliability_scenarios,
    summarize_failure_reliability,
)


def main() -> None:
    policy = InferencePolicy(
        confidence_threshold=0.90,
    )

    results = run_failure_reliability_scenarios(
        policy=policy,
    )

    summary = summarize_failure_reliability(
        results
    )

    print("Failure & Fallback Reliability Analysis")
    print("---------------------------------------")
    print(f"Total Scenarios: {summary.total_scenarios}")
    print(f"Failed Scenarios: {summary.failed_scenarios}")
    print(f"Fallback Scenarios: {summary.fallback_scenarios}")
    print(
        f"Review-Required Scenarios: "
        f"{summary.review_required_scenarios}"
    )
    print(
        f"Failure Rate: "
        f"{summary.failure_rate:.2%}"
    )
    print(
        f"Fallback Usage Rate: "
        f"{summary.fallback_usage_rate:.2%}"
    )
    print(
        f"Review-Required Rate: "
        f"{summary.review_required_rate:.2%}"
    )

    print()
    print("Scenario Results")
    print("----------------")

    for result in results:
        print(
            f"{result.scenario_name}: "
            f"status={result.status.value}, "
            f"fallback_used={result.fallback_used}, "
            f"fallback_reason={result.fallback_reason}, "
            f"review_required={result.review_required}, "
            f"decision_reason={result.decision_reason}"
        )


if __name__ == "__main__":
    main()