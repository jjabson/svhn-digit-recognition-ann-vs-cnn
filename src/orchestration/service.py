from collections.abc import Callable

from src.orchestration.execution import execute_model_attempt
from src.orchestration.policy import (
    create_fallback_decision,
    create_primary_decision,
    should_accept_attempt,
)
from src.schemas.orchestration import (
    InferenceDecision,
    InferencePolicy,
    InferenceAttempt,
)

def run_orchestrated_inference(
    primary_model_name: str,
    primary_predict_fn: Callable[[], tuple[int, float]],
    policy: InferencePolicy,
    fallback_model_name: str | None = None,
    fallback_predict_fn: Callable[[], tuple[int, float]] | None = None,
) -> InferenceDecision:
    """
    Execute primary inference and, when needed, optionally execute
    a fallback model before returning the final decision.
    """

    primary_attempt = execute_model_attempt(
        model_name=primary_model_name,
        predict_fn=primary_predict_fn,
    )

    if should_accept_attempt(primary_attempt, policy):
        return create_primary_decision(
            primary_attempt,
            policy,
        )

    if fallback_model_name is None or fallback_predict_fn is None:
        return create_primary_decision(
            primary_attempt,
            policy,
        )

    fallback_attempt = execute_model_attempt(
        model_name=fallback_model_name,
        predict_fn=fallback_predict_fn,
    )

    return create_fallback_decision(
        primary_attempt,
        fallback_attempt,
        policy,
    )


def orchestrate_inference(
    primary_attempt: InferenceAttempt,
    policy: InferencePolicy,
    fallback_attempt: InferenceAttempt | None = None,
) -> InferenceDecision:
    """
    Coordinate primary and optional fallback inference attempts
    according to the configured orchestration policy.
    """

    if should_accept_attempt(primary_attempt, policy):
        return create_primary_decision(
            primary_attempt,
            policy,
        )

    if fallback_attempt is None:
        return create_primary_decision(
            primary_attempt,
            policy,
        )

    return create_fallback_decision(
        primary_attempt,
        fallback_attempt,
        policy,
    )