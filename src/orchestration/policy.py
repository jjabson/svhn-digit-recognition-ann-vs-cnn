from src.schemas.orchestration import (
    DecisionStatus,
    InferenceAttempt,
    InferenceDecision,
    InferencePolicy,
)

def create_primary_decision(
    attempt: InferenceAttempt,
    policy: InferencePolicy,
) -> InferenceDecision:
    """
    Convert a primary inference attempt into an orchestration
    decision before any fallback model is attempted.
    """

    if not attempt.succeeded:
        return InferenceDecision(
            selected_model=None,
            predicted_digit=None,
            confidence=None,
            status=DecisionStatus.FAILED,
            decision_reason="primary_model_failed",
            fallback_used=False,
            fallback_reason=None,
            review_required=True,
        )

    if should_accept_attempt(attempt, policy):
        return InferenceDecision(
            selected_model=attempt.model_name,
            predicted_digit=attempt.predicted_digit,
            confidence=attempt.confidence,
            status=DecisionStatus.ACCEPTED,
            decision_reason=None,
            fallback_used=False,
            fallback_reason=None,
            review_required=False,
        )

    return InferenceDecision(
        selected_model=attempt.model_name,
        predicted_digit=attempt.predicted_digit,
        confidence=attempt.confidence,
        status=DecisionStatus.UNCERTAIN,
        decision_reason="low_confidence",
        fallback_used=False,
        fallback_reason=None,
        review_required=True,
    )

def should_accept_attempt(
    attempt: InferenceAttempt,
    policy: InferencePolicy,
) -> bool:
    """
    Return True when an inference attempt succeeded and its
    confidence meets or exceeds the configured threshold.
    """

    if not attempt.succeeded:
        return False

    if attempt.confidence is None:
        return False

    return attempt.confidence >= policy.confidence_threshold

def create_fallback_decision(
    primary_attempt: InferenceAttempt,
    fallback_attempt: InferenceAttempt,
    policy: InferencePolicy,
) -> InferenceDecision:
    """
    Produce the final orchestration decision after a fallback
    inference attempt has been performed.
    """

    if primary_attempt.succeeded:
        fallback_reason = "primary_low_confidence"
    else:
        fallback_reason = "primary_model_failed"

    if should_accept_attempt(fallback_attempt, policy):
        return InferenceDecision(
            selected_model=fallback_attempt.model_name,
            predicted_digit=fallback_attempt.predicted_digit,
            confidence=fallback_attempt.confidence,
            status=DecisionStatus.ACCEPTED,
            decision_reason=None,
            fallback_used=True,
            fallback_reason=fallback_reason,
            review_required=False,
        )

    if fallback_attempt.succeeded:
        return InferenceDecision(
            selected_model=fallback_attempt.model_name,
            predicted_digit=fallback_attempt.predicted_digit,
            confidence=fallback_attempt.confidence,
            status=DecisionStatus.UNCERTAIN,
            decision_reason="fallback_low_confidence",
            fallback_used=True,
            fallback_reason=fallback_reason,
            review_required=True,
        )

    return InferenceDecision(
        selected_model=None,
        predicted_digit=None,
        confidence=None,
        status=DecisionStatus.FAILED,
        decision_reason="fallback_model_failed",
        fallback_used=True,
        fallback_reason=fallback_reason,
        review_required=True,
    )