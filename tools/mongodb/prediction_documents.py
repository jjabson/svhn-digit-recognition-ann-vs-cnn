from __future__ import annotations

import random
from collections import Counter
from typing import Any

from tools.reliability.confidence_analysis import (
    ConfidenceObservation,
)


DEFAULT_MODEL_VERSION = "cnn_v2"
DEFAULT_EVALUATION_PROTOCOL = "Historical Stratified Holdout"
DEFAULT_SAMPLES_PER_DIGIT = 50
DEFAULT_RANDOM_SEED = 42


def select_balanced_observations(
    observations: list[ConfidenceObservation],
    samples_per_digit: int = DEFAULT_SAMPLES_PER_DIGIT,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> list[tuple[int, ConfidenceObservation]]:
    """
    Select a deterministic balanced sample of prediction observations.

    The integer paired with each observation is its original persisted
    observation_index.

    Because load_reliability_observations() returns rows ordered by
    observation_index, enumerating the loaded observations reconstructs
    that persisted index.
    """

    if samples_per_digit <= 0:
        raise ValueError(
            "samples_per_digit must be greater than zero."
        )

    observations_by_digit: dict[
        int,
        list[tuple[int, ConfidenceObservation]],
    ] = {
        digit: []
        for digit in range(10)
    }

    for observation_index, observation in enumerate(
        observations
    ):
        if observation.true_label not in observations_by_digit:
            raise ValueError(
                "true_label must be between 0 and 9."
            )

        observations_by_digit[
            observation.true_label
        ].append(
            (
                observation_index,
                observation,
            )
        )

    for digit, digit_observations in (
        observations_by_digit.items()
    ):
        if len(digit_observations) < samples_per_digit:
            raise ValueError(
                f"Digit {digit} contains only "
                f"{len(digit_observations)} observations; "
                f"{samples_per_digit} are required."
            )

    random_generator = random.Random(random_seed)

    selected: list[
        tuple[int, ConfidenceObservation]
    ] = []

    for digit in range(10):
        selected.extend(
            random_generator.sample(
                observations_by_digit[digit],
                samples_per_digit,
            )
        )

    return selected


def build_prediction_document(
    observation_index: int,
    observation: ConfidenceObservation,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    evaluation_protocol: str = DEFAULT_EVALUATION_PROTOCOL,
) -> dict[str, Any]:
    """
    Convert one real reliability observation into a MongoDB document.
    """

    if observation_index < 0:
        raise ValueError(
            "observation_index must not be negative."
        )

    if not model_version.strip():
        raise ValueError(
            "model_version must not be empty."
        )

    if not evaluation_protocol.strip():
        raise ValueError(
            "evaluation_protocol must not be empty."
        )

    return {
        "observation_index": observation_index,
        "model_version": model_version,
        "evaluation_protocol": evaluation_protocol,
        "true_label": int(observation.true_label),
        "predicted_label": int(
            observation.predicted_label
        ),
        "confidence": float(observation.confidence),
    }


def build_balanced_prediction_documents(
    observations: list[ConfidenceObservation],
    samples_per_digit: int = DEFAULT_SAMPLES_PER_DIGIT,
    random_seed: int = DEFAULT_RANDOM_SEED,
    *,
    model_version: str = DEFAULT_MODEL_VERSION,
    evaluation_protocol: str = DEFAULT_EVALUATION_PROTOCOL,
) -> list[dict[str, Any]]:
    """
    Build balanced MongoDB documents from real prediction observations.
    """

    selected = select_balanced_observations(
        observations=observations,
        samples_per_digit=samples_per_digit,
        random_seed=random_seed,
    )

    return [
        build_prediction_document(
            observation_index=observation_index,
            observation=observation,
            model_version=model_version,
            evaluation_protocol=evaluation_protocol,
        )
        for observation_index, observation
        in selected
    ]


def summarize_document_distribution(
    documents: list[dict[str, Any]],
) -> dict[int, int]:
    """
    Count documents by true digit.
    """

    return dict(
        sorted(
            Counter(
                document["true_label"]
                for document in documents
            ).items()
        )
    )