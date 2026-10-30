import pytest

from tools.mongodb.prediction_documents import (
    build_balanced_prediction_documents,
    build_prediction_document,
    select_balanced_observations,
    summarize_document_distribution,
)
from tools.reliability.confidence_analysis import (
    ConfidenceObservation,
)


def build_test_observations(
    observations_per_digit: int = 60,
) -> list[ConfidenceObservation]:
    observations = []

    for digit in range(10):
        for sample_index in range(
            observations_per_digit
        ):
            observations.append(
                ConfidenceObservation(
                    true_label=digit,
                    predicted_label=(
                        digit
                        if sample_index % 10 != 0
                        else (digit + 1) % 10
                    ),
                    confidence=(
                        0.95
                        if sample_index % 10 != 0
                        else 0.70
                    ),
                )
            )

    return observations


def test_build_prediction_document():
    observation = ConfidenceObservation(
        true_label=7,
        predicted_label=7,
        confidence=0.9762,
    )

    document = build_prediction_document(
        observation_index=123,
        observation=observation,
    )

    assert document == {
        "observation_index": 123,
        "model_version": "cnn_v2",
        "evaluation_protocol": (
            "Historical Stratified Holdout"
        ),
        "true_label": 7,
        "predicted_label": 7,
        "confidence": 0.9762,
    }


def test_balanced_selection_contains_50_per_digit():
    observations = build_test_observations()

    selected = select_balanced_observations(
        observations,
        samples_per_digit=50,
    )

    assert len(selected) == 500

    counts = {}

    for _, observation in selected:
        counts[observation.true_label] = (
            counts.get(
                observation.true_label,
                0,
            )
            + 1
        )

    assert counts == {
        digit: 50
        for digit in range(10)
    }


def test_balanced_selection_is_reproducible():
    observations = build_test_observations()

    first = select_balanced_observations(
        observations,
        random_seed=42,
    )

    second = select_balanced_observations(
        observations,
        random_seed=42,
    )

    first_indexes = [
        observation_index
        for observation_index, _
        in first
    ]

    second_indexes = [
        observation_index
        for observation_index, _
        in second
    ]

    assert first_indexes == second_indexes


def test_balanced_documents_have_expected_distribution():
    observations = build_test_observations()

    documents = build_balanced_prediction_documents(
        observations
    )

    distribution = summarize_document_distribution(
        documents
    )

    assert distribution == {
        digit: 50
        for digit in range(10)
    }


def test_documents_contain_only_expected_source_fields():
    observations = build_test_observations()

    documents = build_balanced_prediction_documents(
        observations
    )

    assert set(documents[0]) == {
        "observation_index",
        "model_version",
        "evaluation_protocol",
        "true_label",
        "predicted_label",
        "confidence",
    }


def test_selection_rejects_insufficient_digit_data():
    observations = build_test_observations(
        observations_per_digit=10
    )

    with pytest.raises(
        ValueError,
        match="contains only",
    ):
        select_balanced_observations(
            observations,
            samples_per_digit=50,
        )


def test_selection_rejects_invalid_sample_count():
    observations = build_test_observations()

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        select_balanced_observations(
            observations,
            samples_per_digit=0,
        )


def test_document_rejects_negative_observation_index():
    observation = ConfidenceObservation(
        true_label=7,
        predicted_label=7,
        confidence=0.95,
    )

    with pytest.raises(
        ValueError,
        match="observation_index",
    ):
        build_prediction_document(
            observation_index=-1,
            observation=observation,
        )