import pytest

from tools.reliability.confidence_analysis import (
    ConfidenceObservation,
    build_confidence_observations,
    evaluate_confidence_threshold,
    evaluate_confidence_thresholds,
    summarize_confidence,
    find_high_confidence_errors,
    summarize_error_pairs,
    CalibrationBin,
    build_calibration_bins,
    calculate_expected_calibration_error,
    calculate_maximum_calibration_error,
)


def test_confidence_observation_identifies_correct_prediction():
    observation = ConfidenceObservation(
        true_label=7,
        predicted_label=7,
        confidence=0.97,
    )

    assert observation.correct is True


def test_confidence_observation_identifies_incorrect_prediction():
    observation = ConfidenceObservation(
        true_label=7,
        predicted_label=1,
        confidence=0.96,
    )

    assert observation.correct is False

def test_summarize_confidence_calculates_prediction_statistics():
    observations = [
        ConfidenceObservation(
            true_label=7,
            predicted_label=7,
            confidence=0.98,
        ),
        ConfidenceObservation(
            true_label=3,
            predicted_label=3,
            confidence=0.92,
        ),
        ConfidenceObservation(
            true_label=5,
            predicted_label=8,
            confidence=0.80,
        ),
    ]

    summary = summarize_confidence(observations)

    assert summary.total_predictions == 3
    assert summary.correct_predictions == 2
    assert summary.incorrect_predictions == 1
    assert summary.mean_confidence == pytest.approx(0.90)
    assert summary.mean_correct_confidence == pytest.approx(0.95)
    assert summary.mean_incorrect_confidence == pytest.approx(0.80)
    assert summary.accuracy == pytest.approx(2 / 3)


def test_summarize_confidence_handles_empty_observations():
    summary = summarize_confidence([])

    assert summary.total_predictions == 0
    assert summary.correct_predictions == 0
    assert summary.incorrect_predictions == 0
    assert summary.mean_confidence == 0.0
    assert summary.accuracy == 0.0
    assert summary.mean_correct_confidence == 0.0
    assert summary.mean_incorrect_confidence == 0.0

def test_summarize_confidence_handles_all_correct_predictions():
    observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.90,
        ),
        ConfidenceObservation(
            true_label=2,
            predicted_label=2,
            confidence=0.80,
        ),
    ]

    summary = summarize_confidence(observations)

    assert summary.mean_correct_confidence == pytest.approx(0.85)
    assert summary.mean_incorrect_confidence == 0.0


def test_summarize_confidence_handles_all_incorrect_predictions():
    observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=7,
            confidence=0.70,
        ),
        ConfidenceObservation(
            true_label=2,
            predicted_label=9,
            confidence=0.60,
        ),
    ]

    summary = summarize_confidence(observations)

    assert summary.mean_correct_confidence == 0.0
    assert summary.mean_incorrect_confidence == pytest.approx(0.65)

def test_build_confidence_observations_creates_observations():
    observations = build_confidence_observations(
        true_labels=[1, 2],
        predicted_labels=[1, 7],
        confidences=[0.95, 0.80],
    )

    assert len(observations) == 2

    assert observations[0].true_label == 1
    assert observations[0].predicted_label == 1
    assert observations[0].confidence == pytest.approx(0.95)
    assert observations[0].correct is True

    assert observations[1].true_label == 2
    assert observations[1].predicted_label == 7
    assert observations[1].confidence == pytest.approx(0.80)
    assert observations[1].correct is False


def test_build_confidence_observations_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        build_confidence_observations(
            true_labels=[1, 2],
            predicted_labels=[1],
            confidences=[0.95, 0.80],
        )

def test_evaluate_confidence_threshold_calculates_reliability_metrics():
    observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.98,
        ),
        ConfidenceObservation(
            true_label=2,
            predicted_label=2,
            confidence=0.95,
        ),
        ConfidenceObservation(
            true_label=3,
            predicted_label=8,
            confidence=0.92,
        ),
        ConfidenceObservation(
            true_label=4,
            predicted_label=7,
            confidence=0.70,
        ),
    ]

    evaluation = evaluate_confidence_threshold(
        observations,
        threshold=0.90,
    )

    assert evaluation.total_predictions == 4
    assert evaluation.accepted_predictions == 3
    assert evaluation.uncertain_predictions == 1

    assert evaluation.correct_accepted_predictions == 2
    assert evaluation.incorrect_accepted_predictions == 1

    assert evaluation.coverage == pytest.approx(0.75)
    assert evaluation.accepted_accuracy == pytest.approx(2 / 3)
    assert evaluation.selective_risk == pytest.approx(1 / 3)
    assert evaluation.uncertainty_rate == pytest.approx(0.25)

def test_threshold_accepts_confidence_equal_to_threshold():
    observations = [
        ConfidenceObservation(
            true_label=7,
            predicted_label=7,
            confidence=0.90,
        ),
    ]

    evaluation = evaluate_confidence_threshold(
        observations,
        threshold=0.90,
    )

    assert evaluation.accepted_predictions == 1
    assert evaluation.uncertain_predictions == 0

def test_evaluate_confidence_threshold_rejects_invalid_threshold():
    with pytest.raises(ValueError):
        evaluate_confidence_threshold(
            [],
            threshold=1.01,
        )

def test_evaluate_confidence_thresholds_handles_empty_thresholds():
    observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.98,
        ),
    ]

    evaluations = evaluate_confidence_thresholds(
        observations,
        thresholds=[],
    )

    assert evaluations == []

def test_evaluate_confidence_thresholds_evaluates_each_threshold():
    observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.98,
        ),
        ConfidenceObservation(
            true_label=2,
            predicted_label=2,
            confidence=0.92,
        ),
        ConfidenceObservation(
            true_label=3,
            predicted_label=8,
            confidence=0.80,
        ),
    ]

    evaluations = evaluate_confidence_thresholds(
        observations,
        thresholds=[0.80, 0.90, 0.95],
    )

    assert len(evaluations) == 3

    assert evaluations[0].threshold == pytest.approx(0.80)
    assert evaluations[0].accepted_predictions == 3

    assert evaluations[1].threshold == pytest.approx(0.90)
    assert evaluations[1].accepted_predictions == 2

    assert evaluations[2].threshold == pytest.approx(0.95)
    assert evaluations[2].accepted_predictions == 1

def test_find_high_confidence_errors_returns_matching_errors():
    observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.99,
        ),
        ConfidenceObservation(
            true_label=3,
            predicted_label=5,
            confidence=0.995,
        ),
        ConfidenceObservation(
            true_label=8,
            predicted_label=6,
            confidence=0.97,
        ),
    ]

    errors = find_high_confidence_errors(
        observations,
        minimum_confidence=0.99,
    )

    assert len(errors) == 1
    assert errors[0].true_label == 3
    assert errors[0].predicted_label == 5
    assert errors[0].confidence == pytest.approx(0.995)

def test_find_high_confidence_errors_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        find_high_confidence_errors(
            [],
            minimum_confidence=1.01,
        )

def test_summarize_error_pairs_counts_misclassifications():
    observations = [
        ConfidenceObservation(
            true_label=3,
            predicted_label=5,
            confidence=0.99,
        ),
        ConfidenceObservation(
            true_label=3,
            predicted_label=5,
            confidence=0.98,
        ),
        ConfidenceObservation(
            true_label=8,
            predicted_label=6,
            confidence=0.97,
        ),
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.99,
        ),
    ]

    summary = summarize_error_pairs(
        observations
    )

    assert summary[(3, 5)] == 2
    assert summary[(8, 6)] == 1
    assert (1, 1) not in summary

def test_summarize_error_pairs_handles_no_errors():
    observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.99,
        ),
    ]

    summary = summarize_error_pairs(
        observations
    )

    assert summary == {}

def test_calibration_bin_calculates_accuracy():
    calibration_bin = CalibrationBin(
        lower_bound=0.80,
        upper_bound=0.90,
        total_predictions=10,
        correct_predictions=8,
        mean_confidence=0.85,
    )

    assert calibration_bin.accuracy == pytest.approx(0.80)

def test_calibration_bin_calculates_calibration_gap():
    calibration_bin = CalibrationBin(
        lower_bound=0.80,
        upper_bound=0.90,
        total_predictions=10,
        correct_predictions=8,
        mean_confidence=0.85,
    )

    assert calibration_bin.calibration_gap == pytest.approx(0.05)

def test_calibration_bin_handles_empty_bin():
    calibration_bin = CalibrationBin(
        lower_bound=0.80,
        upper_bound=0.90,
        total_predictions=0,
        correct_predictions=0,
        mean_confidence=0.0,
    )

    assert calibration_bin.accuracy == 0.0
    assert calibration_bin.calibration_gap == 0.0

def test_build_calibration_bins_assigns_observations():
    observations = [
        ConfidenceObservation(
            true_label=1,
            predicted_label=1,
            confidence=0.85,
        ),
        ConfidenceObservation(
            true_label=2,
            predicted_label=2,
            confidence=0.88,
        ),
        ConfidenceObservation(
            true_label=3,
            predicted_label=5,
            confidence=0.95,
        ),
    ]

    bins = build_calibration_bins(
        observations,
        bin_count=10,
    )

    assert len(bins) == 10

    bin_80_90 = bins[8]
    assert bin_80_90.total_predictions == 2
    assert bin_80_90.correct_predictions == 2
    assert bin_80_90.mean_confidence == pytest.approx(0.865)
    assert bin_80_90.accuracy == pytest.approx(1.0)

    bin_90_100 = bins[9]
    assert bin_90_100.total_predictions == 1
    assert bin_90_100.correct_predictions == 0
    assert bin_90_100.mean_confidence == pytest.approx(0.95)
    assert bin_90_100.accuracy == pytest.approx(0.0)

def test_build_calibration_bins_includes_confidence_one():
    observations = [
        ConfidenceObservation(
            true_label=7,
            predicted_label=7,
            confidence=1.0,
        ),
    ]

    bins = build_calibration_bins(
        observations,
        bin_count=10,
    )

    assert bins[9].total_predictions == 1
    assert bins[9].correct_predictions == 1

def test_build_calibration_bins_rejects_invalid_bin_count():
    with pytest.raises(ValueError):
        build_calibration_bins(
            [],
            bin_count=0,
        )

def test_calculate_expected_calibration_error():
    calibration_bins = [
        CalibrationBin(
            lower_bound=0.0,
            upper_bound=0.5,
            total_predictions=2,
            correct_predictions=1,
            mean_confidence=0.40,
        ),
        CalibrationBin(
            lower_bound=0.5,
            upper_bound=1.0,
            total_predictions=8,
            correct_predictions=6,
            mean_confidence=0.80,
        ),
    ]

    ece = calculate_expected_calibration_error(
        calibration_bins
    )

    # First bin:
    # weight = 2/10
    # gap = |0.50 - 0.40| = 0.10
    #
    # Second bin:
    # weight = 8/10
    # gap = |0.75 - 0.80| = 0.05
    #
    # ECE = (0.2 * 0.10) + (0.8 * 0.05)
    #     = 0.06

    assert ece == pytest.approx(0.06)

def test_calculate_expected_calibration_error_handles_empty_bins():
    calibration_bins = []

    ece = calculate_expected_calibration_error(
        calibration_bins
    )

    assert ece == 0.0

def test_calculate_maximum_calibration_error():
    calibration_bins = [
        CalibrationBin(
            lower_bound=0.0,
            upper_bound=0.5,
            total_predictions=10,
            correct_predictions=5,
            mean_confidence=0.40,
        ),
        CalibrationBin(
            lower_bound=0.5,
            upper_bound=1.0,
            total_predictions=10,
            correct_predictions=9,
            mean_confidence=0.80,
        ),
    ]

    mce = calculate_maximum_calibration_error(
        calibration_bins
    )

    assert mce == pytest.approx(0.10)

def test_calculate_maximum_calibration_error_handles_empty_bins():
    calibration_bins = []

    mce = calculate_maximum_calibration_error(
        calibration_bins
    )

    assert mce == 0.0
