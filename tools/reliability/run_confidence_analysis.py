import numpy as np

from src.preprocessing import preprocess_image_batch

from tools.evaluation.evaluate_model import (
    load_historical_evaluation_data,
)
from tools.model.inspect_cnn import load_trained_model
from tools.reliability.confidence_analysis import (
    build_confidence_observations,
    evaluate_confidence_threshold,
    evaluate_confidence_thresholds,
    summarize_confidence,
    find_high_confidence_errors,
    summarize_error_pairs,
    build_calibration_bins,
    calculate_expected_calibration_error,
    calculate_maximum_calibration_error,
)

from tools.reliability.reliability_store import (
    save_reliability_observations,
)

def main() -> None:
    model = load_trained_model()

    X_eval, y_eval, protocol = (
        load_historical_evaluation_data()
    )

    X_eval_processed = preprocess_image_batch(
        X_eval
    )

    prediction_probabilities = model.predict(
        X_eval_processed,
        verbose=0,
    )

    predicted_labels = np.argmax(
        prediction_probabilities,
        axis=1,
    )

    confidences = np.max(
        prediction_probabilities,
        axis=1,
    )

    observations = build_confidence_observations(
        true_labels=y_eval.tolist(),
        predicted_labels=predicted_labels.tolist(),
        confidences=confidences.tolist(),
    )

    save_reliability_observations(
        protocol_name=protocol.protocol_name,
        observations=observations,
    )

    print()
    print("Reliability Persistence")
    print("-----------------------")
    print(
        f"Persisted Observations: "
        f"{len(observations):,}"
    )
    print(
        f"Protocol: "
        f"{protocol.protocol_name}"
    )

    summary = summarize_confidence(
        observations
    )

    threshold_evaluation = evaluate_confidence_threshold(
        observations,
        threshold=0.90,
    )

    thresholds = [
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.92,
        0.95,
        0.97,
        0.99,
    ]

    threshold_evaluations = evaluate_confidence_thresholds(
        observations,
        thresholds=thresholds,
    )

    print()
    print("Threshold Sweep")
    print("---------------")
    print(
        "Threshold  Coverage  Accepted Accuracy  "
        "Selective Risk  Uncertainty Rate  Errors Accepted"
    )

    for evaluation in threshold_evaluations:
        print(
            f"{evaluation.threshold:>8.2f}  "
            f"{evaluation.coverage * 100:>7.2f}%  "
            f"{evaluation.accepted_accuracy * 100:>16.2f}%  "
            f"{evaluation.selective_risk * 100:>13.2f}%  "
            f"{evaluation.uncertainty_rate * 100:>16.2f}%  "
            f"{evaluation.incorrect_accepted_predictions:>15}"
        )

    high_confidence_errors = find_high_confidence_errors(
        observations,
        minimum_confidence=0.99,
    )

    error_pairs = summarize_error_pairs(
        high_confidence_errors
    )

    sorted_error_pairs = sorted(
        error_pairs.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print()
    print("High-Confidence Error Pairs")
    print("---------------------------")
    print("True  Predicted  Count")

    for (
            true_label,
            predicted_label,
    ), count in sorted_error_pairs:
        print(
            f"{true_label:>4}  "
            f"{predicted_label:>9}  "
            f"{count:>5}"
        )

    calibration_bins = build_calibration_bins(
        observations,
        bin_count=10,
    )

    print()
    print("Calibration Analysis")
    print("--------------------")
    print(
        "Confidence Bin  Samples  Mean Confidence  "
        "Actual Accuracy  Calibration Gap"
    )

    for calibration_bin in calibration_bins:
        print(
            f"{calibration_bin.lower_bound:>4.1f}-"
            f"{calibration_bin.upper_bound:<4.1f}  "
            f"{calibration_bin.total_predictions:>7}  "
            f"{calibration_bin.mean_confidence * 100:>15.2f}%  "
            f"{calibration_bin.accuracy * 100:>14.2f}%  "
            f"{calibration_bin.calibration_gap * 100:>15.2f}%"
        )

    expected_calibration_error = (
        calculate_expected_calibration_error(
            calibration_bins
        )
    )

    print()
    print("Expected Calibration Error")
    print("--------------------------")
    print(
        f"ECE: "
        f"{expected_calibration_error:.4f}"
    )
    print(
        f"ECE Percent: "
        f"{expected_calibration_error * 100:.2f}%"
    )

    maximum_calibration_error = (
        calculate_maximum_calibration_error(
            calibration_bins
        )
    )

    print()
    print("Maximum Calibration Error")
    print("-------------------------")
    print(
        f"MCE: "
        f"{maximum_calibration_error:.4f}"
    )
    print(
        f"MCE Percent: "
        f"{maximum_calibration_error * 100:.2f}%"
    )

    print()
    print("High-Confidence Errors")
    print("----------------------")
    print(
        f"Errors with confidence >= 0.99: "
        f"{len(high_confidence_errors)}"
    )

    print("SVHN Confidence Reliability Analysis")
    print("------------------------------------")
    print(
        f"Protocol: {protocol.protocol_name}"
    )
    print(
        f"Total Predictions: "
        f"{summary.total_predictions}"
    )
    print(
        f"Accuracy: "
        f"{summary.accuracy:.4f}"
    )
    print(
        f"Accuracy Percent: "
        f"{summary.accuracy * 100:.2f}%"
    )
    print(
        f"Mean Confidence: "
        f"{summary.mean_confidence:.4f}"
    )
    print(
        f"Mean Correct Confidence: "
        f"{summary.mean_correct_confidence:.4f}"
    )
    print(
        f"Mean Incorrect Confidence: "
        f"{summary.mean_incorrect_confidence:.4f}"
    )
    print()
    print("Threshold Reliability Analysis")
    print("------------------------------")
    print(
        f"Threshold: "
        f"{threshold_evaluation.threshold:.2f}"
    )
    print(
        f"Accepted Predictions: "
        f"{threshold_evaluation.accepted_predictions}"
    )
    print(
        f"Uncertain Predictions: "
        f"{threshold_evaluation.uncertain_predictions}"
    )
    print(
        f"Coverage: "
        f"{threshold_evaluation.coverage * 100:.2f}%"
    )
    print(
        f"Accepted Accuracy: "
        f"{threshold_evaluation.accepted_accuracy * 100:.2f}%"
    )
    print(
        f"Selective Risk: "
        f"{threshold_evaluation.selective_risk * 100:.2f}%"
    )
    print(
        f"Uncertainty Rate: "
        f"{threshold_evaluation.uncertainty_rate * 100:.2f}%"
    )
    print(
        f"Incorrect Accepted Predictions: "
        f"{threshold_evaluation.incorrect_accepted_predictions}"
    )


if __name__ == "__main__":
    main()