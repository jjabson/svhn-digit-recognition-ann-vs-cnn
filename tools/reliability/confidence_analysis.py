from dataclasses import dataclass
from collections import Counter

@dataclass(frozen=True)
class ConfidenceObservation:
    true_label: int
    predicted_label: int
    confidence: float

    @property
    def correct(self) -> bool:
        return self.true_label == self.predicted_label


@dataclass(frozen=True)
class ConfidenceSummary:
    total_predictions: int
    correct_predictions: int
    incorrect_predictions: int
    mean_confidence: float
    mean_correct_confidence: float
    mean_incorrect_confidence: float

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0

        return self.correct_predictions / self.total_predictions


def summarize_confidence(
    observations: list[ConfidenceObservation],
) -> ConfidenceSummary:
    total_predictions = len(observations)

    if total_predictions == 0:
        return ConfidenceSummary(
            total_predictions=0,
            correct_predictions=0,
            incorrect_predictions=0,
            mean_confidence=0.0,
            mean_correct_confidence=0.0,
            mean_incorrect_confidence=0.0,
        )

    correct_observations = [
        observation
        for observation in observations
        if observation.correct
    ]

    incorrect_observations = [
        observation
        for observation in observations
        if not observation.correct
    ]

    correct_predictions = len(correct_observations)
    incorrect_predictions = len(incorrect_observations)

    mean_confidence = sum(
        observation.confidence
        for observation in observations
    ) / total_predictions

    mean_correct_confidence = (
        sum(
            observation.confidence
            for observation in correct_observations
        )
        / correct_predictions
        if correct_predictions > 0
        else 0.0
    )

    mean_incorrect_confidence = (
        sum(
            observation.confidence
            for observation in incorrect_observations
        )
        / incorrect_predictions
        if incorrect_predictions > 0
        else 0.0
    )

    return ConfidenceSummary(
        total_predictions=total_predictions,
        correct_predictions=correct_predictions,
        incorrect_predictions=incorrect_predictions,
        mean_confidence=mean_confidence,
        mean_correct_confidence=mean_correct_confidence,
        mean_incorrect_confidence=mean_incorrect_confidence,
    )

def build_confidence_observations(
    true_labels: list[int],
    predicted_labels: list[int],
    confidences: list[float],
) -> list[ConfidenceObservation]:
    if not (
        len(true_labels)
        == len(predicted_labels)
        == len(confidences)
    ):
        raise ValueError(
            "true_labels, predicted_labels, and confidences "
            "must have the same length."
        )

    return [
        ConfidenceObservation(
            true_label=true_label,
            predicted_label=predicted_label,
            confidence=confidence,
        )
        for true_label, predicted_label, confidence in zip(
            true_labels,
            predicted_labels,
            confidences,
        )
    ]

@dataclass(frozen=True)
class ThresholdEvaluation:
    threshold: float
    total_predictions: int
    accepted_predictions: int
    uncertain_predictions: int
    correct_accepted_predictions: int
    incorrect_accepted_predictions: int

    @property
    def coverage(self) -> float:
        if self.total_predictions == 0:
            return 0.0

        return (
            self.accepted_predictions
            / self.total_predictions
        )

    @property
    def accepted_accuracy(self) -> float:
        if self.accepted_predictions == 0:
            return 0.0

        return (
            self.correct_accepted_predictions
            / self.accepted_predictions
        )

    @property
    def selective_risk(self) -> float:
        if self.accepted_predictions == 0:
            return 0.0

        return (
            self.incorrect_accepted_predictions
            / self.accepted_predictions
        )

    @property
    def uncertainty_rate(self) -> float:
        if self.total_predictions == 0:
            return 0.0

        return (
            self.uncertain_predictions
            / self.total_predictions
        )

def evaluate_confidence_threshold(
    observations: list[ConfidenceObservation],
    threshold: float,
) -> ThresholdEvaluation:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0.0 and 1.0."
        )

    accepted = [
        observation
        for observation in observations
        if observation.confidence >= threshold
    ]

    correct_accepted = [
        observation
        for observation in accepted
        if observation.correct
    ]

    incorrect_accepted = [
        observation
        for observation in accepted
        if not observation.correct
    ]

    return ThresholdEvaluation(
        threshold=threshold,
        total_predictions=len(observations),
        accepted_predictions=len(accepted),
        uncertain_predictions=(
            len(observations) - len(accepted)
        ),
        correct_accepted_predictions=len(
            correct_accepted
        ),
        incorrect_accepted_predictions=len(
            incorrect_accepted
        ),
    )

def evaluate_confidence_thresholds(
    observations: list[ConfidenceObservation],
    thresholds: list[float],
) -> list[ThresholdEvaluation]:
    return [
        evaluate_confidence_threshold(
            observations=observations,
            threshold=threshold,
        )
        for threshold in thresholds
    ]

def find_high_confidence_errors(
    observations: list[ConfidenceObservation],
    minimum_confidence: float,
) -> list[ConfidenceObservation]:
    if not 0.0 <= minimum_confidence <= 1.0:
        raise ValueError(
            "minimum_confidence must be between 0.0 and 1.0."
        )

    return [
        observation
        for observation in observations
        if (
            not observation.correct
            and observation.confidence >= minimum_confidence
        )
    ]

def summarize_error_pairs(
    observations: list[ConfidenceObservation],
) -> dict[tuple[int, int], int]:
    return dict(
        Counter(
            (
                observation.true_label,
                observation.predicted_label,
            )
            for observation in observations
            if not observation.correct
        )
    )

@dataclass(frozen=True)
class CalibrationBin:
    lower_bound: float
    upper_bound: float
    total_predictions: int
    correct_predictions: int
    mean_confidence: float

    @property
    def accuracy(self) -> float:
        if self.total_predictions == 0:
            return 0.0

        return (
            self.correct_predictions
            / self.total_predictions
        )

    @property
    def calibration_gap(self) -> float:
        return abs(
            self.accuracy - self.mean_confidence
        )

def build_calibration_bins(
    observations: list[ConfidenceObservation],
    bin_count: int = 10,
) -> list[CalibrationBin]:
    if bin_count <= 0:
        raise ValueError(
            "bin_count must be greater than zero."
        )

    bins: list[CalibrationBin] = []

    for bin_index in range(bin_count):
        lower_bound = bin_index / bin_count
        upper_bound = (bin_index + 1) / bin_count

        if bin_index == bin_count - 1:
            bin_observations = [
                observation
                for observation in observations
                if (
                    lower_bound
                    <= observation.confidence
                    <= upper_bound
                )
            ]
        else:
            bin_observations = [
                observation
                for observation in observations
                if (
                    lower_bound
                    <= observation.confidence
                    < upper_bound
                )
            ]

        total_predictions = len(
            bin_observations
        )

        correct_predictions = sum(
            observation.correct
            for observation in bin_observations
        )

        mean_confidence = (
            sum(
                observation.confidence
                for observation in bin_observations
            )
            / total_predictions
            if total_predictions > 0
            else 0.0
        )

        bins.append(
            CalibrationBin(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                total_predictions=total_predictions,
                correct_predictions=correct_predictions,
                mean_confidence=mean_confidence,
            )
        )

    return bins

def calculate_expected_calibration_error(
    calibration_bins: list[CalibrationBin],
) -> float:
    total_predictions = sum(
        calibration_bin.total_predictions
        for calibration_bin in calibration_bins
    )

    if total_predictions == 0:
        return 0.0

    return sum(
        (
            calibration_bin.total_predictions
            / total_predictions
        )
        * calibration_bin.calibration_gap
        for calibration_bin in calibration_bins
    )

def calculate_maximum_calibration_error(
    calibration_bins: list[CalibrationBin],
) -> float:
    populated_bins = [
        calibration_bin
        for calibration_bin in calibration_bins
        if calibration_bin.total_predictions > 0
    ]

    if not populated_bins:
        return 0.0

    return max(
        calibration_bin.calibration_gap
        for calibration_bin in populated_bins
    )