from __future__ import annotations

from dataclasses import dataclass
from threading import Lock


class InferenceMetricsError(ValueError):
    pass


@dataclass(frozen=True)
class InferenceMetricsSnapshot:
    total_attempts: int
    accepted: int
    uncertain: int
    failed: int

    total_latency_ms: float
    average_latency_ms: float
    max_latency_ms: float

    def __post_init__(self) -> None:
        integer_values = (
            self.total_attempts,
            self.accepted,
            self.uncertain,
            self.failed,
        )

        if any(value < 0 for value in integer_values):
            raise InferenceMetricsError(
                "Inference counters must not be negative."
            )

        if self.total_latency_ms < 0:
            raise InferenceMetricsError(
                "total_latency_ms must not be negative."
            )

        if self.average_latency_ms < 0:
            raise InferenceMetricsError(
                "average_latency_ms must not be negative."
            )

        if self.max_latency_ms < 0:
            raise InferenceMetricsError(
                "max_latency_ms must not be negative."
            )

        outcome_total = (
            self.accepted
            + self.uncertain
            + self.failed
        )

        if outcome_total != self.total_attempts:
            raise InferenceMetricsError(
                "Outcome counters must equal "
                "total_attempts."
            )


class InferenceMetrics:
    """
    Thread-safe in-process inference metrics.

    Metrics are observational only. They must not influence
    inference policy or authorization decisions.
    """

    def __init__(self) -> None:
        self._lock = Lock()

        self._total_attempts = 0

        self._accepted = 0
        self._uncertain = 0
        self._failed = 0

        self._total_latency_ms = 0.0
        self._max_latency_ms = 0.0

    def record(
        self,
        *,
        status: str,
        duration_ms: float,
    ) -> None:
        if duration_ms < 0:
            raise InferenceMetricsError(
                "duration_ms must not be negative."
            )

        normalized_status = status.upper()

        if normalized_status not in {
            "ACCEPTED",
            "UNCERTAIN",
            "FAILED",
        }:
            raise InferenceMetricsError(
                "Unsupported inference status: "
                f"{status}"
            )

        with self._lock:
            self._total_attempts += 1

            if normalized_status == "ACCEPTED":
                self._accepted += 1

            elif normalized_status == "UNCERTAIN":
                self._uncertain += 1

            else:
                self._failed += 1

            self._total_latency_ms += duration_ms

            self._max_latency_ms = max(
                self._max_latency_ms,
                duration_ms,
            )

    def snapshot(
        self,
    ) -> InferenceMetricsSnapshot:
        with self._lock:
            total_attempts = self._total_attempts

            if total_attempts == 0:
                average_latency_ms = 0.0
            else:
                average_latency_ms = (
                    self._total_latency_ms
                    / total_attempts
                )

            return InferenceMetricsSnapshot(
                total_attempts=total_attempts,
                accepted=self._accepted,
                uncertain=self._uncertain,
                failed=self._failed,
                total_latency_ms=(
                    self._total_latency_ms
                ),
                average_latency_ms=(
                    average_latency_ms
                ),
                max_latency_ms=(
                    self._max_latency_ms
                ),
            )