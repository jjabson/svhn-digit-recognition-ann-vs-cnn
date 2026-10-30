from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from threading import Lock


class ControlledActionMetricsError(ValueError):
    pass


class ControlledActionMetricEvent(str, Enum):
    REQUESTED = "requested"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    REJECTED = "rejected"
    DENIED = "denied"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass(frozen=True)
class ControlledActionMetricsSnapshot:
    requested: int
    approval_required: int
    approved: int
    rejected: int
    denied: int
    executed: int
    failed: int

    execution_latency_count: int
    total_execution_latency_ms: float
    average_execution_latency_ms: float
    max_execution_latency_ms: float

    def __post_init__(self) -> None:
        counters = (
            self.requested,
            self.approval_required,
            self.approved,
            self.rejected,
            self.denied,
            self.executed,
            self.failed,
            self.execution_latency_count,
        )

        if any(value < 0 for value in counters):
            raise ControlledActionMetricsError(
                "Controlled-action counters must not be negative."
            )

        latency_values = (
            self.total_execution_latency_ms,
            self.average_execution_latency_ms,
            self.max_execution_latency_ms,
        )

        if any(
            not isfinite(value) or value < 0
            for value in latency_values
        ):
            raise ControlledActionMetricsError(
                "Controlled-action latency values must be "
                "finite and non-negative."
            )

        execution_outcomes = (
            self.executed
            + self.failed
        )

        if self.execution_latency_count != execution_outcomes:
            raise ControlledActionMetricsError(
                "execution_latency_count must equal "
                "executed + failed."
            )


class ControlledActionMetrics:
    """
    Thread-safe in-process operational metrics for
    controlled-action workflows.

    These metrics are observational only.

    They must never be used as approval evidence,
    authorization state, replay protection, or an
    authoritative audit record.
    """

    def __init__(self) -> None:
        self._lock = Lock()

        self._requested = 0
        self._approval_required = 0
        self._approved = 0
        self._rejected = 0
        self._denied = 0
        self._executed = 0
        self._failed = 0

        self._execution_latency_count = 0
        self._total_execution_latency_ms = 0.0
        self._max_execution_latency_ms = 0.0

    def record(
        self,
        event: ControlledActionMetricEvent | str,
        *,
        duration_ms: float | None = None,
    ) -> None:
        try:
            normalized_event = ControlledActionMetricEvent(
                event
            )
        except ValueError as exc:
            raise ControlledActionMetricsError(
                f"Unsupported controlled-action metric event: {event}"
            ) from exc

        is_execution_outcome = normalized_event in {
            ControlledActionMetricEvent.EXECUTED,
            ControlledActionMetricEvent.FAILED,
        }

        if is_execution_outcome:
            if duration_ms is None:
                raise ControlledActionMetricsError(
                    "Execution outcomes require duration_ms."
                )

            if (
                not isfinite(duration_ms)
                or duration_ms < 0
            ):
                raise ControlledActionMetricsError(
                    "duration_ms must be finite and non-negative."
                )

        elif duration_ms is not None:
            raise ControlledActionMetricsError(
                "duration_ms is only valid for execution outcomes."
            )

        with self._lock:
            if normalized_event is ControlledActionMetricEvent.REQUESTED:
                self._requested += 1

            elif (
                normalized_event
                is ControlledActionMetricEvent.APPROVAL_REQUIRED
            ):
                self._approval_required += 1

            elif normalized_event is ControlledActionMetricEvent.APPROVED:
                self._approved += 1

            elif normalized_event is ControlledActionMetricEvent.REJECTED:
                self._rejected += 1

            elif normalized_event is ControlledActionMetricEvent.DENIED:
                self._denied += 1

            elif normalized_event is ControlledActionMetricEvent.EXECUTED:
                self._executed += 1

            elif normalized_event is ControlledActionMetricEvent.FAILED:
                self._failed += 1

            if is_execution_outcome:
                assert duration_ms is not None

                self._execution_latency_count += 1
                self._total_execution_latency_ms += duration_ms

                self._max_execution_latency_ms = max(
                    self._max_execution_latency_ms,
                    duration_ms,
                )

    def snapshot(
        self,
    ) -> ControlledActionMetricsSnapshot:
        with self._lock:
            if self._execution_latency_count == 0:
                average_execution_latency_ms = 0.0
            else:
                average_execution_latency_ms = (
                    self._total_execution_latency_ms
                    / self._execution_latency_count
                )

            return ControlledActionMetricsSnapshot(
                requested=self._requested,
                approval_required=self._approval_required,
                approved=self._approved,
                rejected=self._rejected,
                denied=self._denied,
                executed=self._executed,
                failed=self._failed,
                execution_latency_count=(
                    self._execution_latency_count
                ),
                total_execution_latency_ms=(
                    self._total_execution_latency_ms
                ),
                average_execution_latency_ms=(
                    average_execution_latency_ms
                ),
                max_execution_latency_ms=(
                    self._max_execution_latency_ms
                ),
            )