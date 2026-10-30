from concurrent.futures import ThreadPoolExecutor

import pytest

from src.observability.controlled_action_metrics import (
    ControlledActionMetricEvent,
    ControlledActionMetrics,
    ControlledActionMetricsError,
    ControlledActionMetricsSnapshot,
)


def test_new_snapshot_is_zero():
    metrics = ControlledActionMetrics()

    snapshot = metrics.snapshot()

    assert snapshot.requested == 0
    assert snapshot.approval_required == 0
    assert snapshot.approved == 0
    assert snapshot.rejected == 0
    assert snapshot.denied == 0
    assert snapshot.executed == 0
    assert snapshot.failed == 0

    assert snapshot.execution_latency_count == 0
    assert snapshot.total_execution_latency_ms == 0.0
    assert snapshot.average_execution_latency_ms == 0.0
    assert snapshot.max_execution_latency_ms == 0.0


def test_records_controlled_action_lifecycle_events():
    metrics = ControlledActionMetrics()

    metrics.record(
        ControlledActionMetricEvent.REQUESTED
    )
    metrics.record(
        ControlledActionMetricEvent.APPROVAL_REQUIRED
    )
    metrics.record(
        ControlledActionMetricEvent.APPROVED
    )
    metrics.record(
        ControlledActionMetricEvent.EXECUTED,
        duration_ms=10.0,
    )

    snapshot = metrics.snapshot()

    assert snapshot.requested == 1
    assert snapshot.approval_required == 1
    assert snapshot.approved == 1
    assert snapshot.executed == 1

    assert snapshot.rejected == 0
    assert snapshot.denied == 0
    assert snapshot.failed == 0


def test_records_rejected_decision():
    metrics = ControlledActionMetrics()

    metrics.record(
        ControlledActionMetricEvent.REJECTED
    )

    assert metrics.snapshot().rejected == 1


def test_records_denied_execution():
    metrics = ControlledActionMetrics()

    metrics.record(
        ControlledActionMetricEvent.DENIED
    )

    assert metrics.snapshot().denied == 1


def test_records_failed_execution():
    metrics = ControlledActionMetrics()

    metrics.record(
        ControlledActionMetricEvent.FAILED,
        duration_ms=25.0,
    )

    snapshot = metrics.snapshot()

    assert snapshot.failed == 1
    assert snapshot.execution_latency_count == 1
    assert snapshot.total_execution_latency_ms == 25.0
    assert snapshot.average_execution_latency_ms == 25.0
    assert snapshot.max_execution_latency_ms == 25.0


def test_execution_latency_statistics():
    metrics = ControlledActionMetrics()

    metrics.record(
        ControlledActionMetricEvent.EXECUTED,
        duration_ms=10.0,
    )

    metrics.record(
        ControlledActionMetricEvent.FAILED,
        duration_ms=30.0,
    )

    snapshot = metrics.snapshot()

    assert snapshot.execution_latency_count == 2
    assert snapshot.total_execution_latency_ms == 40.0
    assert snapshot.average_execution_latency_ms == 20.0
    assert snapshot.max_execution_latency_ms == 30.0


def test_execution_outcome_requires_duration():
    metrics = ControlledActionMetrics()

    with pytest.raises(
        ControlledActionMetricsError,
        match="require duration_ms",
    ):
        metrics.record(
            ControlledActionMetricEvent.EXECUTED
        )


def test_non_execution_event_rejects_duration():
    metrics = ControlledActionMetrics()

    with pytest.raises(
        ControlledActionMetricsError,
        match="only valid for execution outcomes",
    ):
        metrics.record(
            ControlledActionMetricEvent.APPROVED,
            duration_ms=1.0,
        )


@pytest.mark.parametrize(
    "duration_ms",
    [
        -0.01,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_rejects_invalid_execution_duration(
    duration_ms,
):
    metrics = ControlledActionMetrics()

    with pytest.raises(
        ControlledActionMetricsError,
        match="finite and non-negative",
    ):
        metrics.record(
            ControlledActionMetricEvent.FAILED,
            duration_ms=duration_ms,
        )


def test_rejects_unknown_event():
    metrics = ControlledActionMetrics()

    with pytest.raises(
        ControlledActionMetricsError,
        match="Unsupported controlled-action",
    ):
        metrics.record(
            "unknown-event"
        )


def test_snapshot_is_independent_of_future_updates():
    metrics = ControlledActionMetrics()

    metrics.record(
        ControlledActionMetricEvent.REQUESTED
    )

    first_snapshot = metrics.snapshot()

    metrics.record(
        ControlledActionMetricEvent.APPROVAL_REQUIRED
    )

    assert first_snapshot.requested == 1
    assert first_snapshot.approval_required == 0

    second_snapshot = metrics.snapshot()

    assert second_snapshot.requested == 1
    assert second_snapshot.approval_required == 1


def test_snapshot_rejects_execution_latency_mismatch():
    with pytest.raises(
        ControlledActionMetricsError,
        match="execution_latency_count",
    ):
        ControlledActionMetricsSnapshot(
            requested=1,
            approval_required=1,
            approved=1,
            rejected=0,
            denied=0,
            executed=1,
            failed=0,
            execution_latency_count=0,
            total_execution_latency_ms=0.0,
            average_execution_latency_ms=0.0,
            max_execution_latency_ms=0.0,
        )


def test_concurrent_recording_is_thread_safe():
    metrics = ControlledActionMetrics()

    def record_request(
        _: int,
    ) -> None:
        metrics.record(
            ControlledActionMetricEvent.REQUESTED
        )

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:
        list(
            executor.map(
                record_request,
                range(1000),
            )
        )

    snapshot = metrics.snapshot()

    assert snapshot.requested == 1000