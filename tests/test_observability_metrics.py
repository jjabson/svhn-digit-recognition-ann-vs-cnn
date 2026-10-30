from concurrent.futures import ThreadPoolExecutor

import pytest

from src.observability.metrics import (
    InferenceMetrics,
    InferenceMetricsError,
    InferenceMetricsSnapshot,
)


def test_new_metrics_snapshot_is_zero():
    metrics = InferenceMetrics()

    snapshot = metrics.snapshot()

    assert snapshot.total_attempts == 0
    assert snapshot.accepted == 0
    assert snapshot.uncertain == 0
    assert snapshot.failed == 0

    assert snapshot.total_latency_ms == 0.0
    assert snapshot.average_latency_ms == 0.0
    assert snapshot.max_latency_ms == 0.0


def test_record_accepted_inference():
    metrics = InferenceMetrics()

    metrics.record(
        status="ACCEPTED",
        duration_ms=10.0,
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_attempts == 1
    assert snapshot.accepted == 1
    assert snapshot.uncertain == 0
    assert snapshot.failed == 0

    assert snapshot.total_latency_ms == 10.0
    assert snapshot.average_latency_ms == 10.0
    assert snapshot.max_latency_ms == 10.0


def test_record_uncertain_inference():
    metrics = InferenceMetrics()

    metrics.record(
        status="UNCERTAIN",
        duration_ms=20.0,
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_attempts == 1
    assert snapshot.accepted == 0
    assert snapshot.uncertain == 1
    assert snapshot.failed == 0


def test_record_failed_inference():
    metrics = InferenceMetrics()

    metrics.record(
        status="FAILED",
        duration_ms=30.0,
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_attempts == 1
    assert snapshot.accepted == 0
    assert snapshot.uncertain == 0
    assert snapshot.failed == 1


def test_record_accepts_lowercase_status():
    metrics = InferenceMetrics()

    metrics.record(
        status="accepted",
        duration_ms=5.0,
    )

    assert metrics.snapshot().accepted == 1


def test_record_rejects_unknown_status():
    metrics = InferenceMetrics()

    with pytest.raises(
        InferenceMetricsError,
        match="Unsupported inference status",
    ):
        metrics.record(
            status="UNKNOWN",
            duration_ms=10.0,
        )


def test_record_rejects_negative_duration():
    metrics = InferenceMetrics()

    with pytest.raises(
        InferenceMetricsError,
        match="duration_ms must not be negative",
    ):
        metrics.record(
            status="ACCEPTED",
            duration_ms=-0.01,
        )


def test_metrics_calculate_latency_statistics():
    metrics = InferenceMetrics()

    metrics.record(
        status="ACCEPTED",
        duration_ms=10.0,
    )

    metrics.record(
        status="UNCERTAIN",
        duration_ms=20.0,
    )

    metrics.record(
        status="FAILED",
        duration_ms=30.0,
    )

    snapshot = metrics.snapshot()

    assert snapshot.total_attempts == 3

    assert snapshot.total_latency_ms == 60.0
    assert snapshot.average_latency_ms == 20.0
    assert snapshot.max_latency_ms == 30.0


def test_snapshot_is_independent_of_future_updates():
    metrics = InferenceMetrics()

    metrics.record(
        status="ACCEPTED",
        duration_ms=10.0,
    )

    first_snapshot = metrics.snapshot()

    metrics.record(
        status="FAILED",
        duration_ms=20.0,
    )

    assert first_snapshot.total_attempts == 1
    assert first_snapshot.accepted == 1
    assert first_snapshot.failed == 0

    second_snapshot = metrics.snapshot()

    assert second_snapshot.total_attempts == 2
    assert second_snapshot.failed == 1


def test_snapshot_rejects_counter_mismatch():
    with pytest.raises(
        InferenceMetricsError,
        match="Outcome counters must equal",
    ):
        InferenceMetricsSnapshot(
            total_attempts=3,
            accepted=1,
            uncertain=1,
            failed=0,
            total_latency_ms=10.0,
            average_latency_ms=(
                10.0 / 3.0
            ),
            max_latency_ms=10.0,
        )


def test_snapshot_rejects_negative_counter():
    with pytest.raises(
        InferenceMetricsError,
        match="must not be negative",
    ):
        InferenceMetricsSnapshot(
            total_attempts=-1,
            accepted=0,
            uncertain=0,
            failed=0,
            total_latency_ms=0.0,
            average_latency_ms=0.0,
            max_latency_ms=0.0,
        )


def test_concurrent_recording_is_thread_safe():
    metrics = InferenceMetrics()

    def record_accepted(
        _: int,
    ) -> None:
        metrics.record(
            status="ACCEPTED",
            duration_ms=1.0,
        )

    with ThreadPoolExecutor(
        max_workers=8
    ) as executor:
        list(
            executor.map(
                record_accepted,
                range(1000),
            )
        )

    snapshot = metrics.snapshot()

    assert snapshot.total_attempts == 1000
    assert snapshot.accepted == 1000
    assert snapshot.uncertain == 0
    assert snapshot.failed == 0

    assert snapshot.total_latency_ms == 1000.0
    assert snapshot.average_latency_ms == 1.0
    assert snapshot.max_latency_ms == 1.0