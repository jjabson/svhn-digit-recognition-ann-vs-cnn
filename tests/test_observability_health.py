from datetime import datetime, timezone

import pytest

from src.observability.health import (
    ComponentHealth,
    HealthStatus,
    RuntimeHealthSnapshot,
    create_runtime_health_snapshot,
    resolve_runtime_health_status,
)


def test_component_health_requires_name():
    with pytest.raises(
        ValueError,
        match="name must not be empty",
    ):
        ComponentHealth(
            name="",
            status=HealthStatus.HEALTHY,
        )


def test_all_healthy_components_resolve_healthy():
    status = resolve_runtime_health_status(
        [
            ComponentHealth(
                name="model",
                status=HealthStatus.HEALTHY,
            ),
            ComponentHealth(
                name="runtime_config",
                status=HealthStatus.HEALTHY,
            ),
        ]
    )

    assert status is HealthStatus.HEALTHY


def test_degraded_component_resolves_degraded():
    status = resolve_runtime_health_status(
        [
            ComponentHealth(
                name="model",
                status=HealthStatus.HEALTHY,
            ),
            ComponentHealth(
                name="audit_store",
                status=HealthStatus.DEGRADED,
            ),
        ]
    )

    assert status is HealthStatus.DEGRADED


def test_unhealthy_component_takes_precedence():
    status = resolve_runtime_health_status(
        [
            ComponentHealth(
                name="model",
                status=HealthStatus.DEGRADED,
            ),
            ComponentHealth(
                name="runtime_config",
                status=HealthStatus.UNHEALTHY,
            ),
        ]
    )

    assert status is HealthStatus.UNHEALTHY


def test_empty_component_collection_is_healthy():
    status = resolve_runtime_health_status([])

    assert status is HealthStatus.HEALTHY


def test_snapshot_uses_component_status_precedence():
    checked_at = datetime(
        2026,
        10,
        28,
        12,
        0,
        tzinfo=timezone.utc,
    )

    snapshot = create_runtime_health_snapshot(
        [
            ComponentHealth(
                name="model",
                status=HealthStatus.HEALTHY,
            ),
            ComponentHealth(
                name="approval_store",
                status=HealthStatus.DEGRADED,
            ),
        ],
        checked_at=checked_at,
    )

    assert (
        snapshot.status
        is HealthStatus.DEGRADED
    )

    assert snapshot.checked_at == checked_at

    assert len(snapshot.components) == 2


def test_snapshot_components_are_immutable_tuple():
    snapshot = create_runtime_health_snapshot(
        [
            ComponentHealth(
                name="model",
                status=HealthStatus.HEALTHY,
            ),
        ]
    )

    assert isinstance(
        snapshot.components,
        tuple,
    )


def test_snapshot_requires_timezone_aware_time():
    with pytest.raises(
        ValueError,
        match="timezone-aware",
    ):
        RuntimeHealthSnapshot(
            status=HealthStatus.HEALTHY,
            checked_at=datetime(
                2026,
                10,
                28,
                12,
                0,
            ),
            components=(),
        )


def test_snapshot_default_time_is_timezone_aware():
    snapshot = create_runtime_health_snapshot(
        []
    )

    assert snapshot.checked_at.tzinfo is not None