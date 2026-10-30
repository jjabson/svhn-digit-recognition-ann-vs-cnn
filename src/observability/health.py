from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    status: HealthStatus
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Component health name must not be empty."
            )

        if not isinstance(
            self.status,
            HealthStatus,
        ):
            raise ValueError(
                "Component health status must be a "
                "HealthStatus."
            )


@dataclass(frozen=True)
class RuntimeHealthSnapshot:
    status: HealthStatus
    checked_at: datetime
    components: tuple[ComponentHealth, ...]

    def __post_init__(self) -> None:
        if self.checked_at.tzinfo is None:
            raise ValueError(
                "Health snapshot time must be "
                "timezone-aware."
            )

        if not isinstance(
            self.status,
            HealthStatus,
        ):
            raise ValueError(
                "Runtime health status must be a "
                "HealthStatus."
            )


def resolve_runtime_health_status(
    components: Iterable[ComponentHealth],
) -> HealthStatus:
    component_list = tuple(components)

    if any(
        component.status
        is HealthStatus.UNHEALTHY
        for component in component_list
    ):
        return HealthStatus.UNHEALTHY

    if any(
        component.status
        is HealthStatus.DEGRADED
        for component in component_list
    ):
        return HealthStatus.DEGRADED

    return HealthStatus.HEALTHY


def create_runtime_health_snapshot(
    components: Iterable[ComponentHealth],
    *,
    checked_at: datetime | None = None,
) -> RuntimeHealthSnapshot:
    component_tuple = tuple(components)

    timestamp = (
        checked_at
        if checked_at is not None
        else datetime.now(timezone.utc)
    )

    return RuntimeHealthSnapshot(
        status=resolve_runtime_health_status(
            component_tuple
        ),
        checked_at=timestamp,
        components=component_tuple,
    )