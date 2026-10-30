from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from collections.abc import Mapping
from typing import Any


class OperationalLogError(ValueError):
    pass


class SensitiveLogFieldError(
    OperationalLogError
):
    pass


class LogStatus(str, Enum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DENIED = "denied"
    DEGRADED = "degraded"


_SENSITIVE_FIELD_NAMES = {
    "authorization",
    "bearer_token",
    "access_token",
    "token",
    "password",
    "secret",
    "api_key",
    "approval_artifact",
    "artifact_nonce",
    "nonce",
}


def _validate_nonblank(
    name: str,
    value: str,
) -> None:
    if not value.strip():
        raise OperationalLogError(
            f"{name} must not be empty."
        )


def _validate_safe_fields(
    value: Any,
    *,
    path: str = "fields",
) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested_value in value.items():
            key = str(raw_key)

            if key.lower() in _SENSITIVE_FIELD_NAMES:
                raise SensitiveLogFieldError(
                    "Sensitive field must not be "
                    f"logged: {path}.{key}"
                )

            _validate_safe_fields(
                nested_value,
                path=f"{path}.{key}",
            )

        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_fields(
                item,
                path=f"{path}[{index}]",
            )


def _json_copy(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    try:
        serialized = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
        )
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise OperationalLogError(
            "Operational log fields must be "
            "JSON-serializable."
        ) from exc

    decoded = json.loads(serialized)

    if not isinstance(decoded, dict):
        raise OperationalLogError(
            "Operational log fields must be "
            "a JSON object."
        )

    return decoded


@dataclass(frozen=True)
class OperationalLogEvent:
    event: str
    component: str

    status: LogStatus | None = None

    request_id: str | None = None
    thread_id: str | None = None
    runtime_instance_id: str | None = None

    duration_ms: float | None = None

    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )

    fields: Mapping[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _validate_nonblank(
            "event",
            self.event,
        )

        _validate_nonblank(
            "component",
            self.component,
        )

        for name, value in (
            ("request_id", self.request_id),
            ("thread_id", self.thread_id),
            (
                "runtime_instance_id",
                self.runtime_instance_id,
            ),
        ):
            if value is not None:
                _validate_nonblank(
                    name,
                    value,
                )

        if (
            self.status is not None
            and not isinstance(
                self.status,
                LogStatus,
            )
        ):
            raise OperationalLogError(
                "status must be a LogStatus."
            )

        if self.duration_ms is not None:
            if self.duration_ms < 0:
                raise OperationalLogError(
                    "duration_ms must not be negative."
                )

        if self.occurred_at.tzinfo is None:
            raise OperationalLogError(
                "occurred_at must be "
                "timezone-aware."
            )

        _validate_safe_fields(
            self.fields
        )

        safe_fields = _json_copy(
            self.fields
        )

        object.__setattr__(
            self,
            "fields",
            safe_fields,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": self.event,
            "component": self.component,
            "occurred_at": (
                self.occurred_at.isoformat()
            ),
        }

        if self.status is not None:
            payload["status"] = (
                self.status.value
            )

        if self.request_id is not None:
            payload["request_id"] = (
                self.request_id
            )

        if self.thread_id is not None:
            payload["thread_id"] = (
                self.thread_id
            )

        if self.runtime_instance_id is not None:
            payload[
                "runtime_instance_id"
            ] = self.runtime_instance_id

        if self.duration_ms is not None:
            payload["duration_ms"] = (
                self.duration_ms
            )

        payload.update(
            _json_copy(self.fields)
        )

        return payload


def log_operational_event(
    logger: logging.Logger,
    event: OperationalLogEvent,
    *,
    level: int = logging.INFO,
) -> None:
    payload = event.to_dict()

    logger.log(
        level,
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )