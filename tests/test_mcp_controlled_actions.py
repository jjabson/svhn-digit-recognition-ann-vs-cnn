from __future__ import annotations

import pytest

import mcp_server.server as server
from mcp.server.mcpserver.exceptions import ToolError


def _different_threshold(
    current_threshold: float,
) -> float:
    """
    Return a valid threshold different from the current value.
    """
    if current_threshold != 0.95:
        return 0.95

    return 0.90


def test_mcp_threshold_change_requires_trusted_approval():
    """
    MCP may request a controlled threshold change, but the request
    must not mutate runtime configuration before trusted approval.
    """
    current_threshold = (
        server.inference_config_service
        .get_config()
        .confidence_threshold
    )

    requested_threshold = _different_threshold(
        current_threshold
    )

    result = server.set_confidence_threshold(
        requested_threshold
    )

    assert result["status"] == "approval_required"
    assert result["thread_id"]
    assert result["request_id"]

    assert (
        result["action_name"]
        == "set_confidence_threshold"
    )

    assert result["arguments"] == {
        "confidence_threshold": requested_threshold,
    }

    assert (
        result["current_confidence_threshold"]
        == current_threshold
    )

    assert (
        server.inference_config_service
        .get_config()
        .confidence_threshold
        == current_threshold
    )


def test_mcp_can_inspect_pending_approval_status():
    """
    MCP may inspect a controlled-action workflow but cannot authorize it.
    """
    current_threshold = (
        server.inference_config_service
        .get_config()
        .confidence_threshold
    )

    requested_threshold = _different_threshold(
        current_threshold
    )

    request_result = (
        server.set_confidence_threshold(
            requested_threshold
        )
    )

    status_result = (
        server
        .get_confidence_threshold_approval_status(
            request_result["thread_id"]
        )
    )

    assert (
        status_result["status"]
        == "approval_required"
    )

    assert (
        status_result["thread_id"]
        == request_result["thread_id"]
    )

    assert (
        status_result["request_id"]
        == request_result["request_id"]
    )

    assert (
        status_result["action_name"]
        == "set_confidence_threshold"
    )

    assert status_result["arguments"] == {
        "confidence_threshold": requested_threshold,
    }

    assert (
        status_result["confidence_threshold"]
        == current_threshold
    )

    assert (
        server.inference_config_service
        .get_config()
        .confidence_threshold
        == current_threshold
    )


def test_mcp_has_no_approval_authority():
    """
    The former MCP approval resolver must not exist.

    An MCP consumer must never be able to authorize a controlled
    action by supplying approved=True.
    """
    assert not hasattr(
        server,
        "resolve_confidence_threshold_approval",
    )


def test_unknown_approval_thread_cannot_be_inspected():
    """
    Unknown workflow identifiers fail closed.
    """
    with pytest.raises(
        ToolError,
        match=(
            "No confidence-threshold approval "
            "workflow found"
        ),
    ):
        (
            server
            .get_confidence_threshold_approval_status(
                "unknown-threshold-thread"
            )
        )