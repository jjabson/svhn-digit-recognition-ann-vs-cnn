import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from agent.approval_graph import build_approval_graph
from config.approval_artifact_store import (
    ApprovalArtifactStore,
)
from config.inference_config import InferenceConfig
from config.inference_config_service import (
    InferenceConfigService,
)
from src.orchestration.approval import ApprovalRequest
from src.orchestration.trusted_approval import (
    TrustedApprovalVerifier,
)
from src.orchestration.trusted_approval_service import (
    DEFAULT_TRUSTED_APPROVAL_ISSUER,
    TrustedApprovalService,
)


def _create_verifier(
    artifact_store: ApprovalArtifactStore,
) -> TrustedApprovalVerifier:
    return TrustedApprovalVerifier(
        artifact_store,
        trusted_issuers={
            DEFAULT_TRUSTED_APPROVAL_ISSUER,
        },
    )


def _request_from_state(
    state,
) -> ApprovalRequest:
    request_data = state.values[
        "approval_request"
    ]

    return ApprovalRequest(
        request_id=request_data["request_id"],
        action_name=request_data["action_name"],
        arguments=request_data["arguments"],
        reason=request_data["reason"],
    )


def test_approval_checkpoint_survives_graph_reconstruction(
    tmp_path,
):
    checkpoint_database_file = (
        tmp_path / "approval_checkpoints.db"
    )

    artifact_database_file = (
        tmp_path / "approval_artifacts.db"
    )

    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    first_artifact_store = ApprovalArtifactStore(
        artifact_database_file
    )

    first_verifier = _create_verifier(
        first_artifact_store
    )

    first_connection = sqlite3.connect(
        checkpoint_database_file,
        check_same_thread=False,
    )

    first_checkpointer = SqliteSaver(
        first_connection
    )

    first_checkpointer.setup()

    first_graph = build_approval_graph(
        config_service=config_service,
        artifact_store=first_artifact_store,
        trusted_approval_verifier=first_verifier,
        checkpointer=first_checkpointer,
    )

    thread_id = (
        "persistent-approval-test"
    )

    graph_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    first_result = first_graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Persistence test",
        },
        config=graph_config,
    )

    assert "__interrupt__" in first_result

    request_id_before_restart = (
        first_result["__interrupt__"][0]
        .value["request_id"]
    )

    first_connection.close()

    # Simulate reconstruction in a new runtime.
    second_artifact_store = ApprovalArtifactStore(
        artifact_database_file
    )

    second_verifier = _create_verifier(
        second_artifact_store
    )

    second_connection = sqlite3.connect(
        checkpoint_database_file,
        check_same_thread=False,
    )

    second_checkpointer = SqliteSaver(
        second_connection
    )

    second_checkpointer.setup()

    second_graph = build_approval_graph(
        config_service=config_service,
        artifact_store=second_artifact_store,
        trusted_approval_verifier=second_verifier,
        checkpointer=second_checkpointer,
    )

    state = second_graph.get_state(
        graph_config
    )

    assert state is not None

    assert (
        state.values["requested_threshold"]
        == 0.95
    )

    assert (
        "approval_request"
        in state.values
    )

    assert (
        state.values["approval_request"]
        ["request_id"]
        == request_id_before_restart
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.90
    )

    second_connection.close()


def test_approval_can_resume_after_graph_reconstruction(
    tmp_path,
):
    checkpoint_database_file = (
        tmp_path / "approval_checkpoints.db"
    )

    artifact_database_file = (
        tmp_path / "approval_artifacts.db"
    )

    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    thread_id = "restart-resume-test"

    graph_config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    first_artifact_store = ApprovalArtifactStore(
        artifact_database_file
    )

    first_verifier = _create_verifier(
        first_artifact_store
    )

    first_connection = sqlite3.connect(
        checkpoint_database_file,
        check_same_thread=False,
    )

    first_checkpointer = SqliteSaver(
        first_connection
    )

    first_checkpointer.setup()

    first_graph = build_approval_graph(
        config_service=config_service,
        artifact_store=first_artifact_store,
        trusted_approval_verifier=first_verifier,
        checkpointer=first_checkpointer,
    )

    first_result = first_graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": (
                "Restart persistence test"
            ),
        },
        config=graph_config,
    )

    assert "__interrupt__" in first_result

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.90
    )

    first_connection.close()

    # Reconstruct both durable workflow and
    # trusted-approval dependencies.
    second_artifact_store = ApprovalArtifactStore(
        artifact_database_file
    )

    second_verifier = _create_verifier(
        second_artifact_store
    )

    trusted_service = TrustedApprovalService(
        second_artifact_store
    )

    second_connection = sqlite3.connect(
        checkpoint_database_file,
        check_same_thread=False,
    )

    second_checkpointer = SqliteSaver(
        second_connection
    )

    second_checkpointer.setup()

    second_graph = build_approval_graph(
        config_service=config_service,
        artifact_store=second_artifact_store,
        trusted_approval_verifier=second_verifier,
        checkpointer=second_checkpointer,
    )

    recovered_state = second_graph.get_state(
        graph_config
    )

    assert recovered_state is not None

    request = _request_from_state(
        recovered_state
    )

    trusted_service.record_decision(
        request,
        approved=True,
    )

    resumed_result = second_graph.invoke(
        Command(
            resume={
                "source": (
                    "trusted_local_human_channel"
                )
            }
        ),
        config=graph_config,
    )

    assert (
        resumed_result["action_status"]
        == "executed"
    )

    assert (
        resumed_result["action_result"]
        ["confidence_threshold"]
        == 0.95
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.95
    )

    persisted_artifact = (
        second_artifact_store
        .get_by_request_id(
            request.request_id
        )
    )

    assert persisted_artifact is not None
    assert persisted_artifact.approved is True
    assert persisted_artifact.is_consumed is True

    second_connection.close()