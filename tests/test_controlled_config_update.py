import pytest

from config.inference_config import InferenceConfig
from config.inference_config_service import InferenceConfigService
from config.approval_artifact_store import ApprovalArtifactStore
from src.orchestration.approval_artifact import ApprovalArtifact
from src.orchestration.trusted_approval import (
    ApprovalAlreadyConsumedError,
    TrustedApprovalVerifier,
)
from src.orchestration.action_policy import ActionPolicy
from src.orchestration.approval import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
)
from src.orchestration.controlled_executor import (
    ApprovalDeniedError,
    ApprovalMismatchError,
    ApprovalRequiredError,
    ControlledActionExecutor,
)


def test_approved_threshold_update_changes_active_config():
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase inference confidence threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    updated = executor.execute(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        action=config_service.set_confidence_threshold,
        approval_request=request,
        approval_decision=decision,
    )

    assert updated.confidence_threshold == 0.95
    assert config_service.get_config().confidence_threshold == 0.95


def test_threshold_update_without_approval_does_not_change_config():
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    executor = ControlledActionExecutor(ActionPolicy())

    with pytest.raises(ApprovalRequiredError):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            action=config_service.set_confidence_threshold,
        )

    assert config_service.get_config().confidence_threshold == 0.90


def test_rejected_threshold_update_does_not_change_config():
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase inference confidence threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.REJECTED,
        decided_by="human",
    )

    with pytest.raises(ApprovalDeniedError):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.95},
            action=config_service.set_confidence_threshold,
            approval_request=request,
            approval_decision=decision,
        )

    assert config_service.get_config().confidence_threshold == 0.90


def test_mismatched_threshold_arguments_do_not_change_config():
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 0.95},
        reason="Increase inference confidence threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    with pytest.raises(ApprovalMismatchError):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 0.50},
            action=config_service.set_confidence_threshold,
            approval_request=request,
            approval_decision=decision,
        )

    assert config_service.get_config().confidence_threshold == 0.90


def test_invalid_approved_threshold_does_not_replace_config():
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    executor = ControlledActionExecutor(ActionPolicy())

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={"confidence_threshold": 1.50},
        reason="Attempt invalid threshold",
    )

    decision = ApprovalDecision(
        request_id=request.request_id,
        status=ApprovalStatus.APPROVED,
        decided_by="human",
    )

    with pytest.raises(ValueError):
        executor.execute(
            action_name="set_confidence_threshold",
            arguments={"confidence_threshold": 1.50},
            action=config_service.set_confidence_threshold,
            approval_request=request,
            approval_decision=decision,
        )

    assert config_service.get_config().confidence_threshold == 0.90


def test_trusted_approved_threshold_update_is_single_use(
    tmp_path,
):
    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    artifact_store = ApprovalArtifactStore(
        tmp_path / "approval_artifacts.db"
    )

    verifier = TrustedApprovalVerifier(
        artifact_store,
        trusted_issuers={
            "local_trusted_human_channel",
        },
    )

    executor = ControlledActionExecutor(
        ActionPolicy(),
        trusted_approval_verifier=verifier,
        approval_artifact_store=artifact_store,
    )

    request = ApprovalRequest(
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        reason="Increase inference confidence threshold",
    )

    artifact = ApprovalArtifact.issue(
        request_id=request.request_id,
        action_name=request.action_name,
        arguments=request.arguments,
        approved=True,
        issued_by="local_trusted_human_channel",
    )

    artifact_store.save(artifact)

    verified = verifier.verify(request)

    updated = executor.execute_trusted(
        action_name="set_confidence_threshold",
        arguments={
            "confidence_threshold": 0.95,
        },
        action=config_service.set_confidence_threshold,
        approval_request=request,
        verified_approval=verified,
    )

    assert updated.confidence_threshold == 0.95
    assert (
        config_service.get_config().confidence_threshold
        == 0.95
    )

    persisted_artifact = artifact_store.get_by_artifact_id(
        artifact.artifact_id
    )

    assert persisted_artifact is not None
    assert persisted_artifact.is_consumed is True

    with pytest.raises(
        ApprovalAlreadyConsumedError,
    ):
        executor.execute_trusted(
            action_name="set_confidence_threshold",
            arguments={
                "confidence_threshold": 0.95,
            },
            action=config_service.set_confidence_threshold,
            approval_request=request,
            verified_approval=verified,
        )