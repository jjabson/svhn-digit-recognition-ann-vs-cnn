from mcp_server.server import (
    get_inference_config,
    inference_config_service,
    inference_service,
    resolve_confidence_threshold_approval,
    set_confidence_threshold,
)


def test_mcp_threshold_change_requires_approval():
    original = (
        inference_config_service
        .get_config()
        .confidence_threshold
    )

    result = set_confidence_threshold(0.95)

    assert result["status"] == "approval_required"

    assert (
        inference_config_service
        .get_config()
        .confidence_threshold
        == original
    )


def test_mcp_threshold_change_executes_after_approval():
    result = set_confidence_threshold(0.95)

    resolved = resolve_confidence_threshold_approval(
        thread_id=result["thread_id"],
        approved=True,
        reason="Approved for test",
    )

    assert resolved["status"] == "executed"

    assert (
        resolved["confidence_threshold"]
        == 0.95
    )


def test_mcp_threshold_change_remains_unchanged_after_rejection():
    # Restore known starting state first.
    inference_config_service.set_confidence_threshold(0.90)

    result = set_confidence_threshold(0.95)

    resolved = resolve_confidence_threshold_approval(
        thread_id=result["thread_id"],
        approved=False,
        reason="Rejected for test",
    )

    assert resolved["status"] == "rejected"

    assert (
        inference_config_service
        .get_config()
        .confidence_threshold
        == 0.90
    )

def test_get_inference_config_reflects_approved_threshold_change():
    inference_config_service.set_confidence_threshold(0.90)

    request = set_confidence_threshold(0.95)

    resolved = resolve_confidence_threshold_approval(
        thread_id=request["thread_id"],
        approved=True,
        reason="Approved for read-after-write test",
    )

    assert resolved["status"] == "executed"

    config = get_inference_config()

    assert config.model_name == "cnn"
    assert config.confidence_threshold == 0.95

def test_inference_service_uses_shared_runtime_config():
    inference_config_service.set_confidence_threshold(0.90)

    assert (
        inference_service.policy.confidence_threshold
        == 0.90
    )

    inference_config_service.set_confidence_threshold(0.95)

    assert (
        inference_service.policy.confidence_threshold
        == 0.95
    )

def test_persisted_threshold_survives_service_reconstruction(tmp_path):
    from config.inference_config import InferenceConfig
    from config.inference_config_service import InferenceConfigService
    from config.inference_config_store import InferenceConfigStore

    database_file = tmp_path / "runtime_state.db"

    store = InferenceConfigStore(
        database_file=database_file,
    )

    first_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=store,
    )

    first_service.set_confidence_threshold(0.95)

    second_store = InferenceConfigStore(
        database_file=database_file,
    )

    second_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        ),
        store=second_store,
    )

    assert (
        second_service.get_config().confidence_threshold
        == 0.95
    )