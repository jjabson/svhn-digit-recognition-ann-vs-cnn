import sqlite3
from langgraph.types import Command

from langgraph.checkpoint.sqlite import SqliteSaver

from config.inference_config import InferenceConfig
from config.inference_config_service import InferenceConfigService
from agent.approval_graph import build_approval_graph


def test_approval_checkpoint_survives_graph_reconstruction(
    tmp_path,
):
    database_file = tmp_path / "approval_checkpoints.db"

    config_service = InferenceConfigService(
        InferenceConfig(
            model_name="cnn",
            confidence_threshold=0.90,
        )
    )

    first_connection = sqlite3.connect(
        database_file,
        check_same_thread=False,
    )

    first_checkpointer = SqliteSaver(first_connection)
    first_checkpointer.setup()

    first_graph = build_approval_graph(
        config_service=config_service,
        checkpointer=first_checkpointer,
    )

    thread_id = "persistent-approval-test"

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

    first_connection.close()

    second_connection = sqlite3.connect(
        database_file,
        check_same_thread=False,
    )

    second_checkpointer = SqliteSaver(second_connection)
    second_checkpointer.setup()

    second_graph = build_approval_graph(
        config_service=config_service,
        checkpointer=second_checkpointer,
    )

    state = second_graph.get_state(graph_config)

    assert state is not None
    assert state.values["requested_threshold"] == 0.95
    assert "approval_request" in state.values

    second_connection.close()

def test_approval_can_resume_after_graph_reconstruction(
    tmp_path,
):
    database_file = tmp_path / "approval_checkpoints.db"

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

    first_connection = sqlite3.connect(
        database_file,
        check_same_thread=False,
    )

    first_checkpointer = SqliteSaver(first_connection)
    first_checkpointer.setup()

    first_graph = build_approval_graph(
        config_service=config_service,
        checkpointer=first_checkpointer,
    )

    first_result = first_graph.invoke(
        {
            "requested_threshold": 0.95,
            "reason": "Restart persistence test",
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

    second_connection = sqlite3.connect(
        database_file,
        check_same_thread=False,
    )

    second_checkpointer = SqliteSaver(second_connection)
    second_checkpointer.setup()

    second_graph = build_approval_graph(
        config_service=config_service,
        checkpointer=second_checkpointer,
    )

    resumed_result = second_graph.invoke(
        Command(
            resume={
                "approved": True,
                "reason": "Approved after restart",
            }
        ),
        config=graph_config,
    )

    assert resumed_result["action_status"] == "executed"

    assert (
        resumed_result["action_result"][
            "confidence_threshold"
        ]
        == 0.95
    )

    assert (
        config_service
        .get_config()
        .confidence_threshold
        == 0.95
    )

    second_connection.close()