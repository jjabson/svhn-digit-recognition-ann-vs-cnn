import asyncio

from agent.mcp_client import MCPToolDefinition, MCPToolResult
from agent.state import AgentState

from agent.nodes import (
    discover_tools_node,
    execute_tool_node,
    route_after_tool_selection,
    select_tool_node,
    synthesize_response_node,
)

class FakeMCPToolClient:
    async def list_tools(self) -> list[MCPToolDefinition]:
        return [
            MCPToolDefinition(
                name="get_evaluation_summary",
                description="Return evaluation summary",
                input_schema={"type": "object"},
            ),
            MCPToolDefinition(
                name="predict_digit",
                description="Predict a digit",
                input_schema={
                    "type": "object",
                    "properties": {
                        "image_data": {"type": "string"},
                    },
                },
            ),
        ]

class FakeToolExecutionClient:
    def __init__(self) -> None:
        self.called_name: str | None = None
        self.called_arguments: dict | None = None

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
    ) -> MCPToolResult:
        self.called_name = name
        self.called_arguments = arguments

        return MCPToolResult(
            tool_name=name,
            structured_content={
                "accuracy": 0.951375,
                "evaluation_samples": 24000,
            },
            is_error=False,
            error_message=None,
        )


def test_discover_tools_node_returns_available_tools() -> None:
    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
    }

    client = FakeMCPToolClient()

    update = asyncio.run(
        discover_tools_node(
            state,
            client,  # type: ignore[arg-type]
        )
    )

    assert list(update) == ["available_tools"]

    tools = update["available_tools"]

    assert len(tools) == 2
    assert tools[0].name == "get_evaluation_summary"
    assert tools[1].name == "predict_digit"

def test_select_tool_node_selects_evaluation_summary_for_accuracy_request() -> None:
    tool = MCPToolDefinition(
        name="get_evaluation_summary",
        description="Return evaluation summary",
        input_schema={"type": "object"},
    )

    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [tool],
        "selected_tool": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
    }

    update = select_tool_node(state)

    assert update == {
        "selected_tool": "get_evaluation_summary",
        "selection_reason": "accuracy_request",
    }


def test_select_tool_node_does_not_select_unavailable_tool() -> None:
    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
    }

    update = select_tool_node(state)

    assert update == {
        "selected_tool": None,
        "selection_reason": "capability_unavailable",
    }

def test_execute_tool_node_executes_selected_tool() -> None:
    existing_result = MCPToolResult(
        tool_name="previous_tool",
        structured_content={"value": "existing"},
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": "get_evaluation_summary",
        "selection_reason": None,
        "tool_results": [existing_result],
        "final_response": None,
    }

    client = FakeToolExecutionClient()

    update = asyncio.run(
        execute_tool_node(state, client)
    )

    assert client.called_name == "get_evaluation_summary"
    assert client.called_arguments is None

    assert len(update["tool_results"]) == 2
    assert update["tool_results"][0] == existing_result

    new_result = update["tool_results"][1]

    assert new_result.tool_name == "get_evaluation_summary"
    assert new_result.structured_content == {
        "accuracy": 0.951375,
        "evaluation_samples": 24000,
    }
    assert new_result.is_error is False

def test_execute_tool_node_does_not_call_client_without_selected_tool() -> None:
    existing_result = MCPToolResult(
        tool_name="previous_tool",
        structured_content={"value": "existing"},
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selection_reason": None,
        "tool_results": [existing_result],
        "final_response": None,
    }

    client = FakeToolExecutionClient()

    update = asyncio.run(
        execute_tool_node(state, client)
    )

    assert client.called_name is None
    assert client.called_arguments is None
    assert update == {
        "tool_results": [existing_result],
    }

def test_synthesize_response_node_formats_evaluation_summary() -> None:
    evaluation_result = MCPToolResult(
        tool_name="get_evaluation_summary",
        structured_content={
            "accuracy": 0.951375,
            "evaluation_samples": 24000,
        },
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": "get_evaluation_summary",
        "selection_reason": None,
        "tool_results": [evaluation_result],
        "final_response": None,
    }

    update = synthesize_response_node(state)

    assert update == {
        "final_response": (
            "The model achieved 95.14% accuracy "
            "on 24,000 evaluation samples."
        ),
    }

def test_route_after_tool_selection_routes_to_execution() -> None:
    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": "get_evaluation_summary",
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
    }

    route = route_after_tool_selection(state)

    assert route == "execute_tool"


def test_route_after_tool_selection_skips_execution_without_tool() -> None:
    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
    }

    route = route_after_tool_selection(state)

    assert route == "synthesize_response"

def test_synthesize_response_node_preserves_tool_error() -> None:
    tool_error = MCPToolResult(
        tool_name="get_evaluation_summary",
        structured_content=None,
        is_error=True,
        error_message="Evaluation service unavailable.",
    )

    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": "get_evaluation_summary",
        "selection_reason": "accuracy_request",
        "tool_results": [tool_error],
        "final_response": None,
    }

    update = synthesize_response_node(state)

    assert update == {
        "final_response": (
            "Tool get_evaluation_summary failed: "
            "Evaluation service unavailable."
        ),
    }

def test_synthesize_response_node_preserves_failed_application_decision() -> None:
    failed_decision = MCPToolResult(
        tool_name="predict_digit",
        structured_content={
            "selected_model": "cnn",
            "predicted_digit": None,
            "confidence": None,
            "status": "failed",
            "decision_reason": "Primary inference failed.",
            "fallback_used": False,
            "fallback_reason": None,
            "review_required": True,
        },
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": "What digit is this?",
        "image_bytes": b"image-bytes",
        "available_tools": [],
        "selected_tool": "predict_digit",
        "selection_reason": "prediction_request",
        "tool_results": [failed_decision],
        "final_response": None,
    }

    update = synthesize_response_node(state)

    assert update == {
        "final_response": (
            "The inference decision reported status failed. "
            "Review is required."
        ),
    }

def test_select_tool_node_marks_unsupported_request() -> None:
    state: AgentState = {
        "user_request": "Write me a poem.",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
    }

    update = select_tool_node(state)

    assert update == {
        "selected_tool": None,
        "selection_reason": "unsupported_request",
    }

def test_synthesize_response_node_formats_unavailable_capability() -> None:
    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selection_reason": "capability_unavailable",
        "tool_results": [],
        "final_response": None,
    }

    update = synthesize_response_node(state)

    assert update == {
        "final_response": (
            "I understood the request, but the required "
            "capability is not currently available."
        ),
    }