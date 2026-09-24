import asyncio

from agent.mcp_client import MCPToolDefinition, MCPToolResult
from agent.state import AgentState

from agent.nodes import (
    discover_tools_node,
    execute_tool_node,
    route_after_tool_selection,
    select_tool_node,
    synthesize_response_node,
    select_follow_up_tool_node,
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
        "selected_tool_arguments": None,
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
        "selected_tool_arguments": None,
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
        "selected_tool_arguments": None,
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
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [existing_result],
        "final_response": None,
        "workflow_status": None,
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
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [existing_result],
        "final_response": None,
        "workflow_status": None,
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
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [evaluation_result],
        "final_response": None,
        "workflow_status": None,
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
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
    }

    route = route_after_tool_selection(state)

    assert route == "execute_tool"


def test_route_after_tool_selection_skips_execution_without_tool() -> None:
    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
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
        "selected_tool_arguments": None,
        "selection_reason": "accuracy_request",
        "tool_results": [tool_error],
        "final_response": None,
        "workflow_status": None,
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
        "selected_tool_arguments": None,
        "selected_tool": "predict_digit",
        "selection_reason": "prediction_request",
        "tool_results": [failed_decision],
        "final_response": None,
        "workflow_status": None,
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
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
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
        "selected_tool_arguments": None,
        "selection_reason": "capability_unavailable",
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
    }

    update = synthesize_response_node(state)

    assert update == {
        "final_response": (
            "I understood the request, but the required "
            "capability is not currently available."
        ),
    }

def test_select_tool_node_selects_insights_for_worst_digit_request() -> None:
    tool = MCPToolDefinition(
        name="get_evaluation_insights",
        description="Return evaluation insights",
        input_schema={"type": "object"},
    )

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [tool],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
    }

    update = select_tool_node(state)

    assert update == {
        "selected_tool": "get_evaluation_insights",
        "selection_reason": "worst_digit_request",
    }

def test_execute_tool_node_passes_selected_tool_arguments() -> None:
    class FakeToolExecutionClient:
        def __init__(self) -> None:
            self.called_tool_name: str | None = None
            self.called_arguments: dict | None = None

        async def call_tool(
            self,
            name: str,
            arguments: dict | None = None,
        ) -> MCPToolResult:
            self.called_tool_name = name
            self.called_arguments = arguments

            return MCPToolResult(
                tool_name=name,
                structured_content={
                    "digit": 3,
                },
                is_error=False,
                error_message=None,
            )

    client = FakeToolExecutionClient()

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": "get_digit_metrics",
        "selected_tool_arguments": {"digit": 3},
        "selection_reason": "worst_digit_request",
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
    }

    update = asyncio.run(
        execute_tool_node(state, client)
    )

    assert client.called_tool_name == "get_digit_metrics"
    assert client.called_arguments == {"digit": 3}
    assert len(update["tool_results"]) == 1
    assert update["tool_results"][0].tool_name == "get_digit_metrics"

def test_select_follow_up_tool_node_selects_metrics_for_worst_digit() -> None:
    insights_result = MCPToolResult(
        tool_name="get_evaluation_insights",
        structured_content={
            "best_performing_digit": 0,
            "worst_performing_digit": 3,
            "most_common_confusion": {
                "true_digit": 3,
                "predicted_digit": 5,
                "count": 73,
            },
        },
        is_error=False,
        error_message=None,
    )

    metrics_tool = MCPToolDefinition(
        name="get_digit_metrics",
        description="Return evaluation metrics for one digit",
        input_schema={
            "type": "object",
            "properties": {
                "digit": {"type": "integer"},
            },
            "required": ["digit"],
        },
    )

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [metrics_tool],
        "selected_tool": "get_evaluation_insights",
        "selected_tool_arguments": None,
        "selection_reason": "worst_digit_request",
        "tool_results": [insights_result],
        "final_response": None,
        "workflow_status": None,
    }

    update = select_follow_up_tool_node(state)

    assert update == {
        "selected_tool": "get_digit_metrics",
        "selected_tool_arguments": {"digit": 3},
        "workflow_status": None,
    }

def test_select_follow_up_tool_node_skips_unavailable_metrics_tool() -> None:
    insights_result = MCPToolResult(
        tool_name="get_evaluation_insights",
        structured_content={
            "best_performing_digit": 0,
            "worst_performing_digit": 3,
            "most_common_confusion": {
                "true_digit": 3,
                "predicted_digit": 5,
                "count": 73,
            },
        },
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": "get_evaluation_insights",
        "selected_tool_arguments": None,
        "selection_reason": "worst_digit_request",
        "tool_results": [insights_result],
        "final_response": None,
        "workflow_status": None,
    }

    update = select_follow_up_tool_node(state)

    assert update == {
        "selected_tool": None,
        "selected_tool_arguments": None,
        "workflow_status": "follow_up_capability_unavailable",
    }

def test_select_follow_up_tool_node_stops_after_digit_metrics() -> None:
    metrics_result = MCPToolResult(
        tool_name="get_digit_metrics",
        structured_content={
            "digit": 3,
            "precision": 0.95,
            "recall": 0.94,
            "f1_score": 0.945,
        },
        is_error=False,
        error_message=None,
    )

    metrics_tool = MCPToolDefinition(
        name="get_digit_metrics",
        description="Return evaluation metrics for one digit",
        input_schema={
            "type": "object",
            "properties": {
                "digit": {"type": "integer"},
            },
            "required": ["digit"],
        },
    )

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [metrics_tool],
        "selected_tool": "get_digit_metrics",
        "selected_tool_arguments": {"digit": 3},
        "selection_reason": "worst_digit_request",
        "tool_results": [metrics_result],
        "final_response": None,
        "workflow_status": None,
    }

    update = select_follow_up_tool_node(state)

    assert update == {
        "selected_tool": None,
        "selected_tool_arguments": None,
    }

def test_synthesize_response_node_combines_worst_digit_results() -> None:
    insights_result = MCPToolResult(
        tool_name="get_evaluation_insights",
        structured_content={
            "best_performing_digit": 0,
            "worst_performing_digit": 3,
            "most_common_confusion": {
                "true_digit": 3,
                "predicted_digit": 5,
                "count": 73,
            },
        },
        is_error=False,
        error_message=None,
    )

    metrics_result = MCPToolResult(
        tool_name="get_digit_metrics",
        structured_content={
            "digit": 3,
            "precision": 0.95,
            "recall": 0.94,
            "f1_score": 0.945,
        },
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": "worst_digit_request",
        "tool_results": [
            insights_result,
            metrics_result,
        ],
        "final_response": None,
        "workflow_status": None,
    }

    update = synthesize_response_node(state)

    assert update == {
        "final_response": (
            "The model struggles most with digit 3. "
            "For digit 3, precision is 95.00%, "
            "recall is 94.00%, and F1 score is 94.50%."
        ),
    }

def test_select_follow_up_tool_node_stops_after_tool_error() -> None:
    insights_error = MCPToolResult(
        tool_name="get_evaluation_insights",
        structured_content=None,
        is_error=True,
        error_message="Evaluation insights unavailable.",
    )

    metrics_tool = MCPToolDefinition(
        name="get_digit_metrics",
        description="Return evaluation metrics for one digit",
        input_schema={
            "type": "object",
            "properties": {
                "digit": {"type": "integer"},
            },
            "required": ["digit"],
        },
    )

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [metrics_tool],
        "selected_tool": "get_evaluation_insights",
        "selected_tool_arguments": None,
        "selection_reason": "worst_digit_request",
        "tool_results": [insights_error],
        "final_response": None,
        "workflow_status": None,
    }

    update = select_follow_up_tool_node(state)

    assert update == {
        "selected_tool": None,
        "selected_tool_arguments": None,
    }

def test_select_follow_up_tool_node_reports_unavailable_capability() -> None:
    insights_result = MCPToolResult(
        tool_name="get_evaluation_insights",
        structured_content={
            "best_performing_digit": 0,
            "worst_performing_digit": 3,
            "most_common_confusion": {
                "true_digit": 3,
                "predicted_digit": 5,
                "count": 73,
            },
        },
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": "get_evaluation_insights",
        "selected_tool_arguments": None,
        "selection_reason": "worst_digit_request",
        "workflow_status": None,
        "tool_results": [insights_result],
        "final_response": None,
    }

    update = select_follow_up_tool_node(state)

    assert update == {
        "selected_tool": None,
        "selected_tool_arguments": None,
        "workflow_status": "follow_up_capability_unavailable",
    }

def test_synthesize_response_node_rejects_mismatched_digit_metrics() -> None:
    insights_result = MCPToolResult(
        tool_name="get_evaluation_insights",
        structured_content={
            "best_performing_digit": 0,
            "worst_performing_digit": 3,
            "most_common_confusion": {
                "true_digit": 3,
                "predicted_digit": 5,
                "count": 73,
            },
        },
        is_error=False,
        error_message=None,
    )

    metrics_result = MCPToolResult(
        tool_name="get_digit_metrics",
        structured_content={
            "digit": 7,
            "precision": 0.95,
            "recall": 0.94,
            "f1_score": 0.945,
        },
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": "worst_digit_request",
        "workflow_status": None,
        "tool_results": [
            insights_result,
            metrics_result,
        ],
        "final_response": None,
    }

    update = synthesize_response_node(state)

    assert update == {
        "final_response": (
            "The tool results are inconsistent: the identified "
            "worst-performing digit is 3, but the returned metrics "
            "are for digit 7."
        ),
    }