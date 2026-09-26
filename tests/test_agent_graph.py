import asyncio

from agent.graph import build_agent_graph
from agent.mcp_client import MCPToolDefinition, MCPToolResult
from agent.state import AgentState


class FakeMCPToolClient:
    def __init__(
            self,
            include_evaluation_summary: bool = True,
            include_multi_tool_capabilities: bool = False,
            fail_evaluation_insights: bool = False,
            include_digit_metrics: bool = True,
            fail_digit_metrics: bool = False,
    ) -> None:
        self.list_tools_call_count = 0
        self.called_tool_name: str | None = None
        self.call_tool_calls: list[tuple[str, dict | None]] = []
        self.include_evaluation_summary = include_evaluation_summary
        self.include_multi_tool_capabilities = (
            include_multi_tool_capabilities
        )
        self.fail_evaluation_insights = fail_evaluation_insights
        self.include_digit_metrics = include_digit_metrics
        self.fail_digit_metrics = fail_digit_metrics

    async def call_tool(
            self,
            name: str,
            arguments: dict | None = None,
    ) -> MCPToolResult:
        self.called_tool_name = name
        self.call_tool_calls.append((name, arguments))

        if name == "get_evaluation_insights":
            if self.fail_evaluation_insights:
                return MCPToolResult(
                    tool_name=name,
                    structured_content=None,
                    is_error=True,
                    error_message="Evaluation insights unavailable.",
                )
            return MCPToolResult(
                tool_name=name,
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

        if name == "get_digit_metrics":
            if self.fail_digit_metrics:
                return MCPToolResult(
                    tool_name=name,
                    structured_content=None,
                    is_error=True,
                    error_message="Digit metrics unavailable.",
                )

            return MCPToolResult(
                tool_name=name,
                structured_content={
                    "digit": arguments["digit"],
                    "precision": 0.95,
                    "recall": 0.94,
                    "f1_score": 0.945,
                },
                is_error=False,
                error_message=None,
            )

        return MCPToolResult(
            tool_name=name,
            structured_content={
                "accuracy": 0.951375,
                "evaluation_samples": 24000,
            },
            is_error=False,
            error_message=None,
        )

    async def list_tools(self) -> list[MCPToolDefinition]:
        self.list_tools_call_count += 1

        tools = [
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

        if self.include_evaluation_summary:
            tools.insert(
                0,
                MCPToolDefinition(
                    name="get_evaluation_summary",
                    description="Return evaluation summary",
                    input_schema={"type": "object"},
                ),
            )

        if self.include_multi_tool_capabilities:
            tools.append(
                MCPToolDefinition(
                    name="get_evaluation_insights",
                    description="Return evaluation insights",
                    input_schema={"type": "object"},
                )
            )

            if self.include_digit_metrics:
                tools.append(
                    MCPToolDefinition(
                        name="get_digit_metrics",
                        description="Return metrics for one digit",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "digit": {"type": "integer"},
                            },
                            "required": ["digit"],
                        },
                    )
                )

        return tools

def test_agent_graph_discovers_tools() -> None:
    client = FakeMCPToolClient()
    graph = build_agent_graph(client)

    initial_state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "tool_results": [],
        "selected_tool_arguments": None,
        "final_response": None,
        "selected_tool": None,
        "selection_reason": None,
        "workflow_status": None,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert result["selected_tool"] is None

    assert result["user_request"] == "How accurate is the model?"
    assert result["image_bytes"] is None

    assert len(result["available_tools"]) == 2
    assert result["available_tools"][0].name == "get_evaluation_summary"
    assert result["available_tools"][1].name == "predict_digit"
    assert client.list_tools_call_count == 1

    assert client.called_tool_name == "get_evaluation_summary"

    assert len(result["tool_results"]) == 1

    tool_result = result["tool_results"][0]

    assert tool_result.tool_name == "get_evaluation_summary"
    assert tool_result.structured_content == {
        "accuracy": 0.951375,
        "evaluation_samples": 24000,
    }
    assert tool_result.is_error is False

    assert result["final_response"] == (
        "The model achieved 95.14% accuracy "
        "on 24,000 evaluation samples."
    )

def test_agent_graph_skips_execution_without_selected_tool() -> None:
    client = FakeMCPToolClient()
    graph = build_agent_graph(client)

    initial_state: AgentState = {
        "user_request": "Tell me something unsupported.",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "tool_results": [],
        "selected_tool_arguments": None,
        "final_response": None,
        "selection_reason": None,
        "workflow_status": None,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert result["selected_tool"] is None
    assert result["tool_results"] == []
    assert result["final_response"] == (
        "I could not select an available tool "
        "for that request."
    )

    assert result["selection_reason"] == "unsupported_request"

    assert client.list_tools_call_count == 1
    assert client.called_tool_name is None

def test_agent_graph_reports_unavailable_capability() -> None:
    client = FakeMCPToolClient(
        include_evaluation_summary=False,
    )

    graph = build_agent_graph(client)

    initial_state: AgentState = {
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

    result = asyncio.run(graph.ainvoke(initial_state))

    assert client.called_tool_name is None
    assert result["selected_tool"] is None
    assert result["selection_reason"] == "capability_unavailable"
    assert result["tool_results"] == []
    assert result["final_response"] == (
        "I understood the request, but the required "
        "capability is not currently available."
    )

def test_agent_graph_executes_worst_digit_multi_tool_workflow() -> None:
    client = FakeMCPToolClient(
        include_multi_tool_capabilities=True,
    )
    graph = build_agent_graph(client)

    initial_state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert client.call_tool_calls == [
        ("get_evaluation_insights", None),
        ("get_digit_metrics", {"digit": 3}),
    ]

    assert len(result["tool_results"]) == 2
    assert result["tool_results"][0].tool_name == (
        "get_evaluation_insights"
    )
    assert result["tool_results"][1].tool_name == (
        "get_digit_metrics"
    )

    assert result["selected_tool"] is None
    assert result["selected_tool_arguments"] is None
    assert result["selection_reason"] == "worst_digit_request"
    assert result["workflow_status"] is None

    assert result["final_response"] == (
        "The model struggles most with digit 3. "
        "For digit 3, precision is 95.00%, "
        "recall is 94.00%, and F1 score is 94.50%."
    )

def test_agent_graph_stops_multi_tool_workflow_after_insights_error() -> None:
    client = FakeMCPToolClient(
        include_multi_tool_capabilities=True,
        fail_evaluation_insights=True,
    )
    graph = build_agent_graph(client)

    initial_state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert client.call_tool_calls == [
        ("get_evaluation_insights", None),
    ]

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0].tool_name == (
        "get_evaluation_insights"
    )
    assert result["tool_results"][0].is_error is True

    assert result["selected_tool"] is None
    assert result["selected_tool_arguments"] is None

    assert result["final_response"] == (
        "Tool get_evaluation_insights failed: "
        "Evaluation insights unavailable."
    )

def test_agent_graph_stops_multi_tool_workflow_after_digit_metrics_error() -> None:
    client = FakeMCPToolClient(
        include_multi_tool_capabilities=True,
        fail_digit_metrics=True,
    )
    graph = build_agent_graph(client)

    initial_state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert client.call_tool_calls == [
        ("get_evaluation_insights", None),
        ("get_digit_metrics", {"digit": 3}),
    ]

    assert len(result["tool_results"]) == 2

    assert result["tool_results"][0].tool_name == (
        "get_evaluation_insights"
    )
    assert result["tool_results"][0].is_error is False

    assert result["tool_results"][1].tool_name == (
        "get_digit_metrics"
    )
    assert result["tool_results"][1].is_error is True
    assert result["tool_results"][1].error_message == (
        "Digit metrics unavailable."
    )

    assert result["selected_tool"] is None
    assert result["selected_tool_arguments"] is None

    assert result["final_response"] == (
        "Tool get_digit_metrics failed: "
        "Digit metrics unavailable."
    )

def test_agent_graph_reports_unavailable_follow_up_capability() -> None:
    client = FakeMCPToolClient(
        include_multi_tool_capabilities=True,
        include_digit_metrics=False,
    )
    graph = build_agent_graph(client)

    initial_state: AgentState = {
        "user_request": (
            "Which digit does the model struggle with most, "
            "and how well does it recognize that digit?"
        ),
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": None,
        "tool_results": [],
        "final_response": None,
        "workflow_status": None,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert client.call_tool_calls == [
        ("get_evaluation_insights", None),
    ]

    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0].tool_name == (
        "get_evaluation_insights"
    )
    assert result["tool_results"][0].is_error is False

    assert result["selected_tool"] is None
    assert result["selected_tool_arguments"] is None

    assert result["final_response"] == (
        "I identified digit 3 as the model's "
        "worst-performing digit, but the required "
        "digit-metrics capability is not currently available."
    )

def test_agent_graph_handles_unsupported_request_without_tool_call() -> None:
    client = FakeMCPToolClient()
    graph = build_agent_graph(client)

    initial_state: AgentState = {
        "user_request": "What color is the sky?",
        "image_bytes": None,
        "available_tools": [],
        "selected_tool": None,
        "selected_tool_arguments": None,
        "selection_reason": None,
        "workflow_status": None,
        "tool_results": [],
        "final_response": None,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert client.call_tool_calls == []
    assert result["selected_tool"] is None
    assert result["selection_reason"] == "unsupported_request"
    assert result["tool_results"] == []
    assert result["final_response"] == (
        "I could not select an available tool "
        "for that request."
    )
