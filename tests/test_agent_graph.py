import asyncio

from agent.graph import build_agent_graph
from agent.mcp_client import MCPToolDefinition, MCPToolResult
from agent.state import AgentState


class FakeMCPToolClient:
    def __init__(self) -> None:
        self.list_tools_call_count = 0
        self.called_tool_name: str | None = None

    async def call_tool(
            self,
            name: str,
            arguments: dict | None = None,
    ) -> MCPToolResult:
        self.called_tool_name = name

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

def test_agent_graph_discovers_tools() -> None:
    client = FakeMCPToolClient()
    graph = build_agent_graph(client)

    initial_state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [],
        "tool_results": [],
        "final_response": None,
        "selected_tool": None,
    }

    result = asyncio.run(
        graph.ainvoke(initial_state)
    )

    assert result["selected_tool"] == "get_evaluation_summary"

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