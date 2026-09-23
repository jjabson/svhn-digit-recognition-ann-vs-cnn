from agent.mcp_client import MCPToolDefinition, MCPToolResult
from agent.state import AgentState


def test_agent_state_holds_agent_execution_context() -> None:
    tool = MCPToolDefinition(
        name="get_evaluation_summary",
        description="Return evaluation summary",
        input_schema={"type": "object"},
    )

    result = MCPToolResult(
        tool_name="get_evaluation_summary",
        structured_content={"accuracy": 0.951375},
        is_error=False,
        error_message=None,
    )

    state: AgentState = {
        "user_request": "How accurate is the model?",
        "image_bytes": None,
        "available_tools": [tool],
        "tool_results": [result],
        "final_response": None,
        "selected_tool": None,
        "selection_reason": None,
    }

    assert state["user_request"] == "How accurate is the model?"
    assert state["image_bytes"] is None
    assert state["available_tools"] == [tool]
    assert state["tool_results"] == [result]
    assert state["final_response"] is None