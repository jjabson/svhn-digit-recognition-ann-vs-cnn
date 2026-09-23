"""LangGraph nodes for agent execution."""

from typing import Protocol

from agent.mcp_client import MCPToolDefinition, MCPToolResult
from agent.state import AgentState


class ToolDiscoveryClient(Protocol):
    """Capability required by nodes that discover MCP tools."""

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Return MCP tools available to the agent."""
        ...

class ToolExecutionClient(Protocol):
    """Capability required by nodes that execute MCP tools."""

    async def call_tool(
        self,
        name: str,
        arguments: dict | None = None,
    ) -> MCPToolResult:
        """Invoke an MCP tool and return its normalized result."""
        ...

async def discover_tools_node(
    state: AgentState,
    client: ToolDiscoveryClient,
) -> dict:
    """Discover MCP capabilities available to the agent."""
    tools = await client.list_tools()

    return {
        "available_tools": tools,
    }

def select_tool_node(state: AgentState) -> dict:
    """Select an MCP capability for a supported deterministic request."""

    request = state["user_request"].lower()
    available_tool_names = {
        tool.name
        for tool in state["available_tools"]
    }

    selected_tool = None
    selection_reason = "unsupported_request"

    is_accuracy_request = (
        "accurate" in request
        or "accuracy" in request
    )

    if is_accuracy_request:
        if "get_evaluation_summary" in available_tool_names:
            selected_tool = "get_evaluation_summary"
            selection_reason = "accuracy_request"
        else:
            selection_reason = "capability_unavailable"

    return {
        "selected_tool": selected_tool,
        "selection_reason": selection_reason,
    }

def route_after_tool_selection(state: AgentState) -> str:
    """Route based on whether an MCP tool was selected."""

    if state["selected_tool"] is None:
        return "synthesize_response"

    return "execute_tool"

async def execute_tool_node(
    state: AgentState,
    client: ToolExecutionClient,
) -> dict:
    """Execute the selected MCP tool and accumulate its result."""

    selected_tool = state["selected_tool"]

    if selected_tool is None:
        return {
            "tool_results": state["tool_results"],
        }

    result = await client.call_tool(selected_tool)

    return {
        "tool_results": [
            *state["tool_results"],
            result,
        ],
    }

def synthesize_response_node(state: AgentState) -> dict:
    """Create a deterministic response from an MCP tool result."""

    if state["selection_reason"] == "unsupported_request":
        return {
            "final_response": (
                "I could not select an available tool "
                "for that request."
            ),
        }

    if state["selection_reason"] == "capability_unavailable":
        return {
            "final_response": (
                "I understood the request, but the required "
                "capability is not currently available."
            ),
        }

    if not state["tool_results"]:
        return {
            "final_response": "No tool result is available.",
        }

    result = state["tool_results"][-1]

    if result.is_error:
        return {
            "final_response": (
                f"Tool {result.tool_name} failed: "
                f"{result.error_message}"
            ),
        }

    if (
            result.tool_name == "predict_digit"
            and result.structured_content is not None
    ):
        status = result.structured_content["status"]
        review_required = result.structured_content["review_required"]

        review_message = (
            "Review is required."
            if review_required
            else "Review is not required."
        )

        return {
            "final_response": (
                f"The inference decision reported status {status}. "
                f"{review_message}"
            ),
        }

    if (
        result.tool_name == "get_evaluation_summary"
        and not result.is_error
        and result.structured_content is not None
    ):
        accuracy = result.structured_content["accuracy"]
        evaluation_samples = result.structured_content[
            "evaluation_samples"
        ]

        return {
            "final_response": (
                f"The model achieved {accuracy:.2%} accuracy "
                f"on {evaluation_samples:,} evaluation samples."
            ),
        }

    return {
        "final_response": "Unable to synthesize the tool result.",
    }