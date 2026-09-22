"""State contract for LangGraph agent execution."""

from typing import TypedDict

from agent.mcp_client import MCPToolDefinition, MCPToolResult


class AgentState(TypedDict):
    """Shared state passed between agent graph nodes."""

    user_request: str
    image_bytes: bytes | None
    available_tools: list[MCPToolDefinition]
    tool_results: list[MCPToolResult]
    final_response: str | None
    selected_tool: str | None