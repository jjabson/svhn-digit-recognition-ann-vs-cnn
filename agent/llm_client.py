from dataclasses import dataclass
from typing import Any, Literal, Protocol
from openai import AsyncOpenAI
import json

from agent.mcp_client import MCPToolDefinition, MCPToolResult


@dataclass(frozen=True)
class AgentReasoningDecision:
    """Structured decision produced by the agent reasoning layer."""

    action: Literal["tool", "final"]
    tool_name: str | None = None
    tool_arguments: dict[str, Any] | None = None
    final_response: str | None = None


class AgentReasoningClient(Protocol):
    """Reasoning capability used by the agent graph."""

    async def decide(
        self,
        user_request: str,
        available_tools: list[MCPToolDefinition],
        tool_results: list[MCPToolResult],
        image_available: bool,
    ) -> AgentReasoningDecision:
        """Return the agent's next reasoning decision."""
        ...

_AGENT_INSTRUCTIONS = """
You are an ML System Analyst Agent.

Use the available application tools when they are needed to answer the
user's request.

Application tool results are authoritative. Do not redefine inference
status, confidence policy, fallback behavior, review requirements,
evaluation metrics, or other application-owned policy.

If the available information is sufficient, answer the user directly.
""".strip()
class OpenAIReasoningClient:
    """OpenAI-backed implementation of the agent reasoning contract."""

    def __init__(
        self,
        model: str,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self._model = model
        self._client = client or AsyncOpenAI()

    @staticmethod
    def _to_openai_tool(tool: MCPToolDefinition) -> dict[str, Any]:
        """Translate a discovered MCP tool into an OpenAI function tool."""

        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.input_schema,
        }

    async def decide(
            self,
            user_request: str,
            available_tools: list[MCPToolDefinition],
            tool_results: list[MCPToolResult],
            image_available: bool,
    ) -> AgentReasoningDecision:
        """Ask OpenAI for the agent's next reasoning decision."""

        tools = [
            self._to_openai_tool(tool)
            for tool in available_tools
        ]

        input_text = user_request

        if image_available:
            input_text += (
                "\n\nAn image is available for image-based tools."
            )

        if tool_results:
            input_text += (
                    "\n\nPrevious MCP tool observations:\n"
                    + self._format_tool_results(tool_results)
            )

        response = await self._client.responses.create(
            model=self._model,
            instructions=_AGENT_INSTRUCTIONS,
            input=input_text,
            tools=tools,
        )

        for item in response.output:
            if item.type == "function_call":
                return AgentReasoningDecision(
                    action="tool",
                    tool_name=item.name,
                    tool_arguments=json.loads(item.arguments),
                )

        if response.output_text:
            return AgentReasoningDecision(
                action="final",
                final_response=response.output_text,
            )

        raise ValueError(
            "OpenAI response contained neither a supported tool call "
            "nor final text."
        )

    @staticmethod
    def _format_tool_results(
            tool_results: list[MCPToolResult],
    ) -> str:
        """Format prior MCP observations for LLM reasoning."""

        observations = []

        for result in tool_results:
            observations.append(
                {
                    "tool_name": result.tool_name,
                    "structured_content": result.structured_content,
                    "is_error": result.is_error,
                    "error_message": result.error_message,
                }
            )

        return json.dumps(observations)