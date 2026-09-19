"""MCP client boundary for agent-accessible application capabilities."""

import base64
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession


@dataclass(frozen=True)
class MCPToolDefinition:
    """Agent-neutral description of a discovered MCP tool."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class MCPToolResult:
    """Agent-neutral result of an MCP tool invocation."""

    tool_name: str
    structured_content: dict[str, Any] | None
    is_error: bool
    error_message: str | None


class MCPToolClient:
    """Thin adapter between the MCP SDK and agent-facing capabilities."""

    def __init__(self, session: ClientSession) -> None:
        self._session = session

    async def list_tools(self) -> list[MCPToolDefinition]:
        """Discover MCP tools and convert them to agent-neutral definitions."""
        result = await self._session.list_tools()

        return [
            MCPToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.input_schema,
            )
            for tool in result.tools
        ]

    async def call_tool(
            self,
            name: str,
            arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Invoke an MCP tool and convert its result to an agent-neutral result."""
        result = await self._session.call_tool(
            name,
            arguments=arguments or {},
        )

        error_message = None

        if result.is_error and result.content:
            first_content = result.content[0]
            error_message = getattr(first_content, "text", None)

        return MCPToolResult(
            tool_name=name,
            structured_content=result.structured_content,
            is_error=result.is_error,
            error_message=error_message,
        )

    @staticmethod
    def encode_image(image_bytes: bytes) -> str:
        """Encode image bytes for MCP tool transport."""
        return base64.b64encode(image_bytes).decode("ascii")