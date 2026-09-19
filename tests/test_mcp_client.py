import asyncio
from types import SimpleNamespace

from agent.mcp_client import MCPToolClient

def test_encode_image() -> None:
    image_bytes = b"hello"

    encoded = MCPToolClient.encode_image(image_bytes)

    assert encoded == "aGVsbG8="


def test_list_tools_converts_discovered_tools() -> None:
    discovered_tool = SimpleNamespace(
        name="example_tool",
        description="Example MCP capability",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "integer"},
            },
        },
    )

    session = SimpleNamespace()

    async def list_tools():
        return SimpleNamespace(tools=[discovered_tool])

    session.list_tools = list_tools

    client = MCPToolClient(session)

    tools = asyncio.run(client.list_tools())

    assert len(tools) == 1
    assert tools[0].name == "example_tool"
    assert tools[0].description == "Example MCP capability"
    assert tools[0].input_schema == {
        "type": "object",
        "properties": {
            "value": {"type": "integer"},
        },
    }


def test_call_tool_converts_successful_result() -> None:
    session = SimpleNamespace()

    async def call_tool(name, arguments):
        return SimpleNamespace(
            structured_content={
                "model_name": "cnn",
                "confidence_threshold": 0.9,
            },
            is_error=False,
            content=[],
        )

    session.call_tool = call_tool

    client = MCPToolClient(session)

    result = asyncio.run(
        client.call_tool("get_inference_config")
    )

    assert result.tool_name == "get_inference_config"
    assert result.structured_content == {
        "model_name": "cnn",
        "confidence_threshold": 0.9,
    }
    assert result.is_error is False
    assert result.error_message is None


def test_call_tool_converts_error_result() -> None:
    session = SimpleNamespace()

    async def call_tool(name, arguments):
        return SimpleNamespace(
            structured_content=None,
            is_error=True,
            content=[
                SimpleNamespace(
                    text=(
                        "Error executing tool get_digit_metrics: "
                        "No evaluation metrics found for digit 12."
                    )
                )
            ],
        )

    session.call_tool = call_tool

    client = MCPToolClient(session)

    result = asyncio.run(
        client.call_tool(
            "get_digit_metrics",
            {"digit": 12},
        )
    )

    assert result.tool_name == "get_digit_metrics"
    assert result.structured_content is None
    assert result.is_error is True
    assert result.error_message == (
        "Error executing tool get_digit_metrics: "
        "No evaluation metrics found for digit 12."
    )