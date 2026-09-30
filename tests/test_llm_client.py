from agent.llm_client import (
    _AGENT_INSTRUCTIONS,
    AgentReasoningDecision,
    OpenAIReasoningClient,
)
from agent.mcp_client import MCPToolDefinition, MCPToolResult
from types import SimpleNamespace
import asyncio
import json


def test_tool_reasoning_decision_preserves_tool_request():
    decision = AgentReasoningDecision(
        action="tool",
        tool_name="get_digit_metrics",
        tool_arguments={"digit": 3},
    )

    assert decision.action == "tool"
    assert decision.tool_name == "get_digit_metrics"
    assert decision.tool_arguments == {"digit": 3}
    assert decision.final_response is None


def test_final_reasoning_decision_preserves_response():
    decision = AgentReasoningDecision(
        action="final",
        final_response="The model achieved 95.14% accuracy.",
    )

    assert decision.action == "final"
    assert decision.tool_name is None
    assert decision.tool_arguments is None
    assert decision.final_response == "The model achieved 95.14% accuracy."

def test_openai_tool_translation_preserves_mcp_contract():
    tool = MCPToolDefinition(
        name="get_digit_metrics",
        description="Return evaluation metrics for one digit.",
        input_schema={
            "type": "object",
            "properties": {
                "digit": {"type": "integer"},
            },
            "required": ["digit"],
        },
    )

    translated = OpenAIReasoningClient._to_openai_tool(tool)

    assert translated == {
        "type": "function",
        "name": "get_digit_metrics",
        "description": "Return evaluation metrics for one digit.",
        "parameters": {
            "type": "object",
            "properties": {
                "digit": {"type": "integer"},
            },
            "required": ["digit"],
        },
    }

def test_openai_decide_returns_final_response():
    captured_request = {}

    async def fake_create(**kwargs):
        captured_request.update(kwargs)

        return SimpleNamespace(
            output=[],
            output_text="The model achieved 95.14% accuracy.",
        )

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=fake_create,
        )
    )

    reasoning_client = OpenAIReasoningClient(
        model="test-model",
        client=fake_client,
    )

    tool = MCPToolDefinition(
        name="get_evaluation_summary",
        description="Return authoritative evaluation statistics.",
        input_schema={
            "type": "object",
            "properties": {},
        },
    )

    decision = asyncio.run(
        reasoning_client.decide(
            user_request="How accurate is the model?",
            available_tools=[tool],
            tool_results=[],
            image_available=False,
        )
    )

    assert decision == AgentReasoningDecision(
        action="final",
        final_response="The model achieved 95.14% accuracy.",
    )

    assert captured_request["model"] == "test-model"
    assert captured_request["input"] == "How accurate is the model?"
    assert captured_request["instructions"] == _AGENT_INSTRUCTIONS
    assert captured_request["tools"] == [
        {
            "type": "function",
            "name": "get_evaluation_summary",
            "description": "Return authoritative evaluation statistics.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }
    ]

def test_openai_decide_returns_tool_request():
    captured_request = {}

    async def fake_create(**kwargs):
        captured_request.update(kwargs)

        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="function_call",
                    name="get_digit_metrics",
                    arguments='{"digit": 3}',
                )
            ],
            output_text="",
        )

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=fake_create,
        )
    )

    reasoning_client = OpenAIReasoningClient(
        model="test-model",
        client=fake_client,
    )

    tool = MCPToolDefinition(
        name="get_digit_metrics",
        description="Return evaluation metrics for one digit.",
        input_schema={
            "type": "object",
            "properties": {
                "digit": {"type": "integer"},
            },
            "required": ["digit"],
        },
    )

    decision = asyncio.run(
        reasoning_client.decide(
            user_request="How well does the model recognize digit 3?",
            available_tools=[tool],
            tool_results=[],
            image_available=False,
        )
    )

    assert decision == AgentReasoningDecision(
        action="tool",
        tool_name="get_digit_metrics",
        tool_arguments={"digit": 3},
    )

    assert captured_request["model"] == "test-model"
    assert captured_request["input"] == (
        "How well does the model recognize digit 3?"
    )

def test_format_tool_results_preserves_mcp_observation():
    result = MCPToolResult(
        tool_name="get_evaluation_insights",
        structured_content={
            "worst_performing_digit": 3,
        },
        is_error=False,
        error_message=None,
    )

    formatted = OpenAIReasoningClient._format_tool_results([result])

    assert json.loads(formatted) == [
        {
            "tool_name": "get_evaluation_insights",
            "structured_content": {
                "worst_performing_digit": 3,
            },
            "is_error": False,
            "error_message": None,
        }
    ]

def test_openai_decide_includes_prior_tool_observations():
    captured_request = {}

    async def fake_create(**kwargs):
        captured_request.update(kwargs)

        return SimpleNamespace(
            output=[],
            output_text="Digit 3 is the worst-performing digit.",
        )

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            create=fake_create,
        )
    )

    reasoning_client = OpenAIReasoningClient(
        model="test-model",
        client=fake_client,
    )

    tool_result = MCPToolResult(
        tool_name="get_evaluation_insights",
        structured_content={
            "worst_performing_digit": 3,
        },
        is_error=False,
        error_message=None,
    )

    decision = asyncio.run(
        reasoning_client.decide(
            user_request=(
                "Which digit performs worst and how well does it perform?"
            ),
            available_tools=[],
            tool_results=[tool_result],
            image_available=False,
        )
    )

    assert decision == AgentReasoningDecision(
        action="final",
        final_response="Digit 3 is the worst-performing digit.",
    )

    assert (
        "Which digit performs worst and how well does it perform?"
        in captured_request["input"]
    )
    assert "Previous MCP tool observations:" in captured_request["input"]
    assert '"tool_name": "get_evaluation_insights"' in captured_request["input"]
    assert '"worst_performing_digit": 3' in captured_request["input"]

def test_openai_reasoning_client_includes_image_availability():
    import asyncio

    class FakeResponses:
        def __init__(self):
            self.kwargs = None

        async def create(self, **kwargs):
            self.kwargs = kwargs

            class Response:
                output = []
                output_text = "Prediction request understood."

            return Response()

    class FakeOpenAIClient:
        def __init__(self):
            self.responses = FakeResponses()

    fake_client = FakeOpenAIClient()

    reasoning_client = OpenAIReasoningClient(
        model="test-model",
        client=fake_client,
    )

    asyncio.run(
        reasoning_client.decide(
            user_request="Predict this digit.",
            available_tools=[],
            tool_results=[],
            image_available=True,
        )
    )

    assert (
        "An image is available"
        in fake_client.responses.kwargs["input"]
    )