import asyncio
import sys
import base64
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)

def create_server_params() -> StdioServerParameters:
    """
    Return the stdio configuration used to launch the MCP server
    for integration testing.
    """
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
    )

async def run_with_mcp_session(operation):
    """
    Run an integration-test operation using an initialized
    MCP client/server session.
    """
    server_params = create_server_params()

    async with stdio_client(server_params) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()
            return await operation(session)

def test_mcp_tool_discovery() -> None:
    """
    Verify that the MCP server exposes the expected tool capabilities.
    """

    async def discover_tools(session: ClientSession):
        result = await session.list_tools()
        return result.tools

    tools = asyncio.run(
        run_with_mcp_session(discover_tools)
    )

    tools_by_name = {
        tool.name: tool
        for tool in tools
    }

    assert set(tools_by_name) == {
        "get_inference_config",
        "get_evaluation_summary",
        "get_evaluation_insights",
        "get_digit_metrics",
        "get_model_summary",
        "predict_digit",
    }

    digit_metrics = tools_by_name["get_digit_metrics"]

    assert digit_metrics.input_schema["required"] == ["digit"]
    assert (
            digit_metrics.input_schema["properties"]["digit"]["type"]
            == "integer"
    )

    predict_digit = tools_by_name["predict_digit"]

    assert predict_digit.input_schema["required"] == ["image_data"]
    assert (
            predict_digit.input_schema["properties"]["image_data"]["type"]
            == "string"
    )

    prediction_properties = predict_digit.output_schema["properties"]

    assert set(prediction_properties) == {
        "selected_model",
        "predicted_digit",
        "confidence",
        "status",
        "decision_reason",
        "fallback_used",
        "fallback_reason",
        "review_required",
    }


def test_mcp_read_only_tools() -> None:
    """
    Verify read-only MCP inspection capabilities.
    """

    async def check_read_only_tools(
        session: ClientSession,
    ) -> None:
        result = await session.call_tool(
            "get_inference_config",
            arguments={},
        )

        assert result.is_error is False
        assert result.structured_content == {
            "model_name": "cnn",
            "confidence_threshold": 0.9,
        }

        evaluation_result = await session.call_tool(
            "get_evaluation_summary",
            arguments={},
        )

        assert evaluation_result.is_error is False
        assert evaluation_result.structured_content == {
            "accuracy": 0.951375,
            "macro_precision": 0.9516441770999601,
            "macro_recall": 0.9513749999999999,
            "macro_f1": 0.9513949437749426,
            "weighted_precision": 0.9516441770999603,
            "weighted_recall": 0.951375,
            "weighted_f1": 0.9513949437749426,
            "total_support": 24000,
            "correct_predictions": 22833,
            "incorrect_predictions": 1167,
            "is_independent_evaluation": True,
        }

        insights_result = await session.call_tool(
            "get_evaluation_insights",
            arguments={},
        )

        assert insights_result.is_error is False

        insights = insights_result.structured_content

        assert insights["best_class"]["class_label"] == 0
        assert insights["worst_class"]["class_label"] == 3
        assert insights["most_common_misclassification_true"] == 3
        assert insights["most_common_misclassification_predicted"] == 5
        assert insights["most_common_misclassification_count"] == 73

        digit_result = await session.call_tool(
            "get_digit_metrics",
            arguments={"digit": 3},
        )

        assert digit_result.is_error is False
        assert digit_result.structured_content == {
            "class_label": 3,
            "precision": 0.9615384615384616,
            "recall": 0.9166666666666666,
            "f1_score": 0.9385665529010239,
            "support": 2400,
        }

        model_result = await session.call_tool(
            "get_model_summary",
            arguments={},
        )

        assert model_result.is_error is False
        assert model_result.structured_content == {
            "model_name": "sequential",
            "input_shape": [None, 32, 32, 1],
            "output_shape": [None, 10],
            "total_parameters": 164362,
            "number_of_layers": 17,
        }

    asyncio.run(
        run_with_mcp_session(check_read_only_tools)
    )

def test_mcp_prediction() -> None:
    """
    Verify successful digit prediction through the MCP interface.
    """

    async def check_prediction(
        session: ClientSession,
    ) -> None:
        image_path = Path(
            "sample_images/digit_3_true3.png"
        )

        image_data = base64.b64encode(
            image_path.read_bytes()
        ).decode("ascii")

        result = await session.call_tool(
            "predict_digit",
            arguments={
                "image_data": image_data,
            },
        )

        assert result.is_error is False

        prediction = result.structured_content

        assert prediction["selected_model"] == "cnn"
        assert prediction["predicted_digit"] == 3
        assert prediction["confidence"] > 0.99
        assert prediction["status"] == "accepted"
        assert prediction["decision_reason"] is None
        assert prediction["fallback_used"] is False
        assert prediction["fallback_reason"] is None
        assert prediction["review_required"] is False

    asyncio.run(
        run_with_mcp_session(check_prediction)
    )

def test_mcp_tool_errors() -> None:
    """
    Verify caller/input errors are exposed as MCP tool errors.
    """

    async def check_tool_errors(
        session: ClientSession,
    ) -> None:
        # Unsupported digit
        invalid_digit_result = await session.call_tool(
            "get_digit_metrics",
            arguments={
                "digit": 12,
            },
        )

        assert invalid_digit_result.is_error is True
        assert invalid_digit_result.structured_content is None
        assert (
            "No evaluation metrics found for digit 12."
            in invalid_digit_result.content[0].text
        )

        # Malformed Base64
        invalid_base64_result = await session.call_tool(
            "predict_digit",
            arguments={
                "image_data": "this-is-not-valid-base64!!!",
            },
        )

        assert invalid_base64_result.is_error is True
        assert invalid_base64_result.structured_content is None
        assert (
            "image_data must be valid base64-encoded image data."
            in invalid_base64_result.content[0].text
        )

        # Valid Base64, but not an image
        invalid_image_data = base64.b64encode(
            b"this is not an image"
        ).decode("ascii")

        invalid_image_result = await session.call_tool(
            "predict_digit",
            arguments={
                "image_data": invalid_image_data,
            },
        )

        assert invalid_image_result.is_error is True
        assert invalid_image_result.structured_content is None
        assert (
            "The uploaded file is not a valid image."
            in invalid_image_result.content[0].text
        )

        # Empty decoded image
        empty_image_result = await session.call_tool(
            "predict_digit",
            arguments={
                "image_data": "",
            },
        )

        assert empty_image_result.is_error is True
        assert empty_image_result.structured_content is None
        assert (
            "image_data must not decode to an empty image."
            in empty_image_result.content[0].text
        )

    asyncio.run(
        run_with_mcp_session(check_tool_errors)
    )
