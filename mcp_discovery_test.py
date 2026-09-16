import asyncio
import sys
import base64
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    stdio_client,
)


async def main() -> None:
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
    )

    async with stdio_client(server_params) as (
        read_stream,
        write_stream,
    ):
        async with ClientSession(
            read_stream,
            write_stream,
        ) as session:
            await session.initialize()

            result = await session.list_tools()

            print("Discovered MCP tools:")

            for tool in result.tools:
                print(f"- Name: {tool.name}")
                print(f"  Description: {tool.description}")
                print(f"  Input schema: {tool.input_schema}")
                print(f"  Output schema: {tool.output_schema}")

            print("\nCalling get_inference_config...")

            tool_result = await session.call_tool(
                "get_inference_config",
                arguments={},
            )

            print("Tool result:")
            print(tool_result)

            print("\nCalling get_evaluation_summary...")

            evaluation_result = await session.call_tool(
                "get_evaluation_summary",
                arguments={},
            )

            print("Evaluation summary result:")
            print(evaluation_result)

            print("\nCalling get_evaluation_insights...")

            insights_result = await session.call_tool(
                "get_evaluation_insights",
                arguments={},
            )

            print("Evaluation insights result:")
            print(insights_result)

            print("\nCalling get_digit_metrics for digit 3...")

            digit_metrics_result = await session.call_tool(
                "get_digit_metrics",
                arguments={"digit": 3},
            )

            print("Digit metrics result:")
            print(digit_metrics_result)

            print("\nCalling get_digit_metrics for invalid digit 12...")

            invalid_digit_result = await session.call_tool(
                "get_digit_metrics",
                arguments={"digit": 12},
            )

            print("Invalid digit metrics result:")
            print(invalid_digit_result)

            print("\nCalling get_model_summary...")

            result = await session.call_tool(
                "get_model_summary",
                arguments={},
            )

            print("Model summary result:")
            print(result)

            print("\nCalling predict_digit with digit_3_true3.png...")

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

            print("Prediction result:")
            print(result)

            print("\nCalling predict_digit with invalid base64...")

            result = await session.call_tool(
                "predict_digit",
                arguments={
                    "image_data": "this-is-not-valid-base64!!!",
                },
            )

            print("Invalid base64 result:")
            print(result)

            print("\nCalling predict_digit with valid base64 containing invalid image bytes...")

            invalid_image_data = base64.b64encode(
                b"this is not an image"
            ).decode("ascii")

            result = await session.call_tool(
                "predict_digit",
                arguments={
                    "image_data": invalid_image_data,
                },
            )

            print("Invalid image result:")
            print(result)

            print("\nCalling predict_digit with empty image data...")

            result = await session.call_tool(
                "predict_digit",
                arguments={
                    "image_data": "",
                },
            )

            print("Empty image result:")
            print(result)

if __name__ == "__main__":
    asyncio.run(main())