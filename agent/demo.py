"""Interactive demonstration host for the SVHN ML System Analyst Agent."""

from __future__ import annotations

import asyncio
import os
import sys
import argparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.graph import build_llm_agent_graph
from agent.llm_client import OpenAIReasoningClient
from agent.mcp_client import MCPToolClient


EXIT_COMMANDS = {"exit", "quit"}
DEFAULT_MODEL = "gpt-5.6-luna"


def print_banner() -> None:
    """Display the interactive agent banner."""
    print()
    print("SVHN ML System Analyst Agent")
    print("Type 'quit' or 'exit' to stop.")
    print()

def print_tool_trace(tool_results: list) -> None:
    """Display the MCP tools used during an agent request."""
    if not tool_results:
        print("[trace] No MCP tools executed.")
        return

    print("[trace] MCP tool execution:")

    for index, result in enumerate(tool_results, start=1):
        print(f"  {index}. {result.tool_name}")

        if result.structured_content is not None:
            print(
                f"     result: {result.structured_content}"
            )

        if result.is_error:
            print(
                f"     error: {result.error_message}"
            )

async def run_agent(trace: bool = False) -> None:
    """Run the interactive agent against the live MCP server."""
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY must be configured before starting the agent."
        )

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_server.server"],
    )

    reasoning_client = OpenAIReasoningClient(
        model=DEFAULT_MODEL,
    )

    print_banner()

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_client = MCPToolClient(session)

            graph = build_llm_agent_graph(
                client=mcp_client,
                reasoning_client=reasoning_client,
            )

            while True:
                try:
                    user_request = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    print("Goodbye.")
                    break

                if not user_request:
                    continue

                if user_request.lower() in EXIT_COMMANDS:
                    print("Goodbye.")
                    break

                initial_state = {
                    "user_request": user_request,
                    "image_bytes": None,
                    "available_tools": [],
                    "selected_tool": None,
                    "selected_tool_arguments": None,
                    "selection_reason": None,
                    "workflow_status": None,
                    "tool_results": [],
                    "final_response": None,
                }

                result = await graph.ainvoke(initial_state)

                if trace:
                    print()
                    print_tool_trace(result["tool_results"])

                print()
                print("Agent:")
                print(result["final_response"])
                print()


def main() -> None:
    """Start the interactive agent demonstration."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactive demonstration host for the "
            "SVHN ML System Analyst Agent."
        )
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Display MCP tool execution for each request.",
    )

    args = parser.parse_args()

    asyncio.run(
        run_agent(
            trace=args.trace,
        )
    )


if __name__ == "__main__":
    main()