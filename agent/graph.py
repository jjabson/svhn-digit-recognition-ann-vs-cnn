"""LangGraph construction for the ML system agent."""

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    ToolDiscoveryClient,
    ToolExecutionClient,
    discover_tools_node,
    execute_tool_node,
    select_tool_node,
    synthesize_response_node,
)
from agent.state import AgentState
from typing import Protocol

class AgentMCPClient(
    ToolDiscoveryClient,
    ToolExecutionClient,
    Protocol,
):
    """MCP capabilities required by the current agent graph."""

    pass


def build_agent_graph(client: AgentMCPClient):
    """Build the agent graph with its MCP capability dependency."""

    async def execute_tool(state: AgentState) -> dict:
        return await execute_tool_node(state, client)

    async def discover_tools(state: AgentState) -> dict:
        return await discover_tools_node(state, client)

    graph = StateGraph(AgentState)

    graph.add_node("discover_tools", discover_tools)
    graph.add_node("select_tool", select_tool_node)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("synthesize_response", synthesize_response_node)

    graph.add_edge(START, "discover_tools")
    graph.add_edge("discover_tools", "select_tool")
    graph.add_edge("select_tool", "execute_tool")
    graph.add_edge("execute_tool", "synthesize_response")
    graph.add_edge("synthesize_response", END)

    return graph.compile()

