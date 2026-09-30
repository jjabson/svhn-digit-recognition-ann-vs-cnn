"""LangGraph construction for the ML system agent."""

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    ToolDiscoveryClient,
    ToolExecutionClient,
    discover_tools_node,
    execute_tool_node,
    route_after_tool_selection,
    select_tool_node,
    synthesize_response_node,
    select_follow_up_tool_node,
    llm_reasoning_node,
    route_after_llm_reasoning,
    route_after_llm_tool_execution,
)
from agent.state import AgentState
from typing import Protocol
from agent.llm_client import AgentReasoningClient

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
    graph.add_node(
        "select_follow_up_tool",
        select_follow_up_tool_node,
    )
    graph.add_conditional_edges(
        "select_follow_up_tool",
        route_after_tool_selection,
        {
            "execute_tool": "execute_tool",
            "synthesize_response": "synthesize_response",
        },
    )

    graph.add_edge(START, "discover_tools")
    graph.add_edge("discover_tools", "select_tool")
    graph.add_conditional_edges(
        "select_tool",
        route_after_tool_selection,
        {
            "execute_tool": "execute_tool",
            "synthesize_response": "synthesize_response",
        },
    )
    graph.add_edge(
        "execute_tool",
        "select_follow_up_tool",
    )
    graph.add_edge("synthesize_response", END)

    return graph.compile()

def build_llm_agent_graph(
    client: AgentMCPClient,
    reasoning_client: AgentReasoningClient,
):
    """Build the LLM-driven agent workflow."""

    async def discover_tools(state):
        return await discover_tools_node(state, client)

    async def reason(state):
        return await llm_reasoning_node(
            state,
            reasoning_client,
        )

    async def execute_tool(state):
        return await execute_tool_node(state, client)

    graph = StateGraph(AgentState)

    graph.add_node("discover_tools", discover_tools)
    graph.add_node("reason", reason)
    graph.add_node("execute_tool", execute_tool)

    graph.add_edge(START, "discover_tools")
    graph.add_edge("discover_tools", "reason")

    graph.add_conditional_edges(
        "reason",
        route_after_llm_reasoning,
        {
            "execute_tool": "execute_tool",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "execute_tool",
        route_after_llm_tool_execution,
        {
            "reason": "reason",
            "end": END,
        },
    )

    return graph.compile()

