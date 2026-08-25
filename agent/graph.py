"""
LangGraph graph definition for the Learning Path Recommender.

Graph topology:
  intake → [rag + lms_tool] (parallel) → composition
  
On refinement turns: intake → composition (skip rag/lms re-fetch)
The conditional edge checks is_refinement to route correctly.
"""

import os
from langgraph.graph import StateGraph, END
from langsmith import traceable
from langgraph.prebuilt import tools_condition

from agent.state import AgentState
from agent.intake_node import intake_node
from agent.rag_node import rag_node
from agent.lms_tool_node import lms_tool_node
from agent.composition_node import composition_node

USE_REAL_TOOL_CALLING = os.getenv("USE_REAL_TOOL_CALLING", "false").lower() == "true"


def should_fetch_or_compose(state: dict) -> str:
    """
    Routing function after intake node.
    - If this is a refinement turn AND we already have RAG/LMS context,
      go directly to composition (the key architectural requirement).
    - Otherwise, fetch fresh RAG and LMS data.
    """
    is_refinement = state.get("is_refinement", False)
    has_rag = bool(state.get("rag_context"))
    has_lms = state.get("lms_completed") is not None

    if is_refinement and has_rag:
        return "composition"
    else:
        return "fetch_data"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent."""

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("intake", intake_node)
    graph.add_node("rag", rag_node)
    
    if USE_REAL_TOOL_CALLING:
        from agent.lms_agent_node import lms_agent_node, tool_node as lms_tools_node, lms_tool_extractor_node
        graph.add_node("lms_agent", lms_agent_node)
        graph.add_node("lms_tools", lms_tools_node)
        graph.add_node("lms_extractor", lms_tool_extractor_node)
    else:
        graph.add_node("lms_tool", lms_tool_node)
        
    graph.add_node("composition", composition_node)

    # Set entry point
    graph.set_entry_point("intake")

    # Conditional edge from intake: either fetch data or go straight to composition
    graph.add_conditional_edges(
        "intake",
        should_fetch_or_compose,
        {
            "fetch_data": "rag",
            "composition": "composition",
        },
    )

    if USE_REAL_TOOL_CALLING:
        # Flow with real tool calling:
        # rag -> lms_agent
        # lms_agent -> conditional: tools -> lms_tools OR __end__ -> composition
        # lms_tools -> lms_extractor -> composition
        graph.add_edge("rag", "lms_agent")
        graph.add_conditional_edges(
            "lms_agent",
            tools_condition,
            {
                "tools": "lms_tools",
                "__end__": "composition",
            }
        )
        graph.add_edge("lms_tools", "lms_extractor")
        graph.add_edge("lms_extractor", "composition")
    else:
        # Sequential flow: rag → lms_tool → composition
        graph.add_edge("rag", "lms_tool")
        graph.add_edge("lms_tool", "composition")

    # Composition always ends the turn
    graph.add_edge("composition", END)

    return graph.compile()


# Pre-build the graph
agent_graph = build_graph()


@traceable(name="run_agent", run_type="chain")
def run_agent(state: dict) -> dict:
    """Run the agent graph with the given state."""
    result = agent_graph.invoke(state)
    return result
