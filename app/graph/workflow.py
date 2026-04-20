

from langgraph.graph import StateGraph, START, END

from app.graph.state import ClaimState
from app.graph.nodes.segregator import segregator_node
from app.graph.nodes.id_agent import id_agent_node
from app.graph.nodes.discharge_agent import discharge_agent_node
from app.graph.nodes.bill_agent import bill_agent_node
from app.graph.nodes.aggregator import aggregator_node


def build_workflow() -> StateGraph:
    builder = StateGraph(ClaimState)

    # Nodes
    builder.add_node("segregator", segregator_node)
    builder.add_node("id_agent", id_agent_node)
    builder.add_node("discharge_agent", discharge_agent_node)
    builder.add_node("bill_agent", bill_agent_node)
    builder.add_node("aggregator", aggregator_node)

    # Flow
    builder.add_edge(START, "segregator")

    builder.add_edge("segregator", "id_agent")
    builder.add_edge("id_agent", "discharge_agent")
    builder.add_edge("discharge_agent", "bill_agent")
    builder.add_edge("bill_agent", "aggregator")

    builder.add_edge("aggregator", END)

    return builder.compile()


# Module-level compiled graph — imported by main.py
graph = build_workflow()
