"""
LangGraph Workflow

Topology:

  START
    │
    ▼
  segregator          ← AI: classifies all pages into 9 doc types, routes to buckets
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
  id_agent        discharge_agent     bill_agent     ← all 3 run in PARALLEL
    │                  │                  │            each receives only its pages
    └──────────────────┴──────────────────┘
                        │
                        ▼
                    aggregator          ← pure Python: merges all outputs
                        │
                        ▼
                       END

Key LangGraph behaviour:
  • Adding edges from segregator → [id_agent, discharge_agent, bill_agent]
    causes LangGraph to execute all three agents in parallel (same superstep).
  • Because each agent writes to a DIFFERENT state key (id_data, discharge_data,
    bill_data), there is no state merge conflict.
  • aggregator runs only after ALL three agents have completed (LangGraph's
    built-in barrier / join behaviour for converging edges).
  • errors uses operator.add reducer so errors from all parallel branches
    accumulate correctly.
"""

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
