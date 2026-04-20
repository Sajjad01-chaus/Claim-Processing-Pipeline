"""
LangGraph State — the single shared object that flows through all nodes.
Each node reads what it needs and writes only its own output key.
Parallel branches (id_agent, discharge_agent, bill_agent) each write to a
different key, so there are no merge conflicts.
errors uses operator.add so errors from parallel branches accumulate.
"""

from typing import TypedDict, Dict, List, Any, Annotated
import operator


class ClaimState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    claim_id: str
    page_images: List[str]          # base64 PNG, index == page number (0-based)

    # ── Segregator output ────────────────────────────────────────────────────
    page_classification: Dict[int, str]   # {0: "claim_forms", 2: "identity_document", ...}
    id_pages: List[str]                   # base64 images sent to ID agent
    discharge_pages: List[str]            # base64 images sent to Discharge agent
    bill_pages: List[str]                 # base64 images sent to Bill agent

    # ── Extraction agent outputs ─────────────────────────────────────────────
    id_data: Dict[str, Any]
    discharge_data: Dict[str, Any]
    bill_data: Dict[str, Any]

    # ── Aggregator output ────────────────────────────────────────────────────
    final_output: Dict[str, Any]

    # ── Cross-cutting ────────────────────────────────────────────────────────
    errors: Annotated[List[str], operator.add]   # accumulates from all branches
