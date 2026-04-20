"""
Aggregator Node

Pure Python — no LLM call needed.
Merges the outputs of all three extraction agents into one clean final JSON.
Also builds a summary section with key claim metrics.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)


def aggregator_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: merges id_data, discharge_data, and bill_data into
    the final structured JSON output.
    """
    logger.info("[Aggregator] Merging agent outputs.")

    id_data = state.get("id_data", {})
    discharge_data = state.get("discharge_data", {})
    bill_data = state.get("bill_data", {})
    page_classification = state.get("page_classification", {})
    errors = state.get("errors", [])

    # ── Build document inventory from segregator results ──────────────────
    doc_inventory: dict[str, list[int]] = {}
    for page_idx, doc_type in page_classification.items():
        doc_inventory.setdefault(doc_type, []).append(page_idx + 1)  # 1-based for humans

    # ── Build claim summary ───────────────────────────────────────────────
    claim_summary: dict[str, Any] = {
        "patient_name": (
            id_data.get("patient_name")
            or discharge_data.get("patient_name")
            or _extract_from_bills(bill_data, "patient_name")
        ),
        "date_of_birth": (
            id_data.get("date_of_birth")
            or discharge_data.get("date_of_birth")
        ),
        "policy_number": id_data.get("policy_number"),
        "insurance_provider": id_data.get("insurance_provider"),
        "admission_date": discharge_data.get("admission_date"),
        "discharge_date": discharge_data.get("discharge_date"),
        "primary_diagnosis": (
            discharge_data.get("final_diagnosis")
            or discharge_data.get("admission_diagnosis")
        ),
        "attending_physician": discharge_data.get("attending_physician"),
        "hospital_name": (
            discharge_data.get("hospital_name")
            or _extract_from_bills(bill_data, "hospital_name")
        ),
        "total_billed_amount": bill_data.get("grand_total_all_bills"),
        "total_insurance_covered": bill_data.get("total_insurance_covered"),
        "total_patient_responsibility": bill_data.get("total_patient_responsibility"),
    }

    # ── Assemble final output ─────────────────────────────────────────────
    final_output: dict[str, Any] = {
        "claim_id": state["claim_id"],
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_pages": len(state.get("page_images", [])),
        "document_inventory": doc_inventory,
        "claim_summary": claim_summary,
        "extracted_data": {
            "identity": id_data,
            "discharge_summary": discharge_data,
            "billing": bill_data,
        },
        "processing_errors": errors,
        "status": "partial" if errors else "success",
    }

    logger.info(
        "[Aggregator] Done. Status: %s | Errors: %d",
        final_output["status"], len(errors),
    )
    return {"final_output": final_output, "errors": []}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_from_bills(bill_data: dict, field: str) -> Any:
    """Scan the first bill entry for a specific field."""
    for bill in bill_data.get("bills", []):
        value = bill.get(field)
        if value:
            return value
    return None
