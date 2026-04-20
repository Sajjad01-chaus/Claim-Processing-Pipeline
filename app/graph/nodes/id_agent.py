"""
ID Agent Node

Processes ONLY the pages classified as `identity_document` by the Segregator.
Extracts: patient name, date of birth, ID number, gender, blood group, address,
          policy number, contact number, and any other identity fields present.

Returns id_data dict in the state.
"""

import json
import logging
import os
from typing import Any

from groq import Groq

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a medical claims processor specialising in identity documents.
You will be shown one or more identity document pages (government ID cards,
insurance cards, patient registration details, etc.).

Extract ALL available identity and policy information.

Return ONLY valid JSON with this structure (use null for missing fields):
{
  "patient_name": "...",
  "date_of_birth": "...",
  "gender": "...",
  "blood_group": "...",
  "id_number": "...",
  "id_type": "...",
  "address": "...",
  "contact_number": "...",
  "email": "...",
  "policy_number": "...",
  "insurance_provider": "...",
  "id_issue_date": "...",
  "id_expiry_date": "...",
  "additional_fields": {}
}
"""


def id_agent_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: extracts identity information from ID document pages.
    Only processes pages routed by the Segregator (state["id_pages"]).
    """
    id_pages = state.get("id_pages", [])

    if not id_pages:
        logger.warning("[ID Agent] No identity document pages found.")
        return {
            "id_data": {"status": "no_identity_pages_found"},
            "errors": [],
        }

    logger.info("[ID Agent] Processing %d identity page(s).", len(id_pages))
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Build content — one image per page
    content: list[dict] = []
    for i, b64_img in enumerate(id_pages):
        content.append({"type": "text", "text": f"=== Identity Document Page {i + 1} ==="})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_img}"}
        })
    content.append({
        "type": "text",
        "text": "Extract all identity information from the pages above. Return ONLY the JSON."
    })

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=1024,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        id_data = json.loads(raw)
        logger.info("[ID Agent] Extraction successful.")
        return {"id_data": id_data, "errors": []}

    except Exception as exc:
        logger.error("[ID Agent] Extraction failed: %s", exc)
        return {
            "id_data": {"error": str(exc), "status": "extraction_failed"},
            "errors": [f"ID Agent error: {exc}"],
        }
