"""
Discharge Summary Agent Node

Processes ONLY the pages classified as `discharge_summary` by the Segregator.
Extracts: admission date, discharge date, length of stay, primary diagnosis,
          attending physician, hospital course summary, discharge medications,
          follow-up instructions, and condition at discharge.

Returns discharge_data dict in the state.
"""

import json
import logging
import os
from typing import Any

from groq import Groq

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a medical claims processor specialising in hospital discharge summaries.
You will be shown one or more discharge summary pages from a hospital.

Extract ALL clinical and administrative information present.

Return ONLY valid JSON with this structure (use null for missing fields):
{
  "patient_name": "...",
  "mrn": "...",
  "date_of_birth": "...",
  "admission_date": "...",
  "discharge_date": "...",
  "length_of_stay_days": null,
  "hospital_name": "...",
  "attending_physician": "...",
  "admission_diagnosis": "...",
  "final_diagnosis": "...",
  "icd_codes": [],
  "hospital_course_summary": "...",
  "procedures_performed": [],
  "condition_at_discharge": "...",
  "discharge_medications": [
    {"name": "...", "dose": "...", "frequency": "...", "duration": "..."}
  ],
  "follow_up_instructions": "...",
  "activity_restrictions": "...",
  "diet_instructions": "...",
  "signed_by": "...",
  "signed_date": "..."
}
"""


def discharge_agent_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: extracts clinical data from discharge summary pages.
    Only processes pages routed by the Segregator (state["discharge_pages"]).
    """
    discharge_pages = state.get("discharge_pages", [])

    if not discharge_pages:
        logger.warning("[Discharge Agent] No discharge summary pages found.")
        return {
            "discharge_data": {"status": "no_discharge_pages_found"},
            "errors": [],
        }

    logger.info("[Discharge Agent] Processing %d discharge page(s).", len(discharge_pages))
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    content: list[dict] = []
    for i, b64_img in enumerate(discharge_pages):
        content.append({"type": "text", "text": f"=== Discharge Summary Page {i + 1} ==="})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_img}"}
        })
    content.append({
        "type": "text",
        "text": "Extract all discharge summary information from the pages above. Return ONLY the JSON."
    })

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        discharge_data = json.loads(raw)
        logger.info("[Discharge Agent] Extraction successful.")
        return {"discharge_data": discharge_data, "errors": []}

    except Exception as exc:
        logger.error("[Discharge Agent] Extraction failed: %s", exc)
        return {
            "discharge_data": {"error": str(exc), "status": "extraction_failed"},
            "errors": [f"Discharge Agent error: {exc}"],
        }
