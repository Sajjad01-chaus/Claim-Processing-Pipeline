"""
Itemized Bill Agent Node

Processes ONLY the pages classified as `itemized_bill` by the Segregator.
This sample PDF has TWO bill pages (hospital bill + pharmacy bill), so both
get passed here and extracted together.

Extracts: every line item with date, description, quantity, rate, amount,
          subtotal, taxes, discounts, total amount, and insurance deduction.

Returns bill_data dict in the state.
"""
import json
import logging
import os
import time
from typing import Any

from groq import Groq
from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a medical claims processor specialising in itemized hospital and pharmacy bills.
You will be shown ONE bill page.

Extract ALL line items and totals from this page.

Return ONLY valid JSON with this structure:
{
  "bills": [
    {
      "bill_type": "...",
      "bill_number": "...",
      "bill_date": "...",
      "hospital_name": "...",
      "patient_name": "...",
      "patient_id": "...",
      "admission_date": "...",
      "discharge_date": "...",
      "line_items": [
        {
          "date": "...",
          "description": "...",
          "quantity": null,
          "unit_rate": null,
          "amount": null
        }
      ],
      "subtotal": null,
      "discount": null,
      "tax": null,
      "tax_percentage": null,
      "total_amount": null,
      "insurance_payment": null,
      "patient_responsibility": null,
      "payment_method": "..."
    }
  ]
}
"""


def _process_single_bill_page(client, b64_img: str, page_num: int) -> dict:
    """Process ONE bill page with retry + backoff"""

    content = [
        {"type": "text", "text": f"=== Bill Page {page_num} ==="},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_img}"}
        },
        {
            "type": "text",
            "text": "Extract all billing details from this page. Return ONLY JSON."
        }
    ]

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=1500,   # reduced
                temperature=0.0,
            )

            raw = response.choices[0].message.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            parsed = json.loads(raw)

            if not isinstance(parsed, dict):
                raise ValueError("Invalid JSON structure")

            return parsed

        except Exception as e:
            if "rate_limit" in str(e):
                wait = 5 * (attempt + 1)
                logger.warning(f"[Bill Agent] Rate limited. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("Failed after retries")


def clean_amount(val):
    if not val:
        return 0.0
    try:
        return float(str(val).replace("$", "").replace(",", "").strip())
    except Exception:
        return 0.0
    
def bill_agent_node(state: ClaimState) -> dict[str, Any]:
    bill_pages = state.get("bill_pages", [])

    if not bill_pages:
        logger.warning("[Bill Agent] No itemized bill pages found.")
        return {
            "bill_data": {"status": "no_bill_pages_found"},
            "errors": [],
        }

    logger.info("[Bill Agent] Processing %d bill page(s).", len(bill_pages))
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    all_bills = []
    errors = []

    for i, b64_img in enumerate(bill_pages):
        try:
            result = _process_single_bill_page(client, b64_img, i + 1)
            bills = result.get("bills", [])
            all_bills.extend(bills)
        except Exception as e:
            logger.error("[Bill Agent] Page %d failed: %s", i + 1, e)
            errors.append(f"Page {i+1}: {str(e)}")

    # ── Merge totals ───────────────────────────────────────────────
    grand_total = 0.0
    for bill in all_bills:
        try:
            grand_total += clean_amount(bill.get("total_amount"))
        except Exception:
            pass

    final_output = {
        "bills": all_bills,
        "grand_total_all_bills": round(grand_total, 2) if all_bills else None,
        "total_insurance_covered": None,
        "total_patient_responsibility": None,
    }

    logger.info("[Bill Agent] Completed with %d bills.", len(all_bills))

    return {
        "bill_data": final_output,
        "errors": errors,
    }