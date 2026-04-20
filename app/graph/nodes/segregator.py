"""
Segregator Node — the brain of the pipeline.

Responsibilities:
  1. Receives all page images from the state.
  2. Calls Groq vision (llama-3.2-11b-vision-preview) in batches of 4 pages.
  3. Classifies every page into one of 9 document types.
  4. Splits the classified pages into three buckets:
       id_pages        → identity_document pages  (→ ID Agent)
       discharge_pages → discharge_summary pages  (→ Discharge Agent)
       bill_pages      → itemized_bill pages      (→ Bill Agent)
  5. Returns the updated state keys.

Why batches of 4?
  Groq's vision model handles multi-image prompts well up to ~5 images.
  Batching reduces total API calls from N pages → ceil(N/4) calls.
"""

import json
import os
import logging
from typing import Any

from groq import Groq

from app.graph.state import ClaimState

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DOC_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

BATCH_SIZE = 5

SYSTEM_PROMPT = """\
You are a medical-claims document classifier.
You will be shown one or more pages from a scanned medical claim PDF.
Each page is labelled "=== Page N ===" where N is its 0-based index.

Classify every page you see into EXACTLY ONE of these types:
  claim_forms            — medical or insurance claim forms
  cheque_or_bank_details — cheque images, bank account / IFSC details
  identity_document      — government ID, passport, Aadhaar, driver's license
  itemized_bill          — hospital/pharmacy bills with line-item charges
  discharge_summary      — hospital discharge summary (admission/discharge dates)
  prescription           — doctor's prescription listing medications
  investigation_report   — lab reports, blood tests, X-ray, pathology reports
  cash_receipt           — cash payment receipts
  other                  — anything else (consent forms, referral letters,
                           registration forms, appointment letters, etc.)

Respond with ONLY valid JSON, no extra text:
{
  "classifications": {
    "N": "<doc_type>",
    ...
  }
}
"""


# ── Helper ───────────────────────────────────────────────────────────────────

def _build_batch_content(batch: list[tuple[int, str]]) -> list[dict]:
    """
    Build the `content` array for a Groq multi-image message.
    Each (page_index, base64_image) pair becomes a text label + image_url block.
    """
    content: list[dict] = []
    for page_idx, b64_img in batch:
        content.append({
            "type": "text",
            "text": f"=== Page {page_idx} ==="
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_img}"
            }
        })
    content.append({
        "type": "text",
        "text": (
            "Now classify every page shown above. "
            "Return ONLY the JSON object described in the system prompt."
        )
    })
    return content


def _classify_batch(
    client: Groq,
    batch: list[tuple[int, str]],
    model="meta-llama/llama-4-scout-17b-16e-instruct",
) -> dict[int, str]:
    """Send one batch to Groq and return {page_index: doc_type}."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_batch_content(batch)},
            ],
            max_tokens=400,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        classifications = parsed.get("classifications", {})
        return {int(k): v for k, v in classifications.items()}

    except Exception as exc:
        logger.error("Segregator batch failed: %s", exc)
        # Fallback: mark every page in this batch as "other"
        return {page_idx: "other" for page_idx, _ in batch}


# ── LangGraph Node ───────────────────────────────────────────────────────────

def segregator_node(state: ClaimState) -> dict[str, Any]:
    """
    LangGraph node: classifies all pages and routes them to agent buckets.
    """
    logger.info("[Segregator] Starting. Total pages: %d", len(state["page_images"]))

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    page_images = state["page_images"]
    page_classification: dict[int, str] = {}

    # ── Batch classification ──────────────────────────────────────────────
    indexed_pages = list(enumerate(page_images))          # [(0, b64), (1, b64), ...]
    batches = [
        indexed_pages[i : i + BATCH_SIZE]
        for i in range(0, len(indexed_pages), BATCH_SIZE)
    ]

    for batch_num, batch in enumerate(batches):
        logger.info(
            "[Segregator] Classifying batch %d/%d (pages %d–%d)",
            batch_num + 1, len(batches),
            batch[0][0], batch[-1][0],
        )
        result = _classify_batch(client, batch)
        page_classification.update(result)

    logger.info("[Segregator] Classification complete: %s", page_classification)

    # ── Route pages to agent buckets ──────────────────────────────────────
    id_pages: list[str] = []
    discharge_pages: list[str] = []
    bill_pages: list[str] = []

    for page_idx, doc_type in page_classification.items():
        img = page_images[page_idx]
        if doc_type == "identity_document":
            id_pages.append(img)
        elif doc_type == "discharge_summary":
            discharge_pages.append(img)
        elif doc_type == "itemized_bill":
            bill_pages.append(img)

    logger.info(
        "[Segregator] Routing → id:%d  discharge:%d  bill:%d pages",
        len(id_pages), len(discharge_pages), len(bill_pages),
    )

    return {
        "page_classification": page_classification,
        "id_pages": id_pages,
        "discharge_pages": discharge_pages,
        "bill_pages": bill_pages,
        "errors": [],
    }
