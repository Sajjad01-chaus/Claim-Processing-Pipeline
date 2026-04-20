"""
FastAPI Application — Claim Processing Pipeline

Endpoint:  POST /api/process
  Form fields:
    claim_id  (str)  — unique identifier for this claim
    file      (PDF)  — the claim PDF to process

Response: JSON with all extracted claim data
"""

import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

# Validate env early — fail fast if GROQ_API_KEY is missing
if not os.environ.get("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

from app.pdf_utils import pdf_to_images
from app.graph.workflow import graph

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Claim Processing Pipeline",
    description=(
        "Processes medical claim PDFs using LangGraph + Groq vision. "
        "Segregates pages by document type and routes them to specialised "
        "extraction agents (ID, Discharge Summary, Itemized Bill)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "claim-processor"}


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/api/process")
async def process_claim(
    claim_id: str = Form(..., description="Unique claim identifier"),
    file: UploadFile = File(..., description="PDF claim document"),
) -> JSONResponse:
    """
    Process a medical claim PDF through the full LangGraph pipeline:

    1. Convert PDF pages → base64 images
    2. Segregator Agent: classify every page into one of 9 document types
    3. ID Agent: extract identity & policy information (parallel)
    4. Discharge Agent: extract clinical discharge data (parallel)
    5. Bill Agent: extract all itemized charges (parallel)
    6. Aggregator: merge all outputs into a single JSON response
    """
    # ── Validate input ────────────────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    if not claim_id.strip():
        raise HTTPException(status_code=400, detail="claim_id cannot be empty.")

    logger.info("━━━ New claim: %s | file: %s ━━━", claim_id, file.filename)
    start_time = time.perf_counter()

    # ── Read PDF ──────────────────────────────────────────────────────────
    try:
        pdf_bytes = await file.read()
        if len(pdf_bytes) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")

    # ── Convert PDF pages to images ───────────────────────────────────────
    try:
        page_images = pdf_to_images(pdf_bytes)
        logger.info("PDF converted: %d pages", len(page_images))
    except Exception as exc:
        logger.error("PDF conversion failed: %s", exc)
        raise HTTPException(status_code=422, detail=f"PDF processing error: {exc}")

    if not page_images:
        raise HTTPException(status_code=422, detail="PDF has no pages.")

    # ── Run LangGraph pipeline ────────────────────────────────────────────
    initial_state = {
        "claim_id": claim_id,
        "page_images": page_images,
        "page_classification": {},
        "id_pages": [],
        "discharge_pages": [],
        "bill_pages": [],
        "id_data": {},
        "discharge_data": {},
        "bill_data": {},
        "final_output": {},
        "errors": [],
    }

    try:
        result_state = await graph.ainvoke(initial_state)
    except Exception as exc:
        logger.error("LangGraph pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    elapsed = time.perf_counter() - start_time
    logger.info("━━━ Claim %s processed in %.2fs ━━━", claim_id, elapsed)

    final_output = result_state.get("final_output", {})
    final_output["processing_time_seconds"] = round(elapsed, 2)

    return JSONResponse(content=final_output)
