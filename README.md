# 🏥 Claim Processing Pipeline (AI + LangGraph)

An AI-powered system to process medical claim PDFs using a **multi-agent architecture** built with **FastAPI + LangGraph + Groq multimodal LLMs**.

---

## UI Demo Screenshots
![Streamlit UI](https://raw.githubusercontent.com/Sajjad01-chaus/Claim-Processing-Pipeline/main/ui_ss/Screenshot%202026-04-20%20143303.png)

## 🧪 How to Use

### 🔹 Option 1 — Swagger UI 

1. Open the API docs:

   ```
   https://claim-processing-pipeline-wqvz.onrender.com/docs
   ```

2. Go to:

   ```
   POST /api/process
   ```

3. Click **Try it out**

4. Fill:

   * `claim_id`: any string (e.g. `TEST-001`)
   * `file`: upload the sample PDF

5. Click **Execute**

👉 You’ll get structured JSON response with extracted data.

---

### 🔹 Option 2 — cURL

```bash
curl -X POST https://your-app.onrender.com/api/process \
  -F "claim_id=TEST-001" \
  -F "file=@final_image_protected.pdf"
```

---

### 🔹 Option 3 — Streamlit Demo (Local)

create venv then install requirements and run 

```bash
streamlit run frontend.py
```

* Upload PDF
* Click “Process Claim”
* View extracted results

---

## 📝 Notes

* Processing may take **60–90 seconds** depending on API limits
* If rate limits occur, system returns partial results instead of failing


## 🚀 Overview

This system ingests a medical claim PDF and:

1. Converts it into page images
2. Classifies each page into document types using an AI agent
3. Routes relevant pages to specialized extraction agents
4. Aggregates structured outputs into a final JSON response

---

## 🧠 System Architecture

```
POST /api/process (PDF)
        │
        ▼
PDF → page images (PyMuPDF)
        │
        ▼
Segregator Agent (AI)
        │
        ▼
ID Agent
        │
        ▼
Discharge Summary Agent
        │
        ▼
Itemized Bill Agent
        │
        ▼
Aggregator
        │
        ▼
Structured JSON Output
```

---

## 🤖 Agents

### 1. Segregator Agent (Core Intelligence)

* Uses a multimodal LLM to classify each page
* Supports **9 document types**
* Routes only relevant pages to each agent
* Uses batching to reduce API calls

---

### 2. ID Agent

Extracts:

* Patient name
* Date of birth
* Policy number
* Insurance provider

---

### 3. Discharge Summary Agent

Extracts:

* Diagnosis
* Admission & discharge dates
* Physician details
* Hospital information

---

### 4. Itemized Bill Agent

* Processes bill pages **page-wise**
* Extracts all line items
* Computes totals and financial summaries
* Handles noisy numeric formats (`$`, commas, etc.)

---

### 5. Aggregator (Deterministic)

* Merges outputs from all agents
* Builds a unified `claim_summary`
* Produces final structured response

---

## 📄 Supported Document Types

* claim_forms
* cheque_or_bank_details
* identity_document
* itemized_bill
* discharge_summary
* prescription
* investigation_report
* cash_receipt
* other

---

## ⚙️ Key Design Decisions

### 🔹 Multimodal LLM instead of OCR

* Directly processes images → avoids OCR complexity
* More robust for real-world noisy documents

---

### 🔹 Page-level routing

* Each agent only processes relevant pages
* Reduces token usage and improves accuracy

---

### 🔹 Sequential execution (intentional)

* Initially designed for parallel execution using LangGraph
* Observed API rate limits (TPM) causing retries and unstable latency
* Switched to sequential execution to ensure:

  * controlled API usage
  * predictable latency
  * higher reliability

---

### 🔹 Token-aware batching

* Segregator batches pages to reduce API calls
* Balances latency vs rate limits

---

### 🔹 Fault tolerance

* Handles:

  * rate limits (429)
  * partial failures
* Returns `"status": "partial"` instead of crashing

---

## 🧠 Why LangGraph?

LangGraph enables structured orchestration of multi-agent workflows.

In this system, it provides:

* Clear separation of responsibilities between agents
* State management across pipeline stages
* Flexibility to switch between parallel and sequential execution

This makes the pipeline modular, debuggable, and production-ready.

---

## ⚠️ Known Constraints

* Multimodal LLMs introduce latency due to:

  * large image inputs
  * API rate limiting (TPM / TPD)

> Actual compute time is significantly lower than observed latency; delays are primarily due to API throttling.

---

## 🛠️ Local Setup

```bash
git clone <your-repo-url>
cd claim-processor

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY

uvicorn app.main:app --reload --port 8000
```

👉 API Docs: http://localhost:8000/docs

---

## 🧪 Testing

```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLM-001" \
  -F "file=@final_image_protected.pdf"
```


## 🧪 Sample Output

```json
{
  "claim_id": "CLM-2024-789456",
  "total_pages": 18,
  "document_inventory": {
    "identity_document": [3],
    "itemized_bill": [9, 10]
  },
  "claim_summary": {
    "patient_name": "John Smith",
    "total_billed_amount": 6624.65
  },
  "status": "success"
}
```

---

## 🧰 Tech Stack

* FastAPI
* LangGraph
* Groq (Llama 4 Scout – multimodal)
* PyMuPDF

---

## 📌 Future Improvements

* Hybrid OCR + LLM pipeline for faster processing
* Smart caching for repeated document formats
* Adaptive rate control for partial parallel execution

---

## 👨‍💻 Author

Built as part of an AI engineering assignment 

