# 🎓 ScholarLens: Research Paper Digest & Claim Verification Copilot

> **Natural Language Processing (NLP) Course Project Submission**  
> **Topic:** Automated Academic Document Synthesis, Empirical Claim Verification & Lexical Simplification via LLMs  
> **Author:** Solo Project  

---

## 🌟 Project Overview

Academic papers are dense, highly technical, and time-consuming to digest. Researchers, students, and practitioners frequently spend hours attempting to extract core methodologies, assess whether reported benchmark numbers match reality, and decode impenetrable domain jargon.

**ScholarLens** is an end-to-end NLP system that leverages Large Language Models (LLMs) through a multi-stage, grounded pipeline. It ingests academic papers (PDF or text), decomposes their structure, extracts empirical claims, and translates complex concepts into intuitive mental models—all with strict anti-hallucination guardrails.

---

## 🏗️ Architecture & NLP Pipeline

```
┌────────────────────────────────────────────────────────┐
│             Input: PDF / TXT / Paste Text              │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│           Document Parser & Section Slicer             │
│  - Line-break & hyphenation repair (2-column arXiv)    │
│  - Academic section heuristic boundary detector        │
│  - Token estimation & length boundary normalization    │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│            Multi-Stage LLM Pipeline                    │
│                                                        │
│  [Stage 1: Executive Digest]                           │
│  TL;DR, Problem Statement, Novel Methods, Findings     │
│  (System: Senior Peer Reviewer, JSON Output)           │
│                                                        │
│  [Stage 2: Claim Verification Audit]                   │
│  Claims vs. Empirical Evidence vs. Limitations         │
│  (Few-Shot In-Context Learning + CoT Rubric)           │
│                                                        │
│  [Stage 3: Jargon Decoder & Analogies]                 │
│  Domain Terms -> Formal Def + Everyday Metaphors       │
│  (System: Science Communicator, Few-Shot)              │
│                                                        │
│  [Stage 4: Grounded Document Q&A]                      │
│  Closed-domain retrieval & citation Q&A                │
│  (Anti-hallucination negative constraints)             │
└──────────────────────────┬─────────────────────────────┘
                           ▼
┌────────────────────────────────────────────────────────┐
│            Interactive Streamlit Interface             │
│  Executive Tabs | Claims Table | Jargon Cards | Chat   │
└────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```text
scholar-lens/
├── config/
│   ├── config.yaml          # Generation hyperparameters, token budgets, timeouts
│   └── .env.example         # Template for API credentials
├── prompts/
│   ├── __init__.py          # Export interface
│   └── prompts.py           # Dedicated Prompt File (Personas, Few-shot, CoT, Schemas)
├── src/
│   ├── __init__.py
│   ├── config_loader.py     # Configuration & environment variable parser
│   ├── document_parser.py   # PDF text extraction, column repair & section tagging
│   ├── llm_client.py        # Resilient API client (Gemini SDK + REST fallback, OpenAI)
│   ├── models.py            # Pydantic schemas for deterministic JSON parsing
│   └── pipeline.py          # Pipeline orchestrator managing all 4 stages
├── samples/
│   └── attention_is_all_you_need_sample.txt  # Pre-loaded benchmark paper excerpt
├── app.py                   # Streamlit web application & visual dashboard
├── requirements.txt         # Project dependencies
└── README.md                # Comprehensive documentation
```

---

## 🎯 Alignment with Evaluation Criteria

### 1. Quality and Efficiency of Code
- **Modular Design:** Separation of concerns between ingestion (`document_parser.py`), API communication (`llm_client.py`), data validation (`models.py`), prompt engineering (`prompts/prompts.py`), and orchestration (`pipeline.py`).
- **Strict Data Validation:** Utilizes **Pydantic** models to validate API outputs, ensuring deterministic data types and preventing UI crashes.
- **Defensive Error Handling:** Implements exponential backoff retries (handling HTTP 429 / 503) and automated JSON repair routines.

### 2. Functionality and Implementation of the LLM API
- **Primary Provider (Google Gemini API):** Deploys `gemini-2.0-flash` / `gemini-1.5-flash` taking advantage of generous free tiers and large context windows.
- **Dual Invocation Pipeline:** Direct official SDK support with an automated HTTP REST fallback, ensuring seamless operation across any deployment environment.
- **OpenAI Compatibility:** Pluggable architecture allowing switching between Gemini and OpenAI models via configuration or UI.
- **Telemetry:** Measures execution latency, token estimates, and word counts in real time.

### 3. Effectiveness and Efficiency of the Prompt File (`prompts/prompts.py`)
- **System Persona Conditioning:** Distinct system prompts for different stages (e.g., `SYSTEM_ACADEMIC_PEER_REVIEWER` for critical claim auditing vs. `SYSTEM_SCIENCE_COMMUNICATOR` for conceptual analogies).
- **Few-Shot In-Context Exemplars:** Concrete input/output demonstrations to anchor model tone and calibrate ratings (*Empirically Verified*, *Partially Supported*, *Unsupported Speculation*).
- **Chain-of-Thought (CoT) Prompting:** Step-by-step guidance requiring evidence extraction prior to rating assignment.
- **Negative Constraints & Defensive Grounding:** Anti-hallucination rules instructing the model to declare absence of evidence rather than generating unverified assumptions.

### 4. Overall Project Quality and Execution
- **One-Click Benchmarking:** Includes pre-loaded landmark paper (*Attention Is All You Need*) for instantaneous testing without needing a PDF.
- **Multi-Tab Dashboard:** Executive summary, claims breakdown, interactive jargon cards, grounded Q&A chat, and 1-click Markdown export.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.9+ (Tested on Python 3.10, 3.11, 3.12, 3.14)
- A Google Gemini API key (Free at [Google AI Studio](https://aistudio.google.com/)) or an OpenAI API key.

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone <your-repo-link>
cd scholar-lens
pip install -r requirements.txt
```

### 3. Setup Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Edit `.env` with your API key:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
LLM_PROVIDER=gemini
```
*(Alternatively, you can enter the API key directly into the sidebar in the web UI at runtime).*

### 4. Run the Application
Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Testing the Project

1. Open the sidebar.
2. Select **"Load Benchmark Sample"** (pre-loaded with the Transformer paper excerpt).
3. Enter or verify your **Gemini API Key**.
4. Click **"🚀 Analyze Paper"**.
5. Explore the generated tabs:
   - **Executive Digest:** View the punchy TL;DR, problem vs methodology, and key benchmark scores.
   - **Claims Audit:** Review verified claims and identified hardware/dataset limitations.
   - **Jargon Decoder:** Review analogies (e.g., *Scaled Dot-Product Attention* explained as highlighters on a study guide).
   - **Grounded Q&A:** Ask *"What hardware was used to train the big model?"* or *"What was the BLEU score on English-to-German?"*.

---

## 📋 Evaluation Summary Table

| Requirement | Implementation Details |
| :--- | :--- |
| **Solo Project** | Developed independently for NLP course submission. |
| **LLM Integration** | Google Gemini API / OpenAI Chat API with structured schema extraction. |
| **Meaningful NLP Use** | Multi-task pipeline: abstractive summarization, claim extraction, text simplification, grounded closed-domain Q&A. |
| **Prompt File** | Dedicated `prompts/prompts.py` using personas, few-shot examples, JSON schema constraints, and grounding rules. |
| **Configuration File** | Centralized `config/config.yaml` controlling temperature, token limits, and provider options. |
| **Complete Source Code** | Modular architecture with full type annotations, Pydantic schemas, and Streamlit frontend. |
