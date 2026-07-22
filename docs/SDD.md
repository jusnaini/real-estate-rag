


# Software Design Document (SDD)

# Real Estate RAG Assistant

**Version:** 1.1
**Project Type:** LLM Zoomcamp Capstone Project
**Architecture:** Retrieval-Augmented Generation (RAG) Application

---

# 1. Introduction

## 1.1 Purpose

This document describes the software design and implementation plan for **Real Estate RAG Assistant**, an end-to-end Retrieval-Augmented Generation (RAG) application.

The purpose of this project is to demonstrate the practical implementation of an LLM application pipeline, including:

* data ingestion
* document preprocessing
* retrieval
* LLM-based answer generation
* evaluation
* monitoring
* user interaction

The Malaysian real estate domain is selected as a practical case study. The MVP focuses on assisting first-time property buyers in Malaysia with understanding common questions related to the property purchasing process.

---

# 2. Project Objectives

This project will be my [capstone project submission for LLM Zoomcamp 2026](https://github.com/DataTalksClub/llm-zoomcamp/).The objectives of this project are:

1. Build a complete RAG pipeline from document ingestion to user-facing application.
2. Demonstrate retrieval quality evaluation using measurable metrics.
3. Evaluate generated answers using LLM-based evaluation.
4. Provide source-grounded answers to reduce hallucination.
5. Compare multiple retrieval approaches and select the best.
6. Evaluate best practices: hybrid search, re-ranking, and query rewriting.
7. Create a reproducible development environment using Docker.

```
Data ingestion
      ↓
Document processing
      ↓
Retrieval
      ↓
Prompt construction
      ↓
LLM generation
      ↓
Evaluation
      ↓
Monitoring
      ↓
User interface
```

---

# 3. System Scope

## 3.1 MVP Scope

The initial version supports questions related to:

### Property Buying Process

Examples:

* Steps involved in buying a house
* Documents required
* SPA process
* Timeline considerations

### Financing

Examples:

* Home loan application
* Debt Service Ratio (DSR)
* Loan eligibility
* Loan margin

### Transaction Costs

Examples:

* Stamp duty
* Legal fees
* Additional purchasing costs

### Insurance

Examples:

* MRTA
* MLTA
* Fire insurance

### Example questions

* "How do I buy my first house?"
* "What costs should I prepare?"
* "How does home loan eligibility work?"
* "What are MRTA and MLTA?"
* "What legal fees and stamp duties are involved?"
* "What's the stamp duty for a RM500,000 property?"
* "What's the difference between MRTA and MLTA?"
* "What documents do I need for a home loan application?"
* "What should I consider before buying my first property?"

---

## 3.2 Out of Scope

The MVP does not include:

* Real-time property listings
* Personalized financial advice
* Property price prediction
* Legal advice
* Direct integration with banks
* Government database integration

These may be considered future extensions.

---

# 4. High-Level Architecture

```
User
                  |
                  v
          Streamlit Application
                  |
                  v
          RAG Pipeline
                  |
        +---------+---------+
        |                   |
        v                   v
 Retrieval Layer        LLM Generation
        |                   |
        v                   |
 Knowledge Base             |
        |                   |
        v                   v
 Document Processing    Answer + Sources
        |                   |
        v                   v
 Raw Documents         SQLite (logs)
                             |
                             v
                    Monitoring Dashboard
```

---

# 5. Technology Stack

## Programming Language

Python 3.11+

---

## LLM

Supported through OpenAI-compatible API.

Examples:

* Groq
* Gemini
* OpenRouter
* Cerebras

The implementation should allow model switching through configuration.

### llm usage example

```python
def get_client(provider="groq"):
  
    """
    Return an OpenAI-compatible client for the given provider.

    Supported: groq, openrouter, cerebras, gemini
    """

    config = {
        "groq": {
            "base_url": "https://api.groq.com/openai/v1",
            "api_key": os.getenv("GROQ_API_KEY"),
        },
        "openrouter": {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.getenv("OPENROUTER_API_KEY"),
        },
        "cerebras": {
            "base_url": "https://api.cerebras.ai/v1",
            "api_key": os.getenv("CEREBRAS_API_KEY"),
        },
        "gemini": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "api_key": os.getenv("GEMINI_API_KEY"),
        }
    }
    cfg = config[provider]
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"])

def test_openai_client(q, client, model, chat_completion=False):
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": q}
    ]
    if chat_completion:
        response = client.chat.completions.create(messages=messages,model=model)
        output = response.choices[0].message.content
    else:
        response = client.responses.create(input=messages,model=model)
        output = response.output_text
   
    input_tokens = getattr(response.usage,'input_tokens',getattr(response.usage,'prompt_tokens',0))
    output_tokens = getattr(response.usage,'output_tokens',getattr(response.usage,'completion_tokens',0))
    total_tokens = response.usage.total_tokens

    print(f"❓ Q: {q}")
    print(f"🤖 A: {output}")
    print(f">>"*20)
    print(f"Model : {model}")
    print(f"Input Tokens  : {input_tokens}")
    print(f"Output Tokens : {output_tokens}")
    print(f"Total Tokens  : {total_tokens}\n")
```

- groq/openrouter - supports both OpenAI SDK Responses and ChatCompletions method
- gemini/cerabras - supports OpenAI SDK ChatCompletions method only

---

## Embedding Model

The embedding model should be configurable.

Examples:

* OpenAI embeddings
* Sentence Transformers
* Gemini embeddings

---

## Retrieval

Two approaches will be implemented and compared:

* **MinSearch** (keyword-based baseline)
* **Vector search** (sentence-transformers embeddings + cosine similarity)

The best-performing approach from evaluation will be used in the final pipeline.

Optional improvements (see Best Practices):

* **Hybrid search** — combine keyword + vector scores
* **Re-ranking** — re-rank top-K results with a cross-encoder

---

## Application

Streamlit

Purpose:

* user interface
* query interaction
* displaying answers and citations

---

## Storage

Local storage:

* JSON files for processed documents
* SQLite for logging and monitoring

---

# 6. Project Structure

```
real-estate-rag/

├── data/
│   ├── raw/
│   └── processed/
│       └── documents.json

├── ingest/
│   ├── scraper.py
│   ├── cleaner.py
│   ├── chunker.py
│   ├── ingest.py
│   └── dlt_pipeline.py

├── rag/
│   ├── build_index.py
│   ├── search.py
│   ├── prompts.py
│   └── rag_pipeline.py

├── evaluation/
│   ├── generate_ground_truth.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_answers.py
│   └── judge_prompt.txt

├── monitoring/
│   ├── logger.py
│   └── dashboard.py

├── app/
│   └── app.py

├── config.py

├── Dockerfile

├── docker-compose.yml

├── requirements.txt

└── README.md
```

---

# 7. Data Pipeline Design

## 7.1 Data Sources

The MVP uses 16 publicly available Malaysian property educational resources across 4 categories: buying process, legal cost, financing, and insurance. Two calculator-only pages (PropCashflow Legal Fee Calculator, Suppiah & Partners Legal Fee Calculator) were excluded because they contained <50 chars of article text (pure calculator widgets). PropertyGenie was excluded as an upstream — the site consistently returned HTTP 429 despite retry logic.

Categories:

| Category       | Sources                                              |
| -------------- | ---------------------------------------------------- |
| Buying Process | iProperty, IQI Global (2 articles), StashAway        |
| Financing      | iHome, PropCashflow (2 articles), RinggitPlus (2), CalculatorMalaysia |
| Legal Cost     | iProperty, PropCashflow, NewProjek, Rummah           |
| Insurance      | PropCashflow, Foundation                             |

**1. Buying process (step-by-step)**

- iProperty — [The Process of Buying a House in Malaysia 2026](https://www.iproperty.com.my/guides/documents-and-paperwork-buying-a-house-in-malaysia-71908)
- IQI Global — [Step-by-Step Guide to Buying a House in Malaysia](https://iqiglobal.com/blog/step-by-step-guide-buy-house-malaysia/) and their [Comprehensive Guide](https://iqiglobal.com/blog/complete-guide-to-purchasing-property-in-malaysia/)
- StashAway — [Complete Guide For First Time Home Buyer](https://www.stashaway.my/r/complete-guide-first-time-home-buyer-buying-house-in-malaysia)

**2. Forms, stamp duty, legal fees (LHDN/SRO-based)**

- iProperty — [Property Stamp Duty in Malaysia: How to Calculate](https://www.iproperty.com.my/guides/stamp-duty-spa-legal-fees-owning-a-house-malaysia-24760) (SPA, MOT, loan agreement duty + 2026 Budget updates)
- PropCashflow — [Stamp Duty (MOT) Malaysia 2026](https://propcashflow.my/blog/stamp-duty-malaysia-guide-2026/)
- NewProjek — [Stamp Duty Calculator Malaysia 2026](https://newprojek.com/calculators/stamp-duty-calculator) (tiered rate breakdown)
- Rummah — [Malaysian Stamp Duty Calculator](https://rummah.my/tools/stamp-duty-calculator) (first-time buyer exemption conditions)

*Excluded:* PropCashflow Legal Fees Calculator and Suppiah & Partners Legal Fee Calculator — calculator pages with <50 chars of article text.

**3. Loans/financing (DSR, LTV, rates)**

- iHome — [How Much Home Loan Can You Afford? DSR Explained](https://ihome.my/guides/home-loan-dsr-malaysia/)
- PropCashflow — [DSR Calculation Malaysia: Home Loan Eligibility](https://propcashflow.my/blog/home-loan-eligibility-dsr-malaysia/)
- PropCashflow — [How Much Loan Can You Get? LTV Rules 2026](https://propcashflow.my/blog/loan-margin-financing-property-malaysia/)
- RinggitPlus — [Best Housing Loans in Malaysia 2026](https://ringgitplus.com/en/home-loan/) (bank comparison + OPR/SBR explanation)
- RinggitPlus — [DSR Calculator](https://ringgitplus.com/en/calculators/debt-service-ratio-dsr-calculator/)
- CalculatorMalaysia — [Home Loan Eligibility Calculator 2026](https://calculatormalaysia.com/loan/home-loan-eligibility-calculator-malaysia/)

**4. Insurance (MRTA/MLTA/fire)**

- PropCashflow — [Property Insurance Malaysia: Fire, MRTA &amp; Homeowner Coverage Compared](https://propcashflow.my/blog/property-insurance-fire-insurance-malaysia/)
- Foundation — [Bank Loan Fire Insurance Malaysia: Requirements](https://www.getfoundation.com.my/blog/bank-loan-fire-insurance-requirements-malaysia)

*Excluded:* PropertyGenie — consistently returned HTTP 429 despite exponential-backoff retry.

---

## 7.2 Ingestion Pipeline

The ingestion pipeline will be automated using **dlt** (covered in the course workshop), orchestrated to run on a schedule or on-demand:

```
Source Articles
      |
      v
dlt pipeline (extract + load)
      |
      v
Document Extraction (scraper.py)
      |
      v
Cleaning (cleaner.py)
      |
      v
Chunking (chunker.py)
      |
      v
Metadata Creation
      |
      v
documents.json
```

The dlt pipeline will:

* Extract article content from source URLs
* Load raw HTML into a local staging area
* Trigger the cleaning and chunking steps
* Log pipeline run status and errors

---

## 7.3 Document Schema

Each chunk should contain:

```json
{
    "id": "unique_id",
    "source": "website_name",
    "category": "loan",
    "title": "article title",
    "url": "source url",
    "text": "document chunk"
}
```

---

# 8. RAG Pipeline Design

## 8.1 Retrieval Flow

```
User Question

      |
      v

Query Processing

      |
      v

Retriever

      |
      v

Top-K Relevant Chunks

      |
      v

Context Construction
```

---

## 8.2 Generation Flow

```
System Prompt

+

User Question

+

Retrieved Context

        |

        v

        LLM

        |

        v

Grounded Answer

+

Source References
```

---

# 9. Prompt Design Requirements

The assistant should:

* answer only based on retrieved context
* clearly indicate uncertainty
* avoid making legal or financial decisions
* cite source documents
* explain technical terms in simple language

Example:

```
You are a Malaysian property buying assistant.

Answer the user's question using only the provided context.

If the information is insufficient, say that the available sources do not provide enough information.

Always mention the source article used.
```

---

# 10. Evaluation Design

## 10.1 Retrieval Evaluation

Metrics:

* Hit Rate
* Mean Reciprocal Rank (MRR)

Process:

1. Generate ground-truth question-answer pairs from documents.
2. Evaluate **multiple retrieval approaches**:
   * MinSearch (keyword)
   * Vector search (sentence-transformers)
   * Hybrid search (combined keyword + vector)
   * Re-ranked (cross-encoder applied to top-K)
3. Compare metrics across approaches.
4. Select the best-performing approach for the production pipeline.

---

## 10.2 Generation Evaluation

Evaluation criteria:

* Relevance

Evaluation method:

LLM-as-a-Judge.

The system evaluates each generated answer in real-time using an LLM judge.
The judge classifies answers as **RELEVANT**, **PARTLY_RELEVANT**, or **NON_RELEVANT**.
Results are stored alongside user feedback in the monitoring database and
visualised on the dashboard (relevance rate KPI, relevance distribution pie chart).

---

# 11. Monitoring Design

The system records:

```
query
answer
retrieved_documents
latency
timestamp
feedback
```

Stored in:

SQLite database

---

Dashboard displays (minimum 5 charts):

* total queries over time
* average latency over time
* popular questions (top-10)
* user feedback score distribution
* token usage over time (input + output per query)

---

# 12. Deployment

The application should run locally using:

```
docker-compose up
```

Services:

```
Streamlit Application

+

Supporting services (optional)
```

---

# 13. Development Milestones

## Phase 1 - Data Pipeline

Deliverables:

* source collection
* ingestion script
* processed documents

## Phase 2 - RAG Pipeline

Deliverables:

* indexing
* retrieval
* generation

## Phase 3 - Evaluation

Deliverables:

* ground truth dataset
* retrieval evaluation
* answer evaluation

## Phase 4 - Application

Deliverables:

* Streamlit UI
* logging
* documentation

---

# 14. Success Criteria

The project is considered successful when:

* Users can ask property-related questions.
* The system retrieves relevant information.
* Answers are grounded in retrieved documents.
* Retrieval performance can be measured.
* Generated answers can be evaluated.
* The entire system can be reproduced locally.

---

# 15. Best Practices & Bonus Features

## 15.1 Best Practices (targeted for evaluation points)

The following will be implemented or at least evaluated:

### Hybrid Search

Combine keyword-based (MinSearch) and vector-based (sentence-transformers) scores using weighted fusion. Evaluate whether hybrid improves over either approach alone.

### Document Re-Ranking

Apply a cross-encoder model (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to re-rank the top-K retrieved chunks. Evaluate improvement in MRR and answer quality.

### Query Rewriting

Rewrite user queries before retrieval to improve search quality. For example, expand abbreviations (MRTA → Mortgage Reducing Term Assurance) or reformulate vague questions.

---

## 15.2 Bonus: Cloud Deployment

As a stretch goal, deploy the application to a cloud platform (e.g., Hugging Face Spaces, Render, or Railway) so it is accessible publicly without local setup.
