# Enterprise Knowledge Assistant

**An enterprise knowledge chatbot built using RAG (Retrieval-Augmented Generation), Multi-Agent Architecture, and Generative AI.**

> **Repository:** [kartheekbvs89-arch/Langchain](https://github.com/kartheekbvs89-arch/Langchain)  
> **Live Demo (Frontend):** [GitHub Pages](https://kartheekbvs89-arch.github.io/Langchain/)  
> **Status:** RAG backend and employee portal with role-based administration

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [The 3-Phase Project Plan](#3-the-3-phase-project-plan)
4. [Phase 1 — Frontend UI](#4-phase-1--frontend-ui)
5. [Phase 2 — Backend](#5-phase-2--backend)
6. [How the Project Works](#6-how-the-project-works)
7. [File Structure](#7-file-structure)
8. [Running Locally](#8-running-locally)
9. [Static Preview on GitHub Pages](#9-static-preview-on-github-pages)
10. [Runtime Availability](#10-runtime-availability)
11. [Deployment and Operations](#11-deployment-and-operations)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Tech Stack](#13-tech-stack)
14. [License](#14-license)

---

## 1. Overview

The **Enterprise Knowledge Assistant** is a chatbot that lets employees ask natural-language questions about company HR policies, IT procedures, and compliance documents — and get accurate, **source-cited** answers in seconds.

Three specialist AI agents handle different domains:

| Agent | Domain | Example Questions |
|-------|--------|-------------------|
| **HR Agent** | Leaves, payroll, onboarding, benefits | "How many casual leaves am I entitled to?" |
| **IT Agent** | Access, VPN, hardware, software, tickets | "How do I configure VPN on macOS?" |
| **Policy Agent** | Code of conduct, compliance, security | "What are the data classification levels?" |

Each agent grounds its answer in retrieved documents using **RAG**, so every response includes a citation to the source document, page, section, and retrieval score.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER (Employee Browser)                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ HTTPS
┌───────────────────────────────▼─────────────────────────────────┐
│                  FRONTEND (Phase 1 — delivered)                   │
│  HTML · CSS · JavaScript                                          │
│  ├── Login page                                                  │
│  ├── Chat interface (3-agent selector)                            │
│  ├── Document management                                         │
│  └── Analytics dashboard                                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │ Authenticated REST API
┌───────────────────────────────▼─────────────────────────────────┐
│                  BACKEND (Phase 2 — delivered)                    │
│  Python · FastAPI · LangChain · Chroma                          │
│  ├── Multi-Agent Router (HR / IT / Policy)                       │
│  ├── RAG pipeline (retrieve → rerank → generate)                  │
│  ├── Document ingestion (load → chunk → embed → store)           │
│  └── Citation formatter                                           │
└─────────┬─────────────────────────────────────────┬───────────────┘
          │                                         │
┌─────────▼─────────────────┐   ┌──────────────────▼─────────────┐
│   VECTOR DATABASE          │   │         LLM PROVIDER            │
│  Chroma (local file-based)│   │   Gemini Flash-Lite (free tier)│
│                            │   │   setup message (no key)        │
│  Per-agent collections:    │   │                                 │
│   eka_hr, eka_it,          │   │   Generates grounded answers    │
│   eka_policy               │   │   from retrieved context         │
└────────────────────────────┘   └────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────┐
│              CLOUD DEPLOYMENT (Phase 3 — upcoming)                │
│   Container orchestration · Load balancing · Observability        │
└───────────────────────────────────────────────────────────────────┘
```

### RAG Data Flow (Per Query)

```
1. User question → normalize obvious typos and detect standalone conversation
2. Greetings and capability questions are answered without document retrieval
3. Knowledge question → embed with Chroma's ONNX all-MiniLM-L6-v2 (384-dim)
4. Search the selected agent's Chroma collection; convert distances to cosine similarity
5. Filter low-confidence and duplicate passages
6. Build role-separated prompt and generate with Gemini 3.5 Flash-Lite
7. Return JSON {answer, sources[], latency_ms, error_code}; provider/configuration failures stay explicit
8. Frontend renders the answer and source references separately
```

---

## 3. The 3-Phase Project Plan

| Phase | Scope | Tech | Status |
|-------|-------|------|--------|
| **Phase 1** | Frontend UI — login, chat, documents, analytics dashboard | HTML, CSS, JavaScript | ✅ Delivered |
| **Phase 2** | RAG + Multi-Agent backend — LangChain, Chroma vector DB, HR/IT/Policy agents, document ingestion pipeline | Python, FastAPI, LangChain, Chroma ONNX | ✅ Delivered |
| **Phase 3** | Evaluation and deployment preparation; supported single-host Docker Compose instructions | Docker Compose, HTTPS reverse proxy, RAG evaluation | Deployment is documented, not yet hosted |

---

## 4. Employee portal and frontend

### 4.1 Sign-in and employee portal (`index.html`, `portal.html`)

Employees and administrators use the same server-validated sign-in. The Northstar portal uses clearly marked fictional/sample company content and hosts the existing assistant in a floating, responsive widget.

Sessions use an expiring `HttpOnly` cookie and require a CSRF token for state-changing requests. Passwords are stored as PBKDF2 hashes in a local SQLite auth database. Employees can use the portal and assistant; administrators additionally see Documents and Analytics. The backend independently authorizes all administrative endpoints.

### 4.2 Chat Page (`chat.html`)

- Agent picker sidebar (HR / IT / Policy)
- Recent conversations list
- Suggested prompts in empty state
- Message thread with source citations
- Typing indicator animation
- Composer: auto-grow textarea, Enter to send, Shift+Enter for newline
- Uses the existing FastAPI chat API; the same page is reused in the portal's embedded assistant

### 4.3 Documents Page (`documents.html`)

- Top stats (docs, chunks, storage, embeddings)
- Per-agent knowledge-base cards with progress bars
- Drag-and-drop upload zone with agent-namespace selector
- Document table with status pills (indexed / processing / failed), search, agent filter

### 4.4 Analytics Page (`analytics.html`)

- 3-phase roadmap tracker (Phase 1 at 100%, Phases 2 & 3 at 0%)
- KPIs: accuracy, recall@5, P95 latency, weekly queries
- Pure-CSS/SVG charts: bar (query volume), donut (agent split), line (accuracy trend)
- Latency breakdown table
- RAGAS evaluation table per agent

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for full design system details.

---

## 5. Phase 2 — Backend (NEW)

The Phase 2 backend is a Python FastAPI application using LangChain for RAG orchestration and Chroma as the vector database.

### Key Features

- **FastAPI** REST API with Swagger docs at `/docs`
- **LangChain** orchestration: prompt templates, retrievers, document loaders, text splitters
- **Chroma** local vector database — no signup, no external dependencies
- **Chroma ONNX** (`all-MiniLM-L6-v2`) for CPU embeddings — loaded on the first retrieval or ingestion request; no embedding API key needed
- **Three specialist agents** (HR / IT / Policy), each with its own vector namespace, system prompt, and retrieval config
- **Configured language-model generation** — Gemini 3.5 Flash-Lite when `GEMINI_API_KEY` is configured; otherwise knowledge questions receive a clear setup message instead of a fabricated or chunk-based answer. Model access and quotas depend on the Google AI Studio project. OpenAI is an optional paid alternative.
- **Document ingestion pipeline** — load PDF/DOCX/TXT/MD → chunk (size=512, overlap=64) → embed → upsert to vector DB
- **11 sample documents** seeded on first run (4 HR + 3 IT + 4 Policy)
- **Auto-seeding** on startup if the vector DB is empty
- **CORS** configured for trusted local frontend origins
- **Role-based authentication** with SQLite-backed users, expiring sessions, CSRF checks, and admin-only document/analytics/evaluation APIs
- **End-to-end smoke tests** with pytest + FastAPI TestClient

### Quick Start

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Create an initial administrator: set ADMIN_USERNAME and ADMIN_PASSWORD in
# backend/.env (password must have at least 12 characters), or use the CLI below.
# Run (auto-seeds sample docs on first start)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Provision accounts interactively from the `backend` directory:

```powershell
.\.venv\Scripts\python.exe manage_users.py admin@northstar.example --role admin
.\.venv\Scripts\python.exe manage_users.py employee@northstar.example --role employee
```

The CLI prompts for passwords twice and stores only their PBKDF2 hashes. By
default the auth database lives in the current user's OS data directory; the
sample `.env` overrides it to `backend/data/auth/auth.sqlite3` (ignored by Git). A
configured bootstrap account is created only if that username does not already
exist; it does not reset an existing password.

For local UI testing without provisioning accounts, set `DEMO_AUTH_ENABLED=true`
in `backend/.env` while `ENVIRONMENT=dev`. The sign-in screen then accepts any
non-empty username and password for the selected Employee or Administrator
role, and the backend still enforces that role on every protected endpoint.
Demo sign-in is accepted only for loopback requests in development and remains
disabled by default; do not enable it in staging or production.

Then visit:
- **API docs:** <http://localhost:8000/docs>
- **Frontend (served by backend):** <http://localhost:8000/app/>
- **Health check:** <http://localhost:8000/api/health>

### API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/auth/login` | Validate credentials and create a session |
| `GET` | `/api/auth/session` | Validate the session and return its CSRF token |
| `POST` | `/api/auth/logout` | Invalidate the session |
| `POST` | `/api/chat` | Employee/admin chat; authenticated and CSRF-protected |
| `GET` | `/api/documents` | List indexed documents (admin only) |
| `POST` | `/api/documents/upload` | Upload + ingest (admin only; CSRF-protected) |
| `DELETE` | `/api/documents/{agent}/{doc_name}` | Delete chunks (admin only; CSRF-protected) |
| `GET` | `/api/analytics` | Analytics (admin only) |
| `GET` | `/api/health` | Backend status, version, LLM provider, collections |
| `POST` | `/api/seed?force=true` | Re-seed the vector DB (admin only; CSRF-protected) |

### Run Tests

```bash
cd backend
pytest tests/ -v
```

The tests verify: imports, agent specs, router classification, sample doc seeding, end-to-end RAG for all 3 agents, and the FastAPI endpoints via TestClient.

See [`backend/README.md`](./backend/README.md) for full backend docs.

---

## 6. How the Project Works

### User Journey

1. Sign in with an account provisioned by an administrator.
2. Land on the fictional Northstar employee portal.
3. Open the embedded assistant and select HR, IT, or Policy; chat uses the existing `/api/chat` contract.
4. Administrators can also open the protected Documents and Analytics pages.
5. Page guards and backend role checks protect admin data and operations.

### Frontend ↔ Backend Auto-Detection

`js/app.js` probes `/api/health` at:
- Same origin (when served by FastAPI at `http://localhost:8000/app/`)
- `http://127.0.0.1:8000`
- `http://localhost:8000`

The backend must be reachable for sign-in and portal use. The application does
not replace real API calls with mock responses when the service is unavailable.

---

## 7. File Structure

```
Langchain/
├── README.md                         ← this file
├── ARCHITECTURE.md                   ← design system & architecture deep-dive
├── LICENSE
├── .gitignore
│
├── index.html                         ← Login page (Phase 1)
├── portal.html                        ← Employee portal + embedded assistant
├── chat.html                          ← Chat interface (Phase 1, calls Phase 2)
├── documents.html                     ← Document management (Phase 1)
├── analytics.html                     ← Performance dashboard (Phase 1)
│
├── css/
│   ├── style.css                      ← Design system (tokens, components)
│   └── pages.css                      ← Page-specific styles
│
├── js/
│   └── app.js                         ← App shell + authenticated API client
├── frontend_server.py                 ← Restricted static frontend dev server
│
└── backend/                           ← Phase 2 — RAG + Multi-Agent
    ├── README.md                      ← Backend-specific docs
    ├── requirements.txt
    ├── .env.example
    ├── .gitignore
    ├── app/
    │   ├── main.py                    ← FastAPI app + lifespan (auto-seed)
    │   ├── config.py                  ← Settings (from .env)
    │   ├── models.py                  ← Pydantic schemas
    │   ├── agents/
    │   │   ├── agent_specs.py          ← HR / IT / Policy definitions
    │   │   └── router.py               ← Keyword-based classifier
    │   ├── rag/
    │   │   ├── embeddings.py           ← Chroma ONNX MiniLM embeddings
    │   │   ├── vectorstore.py          ← Chroma client
    │   │   ├── llm.py                  ← configured Gemini/OpenAI chat model
    │   │   └── pipeline.py             ← retrieve → rerank → generate
    │   ├── ingestion/
    │   │   ├── loader.py               ← PDF / DOCX / TXT / MD loaders
    │   │   ├── chunker.py              ← RecursiveCharacterTextSplitter
    │   │   └── ingest.py               ← Full pipeline + sample seeding
    │   ├── api/
    │   │   └── routes.py               ← All REST endpoints
    │   └── data/
    │       └── sample_docs_seed.py     ← 11 synthetic policy documents
    ├── data/
    │   └── sample_docs/{hr,it,policy}/ ← Same content as .txt files
    └── tests/
        ├── conftest.py                 ← Auto-seeds DB before tests
        └── test_smoke.py               ← End-to-end smoke tests
```

---

## 8. Running Locally

### Frontend only

```bash
cd Langchain
python frontend_server.py
```

Sign-in still requires the FastAPI backend below.

### Phase 1 + Phase 2 (full stack)

```powershell
# Terminal 1: backend
Set-Location backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000

# Terminal 2: from the project root, use the restricted static server
py frontend_server.py
# Visit http://127.0.0.1:5500/
```

Or run everything from one origin:

```bash
cd Langchain/backend
uvicorn app.main:app --port 8000
# Visit http://localhost:8000/ — redirects to the sign-in page
```

---

## 9. Static Preview on GitHub Pages

The frontend is deployed via **GitHub Pages** (static hosting only — no Python backend).

**URL:** <https://kartheekbvs89-arch.github.io/Langchain/>

GitHub Pages cannot run the authentication backend. This is a visual preview
only; sign-in, chat, document management, and analytics require the backend.

### Limitations of GitHub Pages

- Only serves static files (HTML/CSS/JS) — no Python, no FastAPI, no Chroma
- A deployed, trusted backend is required for authenticated features.

---

## 10. Runtime availability

| Component | Full-stack implementation | GitHub Pages preview |
|-----------|-------------------------------|-------------------------|
| Authentication | **FastAPI-validated user + expiring cookie session** | Unavailable |
| Chat responses | **LangChain RAG → Gemini free tier** (or explicit setup/provider message) | Unavailable |
| Source citations | **Chroma vector search** with source metadata | Unavailable |
| Document ingestion | **Backend pipeline: chunk → embed → store** | Unavailable |
| Analytics | Admin-only backend endpoints; some existing evaluation fields remain sample fallback values | Unavailable |
| Vector DB | **Chroma (local file-based)** | n/a |

---

## 11. Deployment and Operations

### Render free evaluation deployment

The root-level [`render.yaml`](./render.yaml) describes a single Render
Python Web Service. FastAPI already serves the existing frontend from `/app/`,
so the UI and `/api` calls stay same-origin; no separate frontend host or
production CORS allowlist is required. The Render service root must remain the
repository root because frontend assets are outside `backend/`.

The Blueprint installs [`backend/requirements.txt`](./backend/requirements.txt),
builds the bundled sample Chroma index, verifies every enabled agent collection,
starts one Uvicorn worker on Render's `$PORT`, and configures `/api/health` as
its health check. Production startup checks the built index without loading the
embedding model, and refuses to run when an agent collection is missing instead
of quietly starting with an empty knowledge base. The embedding path uses
Chroma's 384-dimensional ONNX MiniLM model instead of PyTorch-backed
`sentence-transformers`; the ONNX model is loaded lazily on the first retrieval
or upload. Gemini remains the chat-generation provider. This avoids an
embedding API dependency or billing requirement. The first retrieval may take
longer while Chroma downloads the ONNX model. The 11
synthetic sample document contents live in
[`backend/app/data/sample_docs_seed.py`](./backend/app/data/sample_docs_seed.py);
PDF/TXT examples are under `backend/data/sample_docs/`. The existing local
918 KB `backend/data/chroma_db/` is ignored by Git and is not deployed. Render
builds the same sample collections from the bundled corpus.

To create the public evaluation service, push this configuration to the
repository's `main` branch, then in Render choose **New → Blueprint**, select
the repository and `render.yaml`, and create the service using the **Free**
plan. Enter `GEMINI_API_KEY` as a secret when prompted. The Blueprint sets
production mode, Gemini, and public evaluation sign-in. The public demo grants
only the shared employee demo role; invalid credentials cannot obtain demo
administrator access, and protected admin routes still require real admin
credentials. Demo authentication is not production-grade; do not submit
sensitive information. No admin account is created by default.

The service root redirects to the existing sign-in page. The browser calls the
API on its current origin. For the separate local frontend server, local
backend discovery remains enabled; a static site hosted elsewhere can use the
existing `?api=https://<backend-origin>` override.

**Free-tier storage and availability:** Render Free services sleep after 15
minutes without traffic, may take about a minute to wake, and lose filesystem
changes when they sleep, restart, or redeploy. The sample index is included in
the build artifact, but uploaded documents, auth/session SQLite data, and
runtime Chroma changes are not durable and must be recreated or re-uploaded.
Persistent disks require a paid plan. Free CPU/memory and external Gemini
quotas can also constrain usage. This setup is for evaluation, not company
data or production.

Keep `GEMINI_API_KEY` only in Render's secret environment configuration or
the ignored local `backend/.env`. The root and backend `.gitignore` files
exclude `.env`, Chroma data, uploads, and the downloaded model cache. Never
put secret values in `render.yaml`, frontend files, or Git.

### Persistent production deployment

For actual company data, use the Docker Compose deployment on a persistent
Linux host: Caddy provides HTTPS, Nginx serves the UI and proxies API requests,
and FastAPI plus Chroma run privately. Compose includes persistent volumes for
the vector database, uploaded source files, accounts, and TLS certificates.
Production demo login is disabled and Gemini credentials are mounted from
secret files rather than copied into the backend image.

Follow the current instructions in [`deploy/README.md`](./deploy/README.md).
The older provider-specific cloud templates are not supported deployment
instructions.

The evaluation endpoints and metrics are implemented, but their output does
not by itself certify answer correctness or production readiness.

---

## 12. Evaluation Metrics

| Metric | What it measures | Target | Current |
|--------|------------------|--------|---------|
| **Faithfulness** | % grounded in retrieved context | ≥ 0.90 | Live via `/api/eval/run` |
| **Answer Relevancy** | How directly addresses query | ≥ 0.85 | Live via `/api/eval/run` |
| **Context Precision** | Relevance of top-K chunks | ≥ 0.85 | Live via `/api/eval/run` |
| **Context Recall** | Coverage of needed chunks | ≥ 0.88 | Live via `/api/eval/run` |
| **Answer accuracy** | End-to-end correctness | ≥ 90% | Live via `/api/eval/run` |
| **Retrieval recall@5** | % of gold chunks in top-5 | ≥ 0.90 | Live via `/api/eval/run` |
| **P50 latency** | Median query → response | ≤ 1.5s | Live via `/api/metrics` |
| **P95 latency** | 95th-percentile latency | ≤ 3.0s | Live via `/api/metrics` |

Run the live eval:
```bash
curl -X POST http://localhost:8000/api/eval/run | python3 -m json.tool
```

Or click the "Run Evaluation" button on the Analytics page.

---

## 13. Tech Stack

### Phase 1 (delivered)

- HTML5, CSS3, vanilla JavaScript (ES6+) — no frameworks, no build step
- Inter + JetBrains Mono fonts

### Phase 2 (delivered)

- Python 3.11+
- FastAPI — async REST API
- LangChain — RAG orchestration
- Chroma — local vector database
- Chroma ONNX — CPU embeddings (`all-MiniLM-L6-v2`, 384 dimensions)
- langchain-openai — optional OpenAI LLM swap-in
- PyPDF, python-docx — document loaders
- Pydantic v2 — request/response schemas
- pytest — smoke tests

### Phase 3 (delivered)

- **Docker + Docker Compose** — full-stack local + cloud deployment
- **AWS CloudFormation** — ECS Fargate + S3 + CloudFront + ALB + auto-scaling
- **Azure Bicep** — Container Apps + Blob Storage + CDN + Log Analytics
- **GCP Deployment Manager** — Cloud Run + Cloud Storage + Load Balancer
- **LangSmith** — automatic RAG call tracing (when `LANGSMITH_API_KEY` set)
- **Prometheus metrics** — `/api/metrics` endpoint with counters + histograms
- **RAGAS evaluation pipeline** — 24-question gold set + 4 metrics + heuristic scorer
- **GitHub Actions** — 7-job CI/CD pipeline (test, lint, build, deploy, eval, security)
- **Trivy** — vulnerability scanning on every push

---

## 14. License

MIT License — see [LICENSE](./LICENSE).
