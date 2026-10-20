# Phase 2 — Backend (Python + FastAPI + LangChain + Chroma)

The RAG + Multi-Agent backend for the Enterprise Knowledge Assistant.

## Quick Start

### 1. Install dependencies

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Google Gemini

```bash
copy .env.example .env
```

By default, the backend uses:
- **Chroma ONNX** (`all-MiniLM-L6-v2`, 384 dimensions) for CPU embeddings — no embedding API key
- **Chroma** for the vector DB — local file-based, no signup
- **Gemini 3.5 Flash-Lite** for hosted answer generation

The server also initializes a local SQLite auth database. By default it is
stored in the current user's OS data directory; `AUTH_DATABASE_PATH` can
override the location. Create the first admin either by setting
`ADMIN_USERNAME` and a unique `ADMIN_PASSWORD` (at least 12 characters) in
`backend/.env`, or by running the interactive account manager after
installation:

```powershell
.\.venv\Scripts\python.exe manage_users.py admin@northstar.example --role admin
.\.venv\Scripts\python.exe manage_users.py employee@northstar.example --role employee
```

The tool prompts for the password twice. User passwords are stored as PBKDF2
hashes. Existing bootstrap users are not reset on server restart. Provision
employee accounts through this administrator-controlled tool; there is no
public self-registration endpoint.

Create a Gemini API key at <https://aistudio.google.com/apikey>, then edit
`backend/.env` for local development:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-3.5-flash-lite
```

Keep the key in the ignored `backend/.env`; never paste it into chat or commit
it. The backend can also read `GEMINI_API_KEY_FILE` for mounted deployment
secrets. `GEMINI_MODEL` is read from environment configuration and defaults to
`gemini-3.5-flash-lite`. Google's model access and free-tier quotas depend on
the AI Studio project; check active access in AI Studio. To avoid charges,
keep the project on the free tier and do not link it to a billing-enabled
Google Cloud account.

**Privacy:** questions and retrieved document passages are sent to Google.
Google's Gemini free-tier terms may allow submitted data to be used to improve
its products. Do not send sensitive company documents unless your organization
allows it.

Knowledge questions return a clear configuration/availability message if the
provider is not configured or unavailable, rather than inventing an answer.
Greetings, thanks, and capability questions are answered locally without a
model call.

The optional OpenAI provider is a separate, paid alternative. It is not needed
for the Gemini setup.

When uploading a document, the backend extracts and chunks its readable
content, embeds those chunks, and stores them in the selected agent's persistent
Chroma collection. The original uploaded file is retained in the server's
`data/uploads` directory. This expands retrieval context; it does **not** train
or fine-tune Gemini. Answers remain limited to evidence in retrieved chunks.
Chroma's ONNX MiniLM embedding model is dimension-compatible with the existing
384-dimensional `all-MiniLM-L6-v2` index. Rebuild the index when changing to a
different embedding model.

### 3. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server auto-seeds sample documents on first run. Embeddings load lazily
when retrieval or ingestion is first used, not during server startup. Chroma
downloads its ONNX model on first use. Once running:

- **API docs (Swagger):** <http://localhost:8000/docs>
- **Health check:** <http://localhost:8000/api/health>
- **Frontend:** <http://localhost:8000/app/> (serves the Phase 1 frontend)

### 4. Use the authenticated chat

Sign in at <http://localhost:8000/app/> or the separate frontend's
`index.html`, then open the employee portal's embedded assistant. The frontend
logs in at `/api/auth/login`, obtains its CSRF token from `/api/auth/session`,
and sends that token with chat and other state-changing requests.

Response shape:
```json
{
  "answer": "Employees are entitled to 12 casual-leave days per calendar year.",
  "agent": "hr",
  "sources": [
    {"document": "Leave_Policy.docx", "page": null, "section": "3.1", "score": 0.55, "snippet": "..."}
  ],
  "latency_ms": 1340,
  "llm_provider": "gemini",
  "conversation_id": "abc-123",
  "timestamp": "2026-..."
}
```

### 5. Run tests

```bash
pytest tests/ -v
```

## API Reference

### `POST /api/chat`
Run the RAG pipeline for an authenticated employee or administrator. Requests
require the session cookie and `X-CSRF-Token` header. Body:
```json
{
  "question": "...",
  "agent": "it",
  "conversation_id": "optional",
  "conversation_history": [
    {"role": "user", "content": "How do I set up VPN?"},
    {"role": "assistant", "content": "Follow these setup steps..."}
  ]
}
```
`agent` is optional. When supplied (as the chat UI does), the selected agent is
honored. If omitted, the backend classifies the question and routes it to an
agent automatically. `conversation_history` is optional and should contain only
the preceding turns from the same conversation (up to six messages, 2,000
characters each); it is used to resolve follow-ups and is not persisted by the
backend.

### `GET /api/health`
Lightweight liveness endpoint for Render. It returns application status without
calling Gemini, initializing embeddings, or retrieving documents.

### `GET /api/documents`
List all indexed documents across all agents (administrator only).

### `POST /api/documents/upload`
Upload and ingest a document (administrator only; requires CSRF token). Multipart form:
- `file`: the file
- `agent`: `hr|it|policy`

### `DELETE /api/documents/{agent}/{doc_name}`
Delete all chunks of a document (administrator only; requires CSRF token).

### `GET /api/analytics`
Return analytics (administrator only).

### `GET /api/health`
Return backend status, version, LLM provider, collections.

### Authentication endpoints

- `POST /api/auth/login` validates credentials and sets an expiring `HttpOnly`,
  `SameSite=Lax` cookie; production/staging cookies are `Secure`.
- `GET /api/auth/session` validates the cookie and returns the role and CSRF token.
- `POST /api/auth/logout` invalidates that server-side session and clears the cookie.
- Employees can use chat but receive `403` for document, analytics, evaluation,
  metrics, and seeding operations. The backend enforces these checks independently
  of frontend navigation.

### `POST /api/seed?force=true`
Re-seed the vector DB with sample documents.

## Architecture

```
backend/
├── app/
│   ├── main.py              ← FastAPI app + lifespan hooks
│   ├── config.py             ← settings (from .env)
│   ├── auth.py               ← PBKDF2 users, SQLite sessions, roles, CSRF
│   ├── models.py             ← Pydantic request/response schemas
│   ├── agents/
│   │   ├── agent_specs.py    ← HR / IT / Policy agent definitions
│   │   └── router.py         ← keyword-based agent classifier
│   ├── rag/
│   │   ├── embeddings.py     ← Chroma ONNX MiniLM embeddings
│   │   ├── vectorstore.py    ← Chroma vector DB client
│   │   ├── llm.py            ← configured Gemini/OpenAI chat model
│   │   └── pipeline.py       ← retrieve → rerank → generate
│   ├── ingestion/
│   │   ├── loader.py         ← PDF / DOCX / TXT / MD loaders
│   │   ├── chunker.py        ← RecursiveCharacterTextSplitter
│   │   └── ingest.py         ← full ingestion pipeline + seed
│   ├── api/
│   │   ├── auth_routes.py     ← login, session, logout
│   │   └── routes.py          ← chat + protected admin REST endpoints
├── manage_users.py            ← interactive admin/employee provisioning
│   └── data/
│       └── sample_docs_seed.py ← synthetic policy documents
├── data/
│   └── sample_docs/{hr,it,policy}/  ← same content as .txt files
├── tests/
│   ├── conftest.py           ← isolated auth database and test accounts
│   ├── test_auth.py          ← sessions, roles, CSRF, and admin operations
│   └── test_smoke.py         ← end-to-end smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```

## RAG Pipeline

```
1. Receive question and optional selected agent; classify only when no agent was supplied
2. Embed question (Chroma ONNX all-MiniLM-L6-v2 → 384-dim vector)
3. Vector search in the resolved agent's Chroma collection (top-K=5), filtering scores below 0.35
4. Rerank and deduplicate confident passages (top-3)
5. Build prompt: system prompt + context chunks + question
6. LLM generate (configured Gemini free-tier model or optional OpenAI model; provider errors are explicit)
7. Return answer + sources + latency
```

## Sample Documents (11 total)

- **HR (4):** HR_Policy_Manual_2024, Leave_Policy, Payroll_Procedure, Onboarding_Checklist
- **IT (3):** IT_SOP_PasswordReset, IT_Network_Guide_v2, Asset_Policy
- **Policy (4):** InfoSec_Policy_v4, Code_of_Conduct_2024, Hybrid_Work_Policy_2024, Security_Incident_Response

Each document is chunked (chunk_size=512, overlap=64) and stored in its agent's Chroma collection (`eka_hr`, `eka_it`, `eka_policy`).

## Frontend Integration

The frontend uses `/api/health` to discover the backend, then authenticates
through the cookie-backed session endpoints. Chat and admin operations never
fall back to mock responses. For split local development, run the restricted
frontend server on port 5500 and keep that origin in `CORS_ORIGINS`.
GitHub Pages is a static preview and cannot use this server-side authentication.

For local-only UI testing, `DEMO_AUTH_ENABLED=true` allows non-empty dummy
credentials to sign in under the role selected on the login page. This is
accepted only for loopback clients when `ENVIRONMENT=dev`; keep it disabled
outside local development.

## Optional: Use OpenAI instead

OpenAI is a paid alternative, not required for local development. To use it,
get a key from <https://platform.openai.com/api-keys> and edit `backend/.env`:
   ```
   LLM_PROVIDER=openai
   OPENAI_API_KEY=sk-your-real-key-here
   OPENAI_MODEL=gpt-4o-mini
   ```

Restart the server after changing the provider.

## Docker (optional)

```bash
# Build a simple image
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Phase 3 Roadmap

- [ ] Replace in-memory query log with Postgres
- [ ] Add LangGraph for multi-agent workflows
- [ ] Add cross-encoder reranker (BGE-reranker)
- [ ] Add SSE streaming for token-by-token responses
- [ ] Add LangSmith tracing
- [ ] Deploy to cloud (AWS ECS / Azure Container Apps)
- [ ] Wire up live RAGAS evaluation pipeline
