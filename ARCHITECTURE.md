# Architecture — Enterprise Knowledge Assistant

> Companion to [README.md](./README.md). This document focuses purely on system architecture, data flow, and design decisions.

**Live (Phase 1 frontend):** <https://kartheekbvs89-arch.github.io/Langchain/>

---

## 1. System Architecture (End-State, 3 Phases)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                                │
│  Loads static frontend from GitHub Pages / CDN                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTPS
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — FRONTEND (delivered)                                    │
│  HTML · CSS · JavaScript (vanilla, no framework)                    │
│                                                                     │
│  index.html ─► login → sessionStorage                                │
│  chat.html ───► agent picker → message thread → composer            │
│  documents.html ► upload zone → table → filters                     │
│  analytics.html ► KPIs → charts (CSS/SVG) → RAGAS table            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ REST / WS  (Phase 2)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — BACKEND (upcoming)                                      │
│  Python 3.11 · FastAPI · LangChain · LangGraph                     │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │  API Gateway │──►│ Agent Router │──►│ Specialist Agent     │    │
│  │  (FastAPI)   │   │ (LLM classif)│   │ (HR / IT / Policy)   │    │
│  └──────────────┘   └──────────────┘   └──────────┬───────────┘    │
│                                                    │                │
│                              ┌──────────────────────▼──────────┐    │
│                              │  RAG Pipeline                   │    │
│                              │  embed → search → rerank        │    │
│                              │  → prompt → LLM → cite         │    │
│                              └──────┬──────────────┬──────────┘    │
│                                     │              │                │
│                              ┌──────▼──────┐ ┌────▼──────────┐     │
│                              │ Vector DB   │ │  LLM Provider │     │
│                              │ (Pinecone / │ │  (OpenAI /    │     │
│                              │  Qdrant)    │ │   Claude)     │     │
│                              └─────────────┘ └───────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — CLOUD DEPLOYMENT (upcoming)                             │
│  Container orchestration · Auto-scaling · Observability             │
│  AWS ECS Fargate / Azure Container Apps / GCP Cloud Run             │
│  Prometheus + Grafana · OpenTelemetry · LangSmith                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. RAG Data Flow (Per Query)

```
1. User types: "How many casual leaves am I entitled to?"
                    │
                    ▼
2. Frontend POST /api/chat {question, agent_hint?, conversation_id}
                    │
                    ▼
3. Backend Agent Router (LangChain LLM call)
   classifies → HR agent
                    │
                    ▼
4. Embed question
   OpenAI text-embedding-3-large → 3072-dim vector
                    │
                    ▼
5. Vector search
   cosine similarity vs HR namespace → top-K=5 chunks
                    │
                    ▼
6. (Optional) Cross-encoder rerank
   BGE-reranker or Cohere rerank → top-3 most relevant
                    │
                    ▼
7. Build prompt
   system_prompt + retrieved_context + user_question
                    │
                    ▼
8. LLM generate
   GPT-4o / Claude / Llama 3 → grounded answer
                    │
                    ▼
9. Citation extraction
   Parse answer → link each claim to source chunk
                    │
                    ▼
10. Return JSON
    { answer, sources: [{doc, page, score}], latency_ms }
                    │
                    ▼
11. Frontend renders message + source chips
```

---

## 3. Multi-Agent Design

```
                    ┌─────────────────────┐
                    │   User Question     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Agent Router       │
                    │  (LLM classifier)  │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │  HR Agent     │    │  IT Agent    │    │ Policy Agent │
   │              │    │              │    │              │
   │ Namespace:   │    │ Namespace:   │    │ Namespace:   │
   │   hr_docs    │    │   it_docs    │    │ policy_docs  │
   │              │    │              │    │              │
   │ System prompt│    │ System prompt│    │ System prompt│
   │ tailored to  │    │ tailored to  │    │ tailored to  │
   │ HR domain    │    │ IT domain   │    │ compliance   │
   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Shared RAG engine  │
                    │  (embed/search/LLM) │
                    └─────────────────────┘
```

### Why Multi-Agent?

| Concern | Single-Agent Solution | Multi-Agent Solution (ours) |
|---------|----------------------|----------------------------|
| Retrieval scope | All documents in one index → noisy | Per-domain vector namespaces → precise |
| System prompt | One prompt fits all → generic | Tailored prompts per domain → on-brand |
| Cost | All queries hit large index | Smaller per-domain indexes → faster + cheaper |
| Maintainability | One monolithic prompt | Each agent updated independently |
| Extensibility | Refactor entire system | Add a new agent (Finance, Legal, etc.) in isolation |

---

## 4. Document Ingestion Pipeline (Phase 2)

```
┌──────────────┐
│  PDF / DOCX  │
│  / TXT / MD  │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ 1. Loader         │  PyPDFLoader / UnstructuredMarkdownLoader / Docx2txt
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│ 2. Chunker       │  RecursiveCharacterTextSplitter
│                  │  chunk_size=512, overlap=64
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│ 3. Metadata      │  {source, agent, page, section,
│                  │   uploaded_by, uploaded_at}
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│ 4. Embedder       │  text-embedding-3-large (3072-dim)
└──────────┬───────┘
           │
           ▼
┌──────────────────┐
│ 5. Vector store   │  Upsert to Pinecone / Qdrant
│                  │  namespace = agent_id (hr / it / policy)
└──────────────────┘
```

---

## 5. Frontend Component Map

```
js/app.js
├── APP            ← static config (name, owner, version)
├── AGENTS[]       ← agent definitions (id, name, mock replies, suggested prompts)
├── requireAuth()  ← session guard, redirects to index.html if no user
├── logout()       ← clears session
├── renderSidebar(activePage)
├── renderHeader(title, breadcrumb, actions)
├── mountShell(activePage, title, breadcrumb, mainContent)
└── Icons: chatIcon(), docIcon(), chartIcon()

css/style.css
├── Design tokens (CSS custom properties)
├── Reset
├── Typography
├── Buttons (primary / secondary / ghost / icon)
├── Forms (input / textarea / select)
├── Layout (app-shell grid: sidebar / header / main)
├── Cards / badges / tags / tables
├── Stats / progress bars
└── Scrollbar + responsive breakpoints

css/pages.css
├── Login (split-screen B&W)
├── Chat (layout, agent cards, messages, composer, typing indicator)
├── Documents (upload zone, table, status pills, filters)
└── Analytics (bar chart, donut, line chart, phase tracker)
```

---

## 6. Design Decisions

### 6.1 Why vanilla HTML/CSS/JS for Phase 1?

| Factor | Decision |
|--------|---------|
| Speed to demo | No build step → instant iteration |
| GitHub Pages compatibility | Static files deploy in seconds |
| Lower review friction | Stakeholders open the URL, no install |
| Future flexibility | When Phase 2 backend ships, the frontend can be migrated to Next.js / React if needed, or kept as-is if vanilla is sufficient |

### 6.2 Why strict black & white?

| Factor | Decision |
|--------|---------|
| Enterprise tone | B&W signals seriousness, neutrality, accessibility |
| Accessibility | High contrast meets WCAG AAA |
| Visual hierarchy | Force hierarchy through typography & spacing, not color |
| Brand-agnostic | Works for any enterprise brand |

### 6.3 Why per-agent vector namespaces?

| Factor | Decision |
|--------|---------|
| Retrieval precision | Only the relevant domain's chunks are searched |
| Cost | Smaller indexes = faster + cheaper |
| Isolation | HR documents don't leak into IT responses |
| Maintainability | Each namespace can be re-indexed independently |

### 6.4 Why GitHub Pages for Phase 1 (and not for Phase 3)?

- Phase 1 = static files → Pages is free, fast, HTTPS.
- Phase 3 = backend + database → Pages can't host Python. Move to cloud (AWS/Azure/GCP) with container orchestration.

---

## 7. Security Posture

### Phase 1 (current)

- **Authentication**: mocked via `sessionStorage`. No real credentials.
- **No backend**: no attack surface for injection, RCE, or data exfiltration.
- **No user data stored**: session is cleared on logout.
- **CSP-friendly**: no inline event handlers in production (one small inline script in `index.html` for the login form).

### Phase 2 (planned)

- **SSO / OIDC**: enterprise identity provider integration.
- **Rate limiting**: per-user rate limits on `/api/chat`.
- **Input sanitization**: LangChain's input guardrails + custom regex filters.
- **PII redaction**: detect and redact PII before sending to LLM.
- **Audit log**: every query + response stored for compliance.

### Phase 3 (planned)

- **Secrets management**: AWS Secrets Manager / Azure Key Vault.
- **Network isolation**: VPC, private subnets for backend.
- **DDoS protection**: CloudFront / Azure Front Door.
- **Vulnerability scanning**: Snyk / Dependabot on every PR.

---

## 8. Performance Targets

| Metric | Target | Phase 1 (mock) | Phase 3 (live) |
|--------|--------|-----------------|----------------|
| First Contentful Paint | ≤ 1.5s | < 0.5s (static) | < 1.0s (CDN) |
| Chat P50 latency | ≤ 1.5s | 1.4s (mock) | TBD |
| Chat P95 latency | ≤ 3.0s | 2.4s (mock) | TBD |
| Vector search P95 | ≤ 400ms | n/a | TBD |
| Document ingestion (per MB) | ≤ 10s | n/a | TBD |

---

## 9. Open Questions for Phase 2

1. **LLM provider**: OpenAI GPT-4o vs. Azure OpenAI (for data residency) vs. self-hosted Llama 3 (for cost)?
2. **Vector DB**: Pinecone (managed, simplest) vs. Qdrant (self-hosted, cheaper at scale)?
3. **Conversation memory**: in-memory (lost on restart) vs. Redis (fast, ephemeral) vs. Postgres (durable)?
4. **Streaming**: SSE vs. WebSocket for token-by-token streaming responses?
5. **Eval cadence**: nightly batch vs. per-deploy vs. continuous shadow-eval on prod traffic?

---

## 10. References

- **LangChain docs**: <https://python.langchain.com>
- **RAGAS (eval framework)**: <https://docs.ragas.io>
- **Pinecone**: <https://www.pinecone.io>
- **Qdrant**: <https://qdrant.tech>
- **OpenAI embeddings**: <https://platform.openai.com/docs/guides/embeddings>
