# Hugging Face hosted model metadata

EKA can use Hugging Face Inference Providers through the hosted OpenAI-compatible
chat-completions endpoint. The application does not download model weights.

- **Default model ID:** `openai/gpt-oss-20b:fastest`
- **Model card:** <https://huggingface.co/openai/gpt-oss-20b>
- **License:** Apache-2.0, see the model card's license section.
- **Hosted API:** <https://router.huggingface.co/v1>
- **Provider policy:** `HF_PROVIDER=auto` uses Hugging Face's automatic/fastest
  provider selection. An explicit provider name can be configured where the model
  supports it; alternatively use a provider policy suffix in `HF_MODEL_ID`.

Hugging Face currently lists this model with Inference Providers (the model page
shows multiple providers). Availability, provider coverage, quotas, and pricing
can change. This selection is a configurable default, not a guarantee that every
provider is available to every account. Hosted inference may consume included
credits or incur pay-as-you-go charges; check the current Inference Providers
pricing and account limits before enabling it for real users.

To opt in, set `LLM_PROVIDER=huggingface`, `HF_TOKEN`, `HF_MODEL_ID`, and
`HF_PROVIDER` on the backend only. Create a Hugging Face user token with the
minimum `Inference Providers` permission required by the hosted API. Never put
the token in browser code or commit it. In Render, set `LLM_PROVIDER` to
`huggingface` and add `HF_TOKEN` as a secret environment variable. The checked-in
Render configuration deliberately keeps Gemini as the deployment default.

## Stream protocol

Connect to `ws(s)://<backend>/api/chat/ws`. The browser must send an allowed
`Origin`, an authenticated `eka_session` cookie, and then a version 1 `chat`
event containing the session's CSRF token and a normal `ChatRequest` payload.
The server accepts one request per connection. Events are JSON objects with
`v: 1`, `type`, and `request_id`; payload events are `start`, `status`, `token`,
and `complete`, with sanitized `error` and `cancel` events for failures or
cancellation. `complete` carries the final validated answer, actual responder,
sources, conversation ID, configured model ID/provider, first-token time, total
time, and retrieval time. The REST `POST /api/chat` remains available.

The backend limits each signed-in user to one active WebSocket chat per process,
and each stream to one request. Render currently starts one Uvicorn worker, so
that in-memory limit applies to its configured deployment. Multi-worker or
multi-instance deployments would need shared coordination to enforce a global
limit. Model timeouts bound a stream if the client disconnects while an upstream
read is blocked. For local testing, run the backend and frontend as usual, select
Hugging Face in `backend/.env`, then send a real chat request; `/api/health` and
`/api/llm/status` report configuration, not proof that inference succeeded.
`/api/llm/status.last_successful_inference_at` is populated only after an actual
successful generated answer. Automated tests mock model calls and make no paid
requests.

For local PowerShell use, set the variables in `backend/.env` and start the
backend from the `backend` directory with the existing Uvicorn command. To
exercise streaming manually, sign in through the normal app, ask a question in
the existing chat, and confirm token updates followed by a `complete` event in
the browser's network WebSocket inspector. To exercise the legacy fallback,
send an authenticated `POST /api/chat` request with the same CSRF header.
