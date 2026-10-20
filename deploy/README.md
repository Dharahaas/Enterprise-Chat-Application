# Deployment

The production full-stack deployment in this repository is Docker Compose on a
single persistent Linux host. Caddy terminates HTTPS, Nginx serves the portal
and forwards `/api` requests, and FastAPI runs privately on the Compose network.
Chroma, uploaded originals, authentication accounts, and Caddy certificates use
persistent Docker volumes.

The root [`render.yaml`](../render.yaml) is a separate free-tier public
evaluation option, not a production deployment. Render Free has ephemeral
storage and cannot retain uploaded documents, Chroma edits, or auth databases.
See the Render deployment section in the root [`README.md`](../README.md).

The AWS/Azure/GCP deployment snippets previously described here are not part of
the supported deployment and should not be used as-is. They do not represent
the current Gemini configuration, server-validated authentication, or storage
requirements.

## Prerequisites

- A Linux server with Docker Engine and the Docker Compose plugin installed.
- A domain name with an `A`/`AAAA` record pointing to that server.
- Inbound TCP ports 80 and 443 permitted by the host firewall and cloud firewall.
- A Gemini API key from Google AI Studio.

## Configure secrets

Clone the private source repository onto the server. From its project root:

```sh
cp .env.example .env
mkdir -p secrets
chmod 700 secrets
openssl rand -base64 32 > secrets/admin_password
chmod 600 .env secrets/admin_password
```

Edit `.env` and set `APP_DOMAIN` to the DNS name. Set `ADMIN_USERNAME` to the
first administrator's login name. Put the Gemini API key, by itself, in
`secrets/gemini_api_key`, then restrict it:

```sh
read -rsp "Gemini API key: " GEMINI_API_KEY
printf '\n'
printf '%s' "$GEMINI_API_KEY" > secrets/gemini_api_key
unset GEMINI_API_KEY
chmod 600 secrets/gemini_api_key
```

The root `.env` and `secrets/` directory are ignored by Git. Do not put real
keys or passwords in source files, container images, or chat messages. The
backend reads the two secrets from mounted Compose secret files. Production
always disables dummy/demo sign-in.

## Start and update

```sh
docker compose up -d --build
docker compose ps
docker compose logs -f backend
```

When DNS has propagated and Caddy has obtained a certificate, visit
`https://<APP_DOMAIN>`. The backend port is not published to the internet.
Only Caddy publishes ports 80 and 443; Nginx and FastAPI remain private on the
Compose network. For a host that already has a TLS reverse proxy, adjust the
published ports and proxy configuration deliberately rather than exposing
FastAPI directly.

The first startup creates the configured admin account if it does not already
exist. Bootstrap settings do not reset an existing password. Create employee
accounts interactively inside the running backend container:

```sh
docker compose exec backend python manage_users.py employee@example.com --role employee
```

## Data and backups

Compose persists:

- Chroma index: `backend-data`
- Original uploaded documents: `backend-uploads`
- Users and sessions: `backend-auth`
- Caddy certificates/configuration: `caddy-data` and `caddy-config`

Back up these volumes to encrypted storage on a schedule and test restoring
them. Keep separate copies of source documents and secret files. Do not run
`docker compose down -v` unless you intentionally want to permanently delete
the app's persistent volumes.

Uploaded documents are indexed for retrieval; Gemini is not fine-tuned or
retrained. The agent can answer a wider range of questions only when its
assigned collection contains relevant, readable content. Review answers and
citations after adding important policy documents.

## Verification

After deployment:

1. Confirm `https://<APP_DOMAIN>/api/health` returns a healthy response.
2. Sign in using the configured admin account and verify Documents and
   Analytics are available.
3. Create an employee account and confirm it can use the portal/chat but
   receives an access-denied response for admin pages and APIs.
4. Upload a test document, verify it appears in Documents, and ask the matching
   agent a question whose answer is explicitly supported by that file.
5. Confirm Gemini responses and citations, then verify data remains after a
   container restart.
