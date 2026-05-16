# Integration & Security Guide — Support RAG API (MS3)

This document covers everything an external team needs to integrate the
Customer Support RAG chatbot into a support portal: API surface, auth model,
deployment topology, threat assumptions, and operational runbook.

---

## 1. Service overview

| Property | Value |
|---|---|
| Service name | `support-rag-api` |
| Runtime | Python 3.11, FastAPI + Uvicorn, packaged as a single container |
| Hosting | Azure Container Apps (ACA) — see `deploy/azure/main.bicep` |
| Image registry | **GitHub Container Registry** (`ghcr.io/nhahub/support-rag-api`) — free |
| Vector store | **MongoDB Atlas M0** (free tier, 512 MB) |
| LLM backend | **HuggingFace Inference API** (free serverless) in cloud; Ollama+Mistral locally |
| Public surface | HTTPS, single ingress with cert managed by ACA |
| Auth | API key (`X-API-Key` header) |
| Latency budget | < 30 s p95 end-to-end (HF cold-start + RAG) |

The API exposes the existing `RAGPipeline` (BGE retrieval → LLM
generation) over REST. The LLM backend is pluggable via the
`RAG_GENERATION_PROVIDER` env var (`ollama` for local, `huggingface`
for cloud) — see `src/rag/generator_factory.py`. The pipeline is
intentionally small and stateless: every request stands alone.

---

## 2. Endpoints

Base URL in production: `https://supportrag-api.<region>.azurecontainerapps.io`

OpenAPI UI: `<base>/docs`  ·  Schema: `<base>/openapi.json`

| Method | Path     | Auth   | Purpose |
|--------|----------|--------|---------|
| GET    | `/health`| none   | Liveness + dependency probe |
| POST   | `/search`| API key| Vector search only (no LLM) |
| POST   | `/chat`  | API key| Full RAG: retrieve + generate |

### `POST /chat`

```http
POST /chat
Content-Type: application/json
X-API-Key: <secret>

{
  "query":           "How do I cancel my order?",
  "top_k":           5,                  // optional, 1–20
  "filter_category": "ORDER",            // optional
  "include_chunks":  false               // include retrieved chunks in response
}
```

```json
{
  "query":      "How do I cancel my order?",
  "answer":    "Orders can be cancelled within 24 hours of purchase…",
  "top_intents": ["cancel_order", "cancel_order", "change_order", "track_order", "delivery_options"],
  "chunks":      null,
  "latency_ms":  3127
}
```

### `POST /search`

Same request body shape as `/chat` minus `include_chunks`. Returns the
ranked chunks without invoking the LLM — useful for portal-side
autocompletion / "did you mean" UX.

### `GET /health`

```json
{
  "status":            "ok",
  "mongo":             "ok",
  "ollama":            "ok",
  "vector_store_docs": 24001,
  "embedding_model":   "BAAI/bge-base-en-v1.5",
  "generation_model":  "mistral"
}
```

`status` is `degraded` when any backing dependency reports an error. ACA's
liveness probe ignores the body — it only checks for `200`.

---

## 3. Authentication

### Why API key (not Azure AD)

| Factor | API key | Azure AD (OAuth2) |
|---|---|---|
| Setup cost | None | App registration, tenant config |
| Per-call cost | $0 | $0 (free tier) |
| Integration code (portal side) | One header | OAuth flow + token cache |
| Key rotation | Rotate Key Vault secret, restart revision | Token expiry handled by SDK |
| Audit trail | Application logs only | Azure AD sign-in logs |

For a chatbot embedded in an internal portal that already authenticates
its users, an API key is the right trade-off. Upgrade path to AAD is
straightforward (add a second middleware in `api/auth.py`); see §7.

### How keys are handled

| Stage | Where the key lives |
|---|---|
| Local dev | `API_KEY` env var in your shell |
| CI builds | GitHub Actions repository secret |
| Production | Azure Key Vault secret `api-key`, mounted into the Container App as `API_KEY` env var |

The API compares the inbound header against `API_KEY` using `hmac.compare_digest`
to avoid timing oracles. If `API_KEY` is unset, the API refuses every
request with `503 Service Unavailable` rather than failing open.

### Rotation

```bash
# Generate a new key
NEW_KEY=$(openssl rand -hex 32)

# Update Key Vault
az keyvault secret set --vault-name <kv-name> --name api-key --value "$NEW_KEY"

# Force a new ACA revision so the container picks up the new value
az containerapp update --name supportrag-api --resource-group <rg> \
    --set-env-vars API_KEY=secretref:api-key
```

Distribute `$NEW_KEY` to portal integrators. Old key is invalid the
moment the new revision is healthy.

---

## 4. Deployment topology — free-tier production

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Azure Container Apps env                       │
│                                                                      │
│   ┌────────────────────┐    HTTPS    ┌────────────────────────────┐  │
│   │  Support Portal     │ ──────────▶ │  supportrag-api (ACA)     │  │
│   │  (browser / server) │  X-API-Key  │  FastAPI + BGE in image    │  │
│   └────────────────────┘              │  0.5 vCPU · 1 GiB         │  │
│                                       │  minReplicas: 0 → free    │  │
│                                       └──────┬───────────┬─────────┘  │
│                                              │           │            │
│                                  vector search│           │ generation │
│                                              ▼           ▼            │
│                                  ┌─────────────────┐  ┌──────────────┐│
│                                  │ MongoDB Atlas    │  │ HuggingFace   ││
│                                  │ M0 (free)        │  │ Inference API ││
│                                  │ chunk_embeddings │  │ (free tier)   ││
│                                  └─────────────────┘  └──────────────┘│
└──────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ docker push
                  ┌───────────┴───────────┐
                  │ GitHub Container Reg. │
                  │ ghcr.io/nhahub/...    │
                  └───────────────────────┘
```

**What lives where (all on free tiers)**

| Component | Provider | Free-tier cap |
|---|---|---|
| `supportrag-api` Container App | Azure Container Apps | 180k vCPU-sec + 360k GiB-sec free/mo. `minReplicas: 0` keeps idle cost ≈ $0; first request after sleep cold-starts in ~30 s |
| Image registry | GitHub Container Registry | Free for public packages; private packages free up to 500 MB |
| Key Vault `supportragkv*` | Azure Key Vault | 10k operations/mo free; holds `api-key`, `hf-token`, `mongo-uri` |
| Log Analytics `supportrag-logs` | Azure Monitor | 5 GB/mo ingest free |
| Vector store | MongoDB Atlas M0 | 512 MB free (we use ~75 MB) |
| LLM | HuggingFace Inference API | Free serverless; ~30 s cold-start, rate limits apply |

**Selecting the LLM backend.** The image runs whichever provider
`RAG_GENERATION_PROVIDER` selects at startup:
  - `huggingface` (default in cloud) → `HuggingFaceGenerator` reads
    `HF_TOKEN`, `HF_MODEL`. No additional infrastructure.
  - `ollama` (default locally) → original `Generator` talking to
    Ollama on `localhost:11434`.

To move to a paid hosted LLM (Azure OpenAI, Together.ai, etc.), add a
sibling class in `src/rag/` and an `elif` branch in
`generator_factory.py`. No other file changes.

---

## 5. Deployment runbook

### Prereqs (one-time, all free)

1. **Azure** — `az login` with a subscription that has free-tier credit.
2. **MongoDB Atlas** — sign up at `mongodb.com/atlas`, create an M0
   cluster, whitelist `0.0.0.0/0` (or ACA's outbound IPs), copy the
   `mongodb+srv://...` URI.
3. **Populate Atlas with embeddings.** Run *locally*, once:
   ```bash
   MONGO_URI="mongodb+srv://..." python -m src.vector_store.build_store --drop
   ```
   This embeds the 24k chunks with BGE-base and bulk-upserts them.
4. **HuggingFace** — sign up, generate a token at
   `huggingface.co/settings/tokens` (Read access is enough for Inference API).
5. **GitHub Container Registry** — log in to `gh` CLI (`gh auth login`)
   so the deploy script can pull a token for `docker push`. Alternatively
   set `GHCR_TOKEN` to a PAT with `write:packages` scope.

### One-shot deploy

```bash
cd deploy/azure

export MONGO_URI="mongodb+srv://user:pass@cluster.xxxx.mongodb.net/support_rag"
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

./deploy.sh support-rag-rg westeurope
```

The script:
1. Builds the image (forcing `linux/amd64` for ACA compatibility).
2. Pushes to `ghcr.io/nhahub/support-rag-api:latest`.
3. Creates the resource group if missing.
4. Deploys `main.bicep` (Log Analytics, Key Vault, ACA env, container app).
5. Rolls out a new ACA revision pointing at the freshly-pushed image.
6. Prints the API URL, OpenAPI URL, and the generated API key.

### Re-deploy a new code version

```bash
IMAGE_TAG=v2 ./deploy.sh support-rag-rg westeurope
```

ACA holds previous revisions; rollback is a single `az containerapp
revision activate` call.

### Local-only stack (no Azure, no HF)

```bash
export API_KEY=$(openssl rand -hex 32)
docker compose -f deploy/docker-compose.full.yml up --build
# API on http://localhost:8000, OpenAPI on http://localhost:8000/docs
```

This brings up Mongo + Ollama + API together. The first run pulls the
Mistral model (~4 GB) into a named volume; subsequent starts are fast.

---

## 6. Threat model (STRIDE)

| Threat | Surface | Mitigation |
|---|---|---|
| **Spoofing** — caller pretends to be the portal | `/chat`, `/search` | `X-API-Key` constant-time check; missing/wrong header → 401 |
| **Tampering** — request body modified in transit | All endpoints | TLS terminated by ACA ingress; HTTP rejected (`allowInsecure: false`) |
| **Repudiation** — caller denies sending a query | All endpoints | Every request logged to Log Analytics with timestamp + remote IP |
| **Info disclosure** — chunks leak PII | `/chat` (when `include_chunks=true`), `/search` | Source data is anonymised by preprocessing (`{{Customer Name}}` → `[CUSTOMER_NAME]`, etc.). `include_chunks` is opt-in per request |
| **DoS** — LLM is expensive | `/chat` | `top_k` capped at 20, query length capped at 2000 chars, ACA per-revision concurrency cap + autoscale max 3 |
| **Elevation of privilege** | API key compromised | Rotate via §3.Rotation; revoke is instant on new revision |

Out-of-scope (handled elsewhere): WAF (use Front Door if needed), DDoS
absorption (Azure DDoS Standard on the env's vnet), network egress
restrictions (Container Apps egress can be locked to a vnet).

---

## 7. Operations

### Metrics to watch

| Metric | Source | Alert threshold |
|---|---|---|
| `/chat` p95 latency | App logs (`latency_ms`) | > 8 s sustained for 5 min |
| `/chat` 5xx rate | ACA system metrics | > 1 % over 5 min |
| Replica count | ACA system metrics | At max (3) for 10 min — capacity warning |
| Mongo connection failures | App logs | Any |
| Ollama timeouts | App logs (`[TIMEOUT]` answers) | > 5 % of requests |

Querying logs (KQL):

```kql
ContainerAppConsoleLogs_CL
| where ContainerAppName_s == "supportrag-api"
| where Log_s contains "latency_ms"
| project TimeGenerated, Log_s
| order by TimeGenerated desc
```

### Common incidents

| Symptom | Likely cause | Action |
|---|---|---|
| `/health` reports `mongo: down` | Atlas connection string expired / IP firewall | Verify `MONGO_URI` secret; whitelist ACA env outbound IPs in Atlas |
| `/health` reports `ollama: down` | Backend container restarted | Check Ollama logs; the `mistral` model may need re-pulling |
| All `/chat` returning `[TIMEOUT]` | Mistral CPU saturated | Scale Ollama replica or move to Azure OpenAI (see §4) |
| Sudden 401s from portal | Key rotated without distribution | Re-issue current `api-key` secret to portal team |
| New revision crash-loops | Bad image / missing env var | `az containerapp revision deactivate <bad>`, redeploy previous tag |

---

## 8. Future hardening checklist

- [ ] Swap MongoDB local → **MongoDB Atlas Vector Search** (`$vectorSearch` aggregation; one method swap in `mongo_store.vector_search`).
- [ ] Add **Azure AD bearer** auth as an alternative to API key (second dep in `api/auth.py`).
- [ ] Add **per-key rate limiting** (Redis-backed sliding window).
- [ ] Restrict CORS `allow_origins` to the exact portal origin in production (currently `*`).
- [ ] Move from ACA admin-credentials registry pull to **managed identity** + AcrPull role.
- [ ] Add a **request/response audit log** (separate stream from app logs) for support compliance.
- [ ] Wire `/chat` to **MLflow tracking** so prompt/response pairs feed back into MS4 monitoring.

---

## 9. Quick-reference: portal integration

```javascript
async function askSupportBot(question) {
  const r = await fetch("https://supportrag-api.example.azurecontainerapps.io/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-API-Key":    process.env.SUPPORT_RAG_API_KEY,  // server-side only
    },
    body: JSON.stringify({ query: question, top_k: 5 }),
  });
  if (!r.ok) throw new Error(`Support API ${r.status}`);
  const { answer, latency_ms } = await r.json();
  return { answer, latency_ms };
}
```

**Do not** put the API key in browser-side code. Always proxy through your
own server. The expected call site is the portal's backend, not the user's
browser.
