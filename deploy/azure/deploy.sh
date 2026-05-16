#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
#  Zero-budget Azure deploy helper.
#
#  What this does:
#    1. Builds the Docker image from deploy/Dockerfile
#    2. Pushes it to GitHub Container Registry (free)
#    3. Deploys the Bicep template (Log Analytics, Key Vault, ACA env, app)
#    4. Rolls out a new revision pointing at the freshly-pushed image
#
#  Prereqs:
#    - az CLI logged in:           az login   &&   az account set -s <sub-id>
#    - Docker daemon running:      docker ps
#    - gh CLI logged in:           gh auth login   (used to derive GHCR_TOKEN
#      if you don't pass one yourself)
#
#  Required env vars:
#    MONGO_URI   — MongoDB Atlas connection string (mongodb+srv://...)
#    HF_TOKEN    — HuggingFace Inference API token
#
#  Optional env vars:
#    API_KEY            — defaults to a fresh openssl-generated value
#    GHCR_USERNAME      — defaults to 'nhahub'
#    GHCR_TOKEN         — read:packages/write:packages PAT. If unset, we
#                         try `gh auth token` (works if gh CLI is logged in).
#    GHCR_PULL_TOKEN    — separate read-only PAT for ACA to pull the image.
#                         Defaults to GHCR_TOKEN. Leave empty to make the
#                         package public and skip auth.
#    GHCR_IMAGE_NAME    — defaults to 'support-rag-api'
#    IMAGE_TAG          — defaults to 'latest'
#
#  Usage:
#      ./deploy.sh <resource-group> <region>
#
#  Example:
#      MONGO_URI=mongodb+srv://... HF_TOKEN=hf_... \
#        ./deploy.sh support-rag-rg westeurope
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RG="${1:?resource group name required}"
LOCATION="${2:?azure region required}"

GHCR_USERNAME="${GHCR_USERNAME:-nhahub}"
GHCR_IMAGE_NAME="${GHCR_IMAGE_NAME:-support-rag-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

: "${MONGO_URI:?MONGO_URI env var required (e.g. mongodb+srv://...)}"
: "${HF_TOKEN:?HF_TOKEN env var required (huggingface.co/settings/tokens)}"

# Fall back to `gh auth token` if GHCR_TOKEN isn't supplied.
if [[ -z "${GHCR_TOKEN:-}" ]]; then
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    GHCR_TOKEN="$(gh auth token)"
  else
    echo "ERROR: GHCR_TOKEN env var not set and 'gh auth token' unavailable." >&2
    echo "       Create a PAT with write:packages at github.com/settings/tokens" >&2
    exit 1
  fi
fi

GHCR_PULL_TOKEN="${GHCR_PULL_TOKEN:-$GHCR_TOKEN}"

API_KEY="${API_KEY:-$(openssl rand -hex 32)}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
FULL_IMAGE="ghcr.io/${GHCR_USERNAME}/${GHCR_IMAGE_NAME}:${IMAGE_TAG}"

echo "▶ Target:"
echo "    Resource group:  $RG ($LOCATION)"
echo "    Image:           $FULL_IMAGE"
echo "    GHCR user/org:   $GHCR_USERNAME"
echo

# ── 1. Build the image ─────────────────────────────────────────────────────
echo "▶ Building image from $PROJECT_ROOT/deploy/Dockerfile ..."
# Apple Silicon → force linux/amd64 so ACA can pull it (otherwise you get
# 'no matching manifest for linux/amd64' at runtime).
docker build \
  --platform linux/amd64 \
  -f "$PROJECT_ROOT/deploy/Dockerfile" \
  -t "$FULL_IMAGE" \
  "$PROJECT_ROOT"

# ── 2. Push to GHCR ────────────────────────────────────────────────────────
echo "▶ Logging in to ghcr.io ..."
echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin

echo "▶ Pushing $FULL_IMAGE ..."
docker push "$FULL_IMAGE"

# ── 3. Provision / update Azure infra ───────────────────────────────────────
echo "▶ Ensuring resource group $RG exists ..."
az group create --name "$RG" --location "$LOCATION" --output none

echo "▶ Deploying main.bicep ..."
DEPLOY_OUT=$(az deployment group create \
    --resource-group "$RG" \
    --template-file "$PROJECT_ROOT/deploy/azure/main.bicep" \
    --parameters \
        image="$FULL_IMAGE" \
        mongoUri="$MONGO_URI" \
        hfToken="$HF_TOKEN" \
        apiKey="$API_KEY" \
        ghcrUsername="$GHCR_USERNAME" \
        ghcrPullToken="$GHCR_PULL_TOKEN" \
    --query "properties.outputs" --output json)

API_FQDN=$(echo "$DEPLOY_OUT" | python -c "import json,sys; print(json.load(sys.stdin)['apiFqdn']['value'])")
KV_NAME=$(echo "$DEPLOY_OUT"  | python -c "import json,sys; print(json.load(sys.stdin)['keyVaultName']['value'])")

# ── 4. Force a fresh revision so the new image is picked up ───────────────
echo "▶ Triggering a fresh revision ..."
az containerapp update \
    --name "supportrag-api" \
    --resource-group "$RG" \
    --image "$FULL_IMAGE" \
    --output none

cat <<EOF

✅ Deployed.

  API URL  : https://$API_FQDN
  Health   : https://$API_FQDN/health
  OpenAPI  : https://$API_FQDN/docs

  Test it:
    curl -X POST https://$API_FQDN/chat \\
      -H "Content-Type: application/json" \\
      -H "X-API-Key: \$API_KEY" \\
      -d '{"query":"How do I cancel my order?"}'

  API key  : $API_KEY
             (also stored in Key Vault '$KV_NAME' as secret 'api-key')

  Notes:
    - minReplicas=0 → first request after idle warms up in ~30s.
    - HF Inference is cold-start lazy too; the first /chat may take 20-30s
      extra while Mistral loads on HF's side.
    - All resources are in free / consumption tiers. Watch:
        az consumption usage list --start-date \$(date -v-7d +%F)

EOF
