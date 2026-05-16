#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
#  One-shot Azure deployment helper.
#
#  Prereqs:
#    - az CLI logged in:    az login
#    - Subscription set:    az account set -s <subscription-id>
#    - Docker running locally for the build/push step
#
#  Usage:
#      ./deploy.sh <resource-group> <location> [<image-tag>]
#
#  Example:
#      ./deploy.sh support-rag-rg westeurope v1
# ────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RG="${1:?resource group name required}"
LOCATION="${2:?azure region required}"
IMAGE_TAG="${3:-latest}"
IMAGE_NAME="support-rag-api"

# Secrets must be provided as env vars — never hard-coded.
: "${MONGO_URI:?MONGO_URI env var required (e.g. mongodb+srv://...)}"
API_KEY="${API_KEY:-$(openssl rand -hex 32)}"

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

echo "▶ Resource group: $RG  ($LOCATION)"
az group create --name "$RG" --location "$LOCATION" --output none

# ── First deploy: just create the ACR so we can push to it ─────────────────
# Subsequent runs reuse the same ACR (idempotent).
echo "▶ Deploying base infra (ACR + Log Analytics + Key Vault + ACA env)..."
DEPLOY_OUT=$(az deployment group create \
    --resource-group "$RG" \
    --template-file "$PROJECT_ROOT/deploy/azure/main.bicep" \
    --parameters \
        mongoUri="$MONGO_URI" \
        apiKey="$API_KEY" \
        imageTag="$IMAGE_TAG" \
    --query "properties.outputs" --output json)

ACR_LOGIN_SERVER=$(echo "$DEPLOY_OUT" | python -c "import json,sys; print(json.load(sys.stdin)['acrLoginServer']['value'])")
API_FQDN=$(echo "$DEPLOY_OUT"        | python -c "import json,sys; print(json.load(sys.stdin)['apiFqdn']['value'])")

echo "▶ Building image..."
docker build -f "$PROJECT_ROOT/deploy/Dockerfile" -t "$IMAGE_NAME:$IMAGE_TAG" "$PROJECT_ROOT"

echo "▶ Pushing to $ACR_LOGIN_SERVER..."
az acr login --name "${ACR_LOGIN_SERVER%%.*}"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "▶ Triggering Container App revision rollout..."
az containerapp update \
    --name "supportrag-api" \
    --resource-group "$RG" \
    --image "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
    --output none

cat <<EOF

✅ Deployed.

  API URL : https://$API_FQDN
  Health  : https://$API_FQDN/health
  Docs    : https://$API_FQDN/docs   (FastAPI OpenAPI UI)

  Send a test request:
    curl -X POST https://$API_FQDN/chat \\
      -H "Content-Type: application/json" \\
      -H "X-API-Key: \$API_KEY" \\
      -d '{"query":"How do I cancel my order?"}'

  Your API key: $API_KEY   (also stored in Key Vault as 'api-key')

EOF
