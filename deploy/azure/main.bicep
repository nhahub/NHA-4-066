// ────────────────────────────────────────────────────────────────────────────
//  Azure Container Apps deployment — ZERO-BUDGET variant.
//
//  Cost-conscious choices vs. the original template:
//    - No Azure Container Registry: image is pulled from GitHub Container
//      Registry (ghcr.io/nhahub/support-rag-api).  GHCR is free for any
//      visibility level on a public repo.
//    - LLM is HuggingFace Inference API (free tier) — no Ollama container.
//    - MongoDB is expected to be MongoDB Atlas free M0 (mongodb+srv URI).
//    - Container App is configured with minReplicas: 0 (scale-to-zero)
//      so the ACA bill stays inside the Azure free-tier vCPU-second pool.
//
//  Still provisioned (all free at low volume):
//    - Log Analytics workspace  (ACA requirement; 5 GB/mo free ingest)
//    - Key Vault                (10k ops/mo free)
//    - Container Apps env       (no flat fee)
//    - Container App            (consumption: 180k vCPU-sec + 360k GiB-sec free / mo)
//
//  Deploy from deploy/azure/:
//      ./deploy.sh <resource-group> <region>
//  See deploy.sh for the env vars it expects.
// ────────────────────────────────────────────────────────────────────────────

@description('Azure region (e.g. westeurope, eastus).')
param location string = resourceGroup().location

@description('Short name prefix used for all resources.')
@minLength(3)
@maxLength(12)
param namePrefix string = 'supportrag'

@description('GHCR image, e.g. ghcr.io/nhahub/support-rag-api:latest')
param image string

@description('External MongoDB URI (e.g. MongoDB Atlas). Stored as a Container App secret.')
@secure()
param mongoUri string

@description('HuggingFace Inference API token. Free at huggingface.co/settings/tokens.')
@secure()
param hfToken string

@description('Optional override for the HF model name. Default works for free tier.')
param hfModel string = 'mistralai/Mistral-7B-Instruct-v0.3'

@description('Secret value for X-API-Key. Generate with: openssl rand -hex 32')
@secure()
param apiKey string

@description('Min replica count. 0 = scale-to-zero (free-tier friendly; ~30s cold start).')
@minValue(0)
@maxValue(10)
param minReplicas int = 0

@description('Max replica count for HTTP-scale rule.')
@minValue(1)
@maxValue(10)
param maxReplicas int = 2

@description('If pulling a private GHCR package, set this to a PAT with read:packages.')
@secure()
param ghcrPullToken string = ''

@description('GitHub username/org that owns the GHCR package (used as the registry username).')
param ghcrUsername string = 'nhahub'

// ── Log Analytics (ACA requirement) ─────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${namePrefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Key Vault (holds API key, HF token, Mongo URI) ──────────────────────────
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: toLower('${namePrefix}kv${uniqueString(resourceGroup().id)}')
  location: location
  properties: {
    sku: { family: 'A', name: 'standard' }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enabledForTemplateDeployment: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

resource kvApiKeySecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'api-key'
  properties: { value: apiKey }
}
resource kvHfSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'hf-token'
  properties: { value: hfToken }
}
resource kvMongoSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: keyVault
  name: 'mongo-uri'
  properties: { value: mongoUri }
}

// ── Container Apps environment ──────────────────────────────────────────────
resource acaEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${namePrefix}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// ── Container App ───────────────────────────────────────────────────────────
// If ghcrPullToken is provided we configure GHCR as a private registry;
// otherwise we assume the package is public and skip credentials entirely.
var useGhcrAuth = !empty(ghcrPullToken)

resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${namePrefix}-api'
  location: location
  properties: {
    managedEnvironmentId: acaEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        allowInsecure: false
        traffic: [
          { weight: 100, latestRevision: true }
        ]
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'OPTIONS']
          allowedHeaders: ['X-API-Key', 'Content-Type']
          maxAge: 600
        }
      }
      registries: useGhcrAuth ? [
        {
          server: 'ghcr.io'
          username: ghcrUsername
          passwordSecretRef: 'ghcr-pull-token'
        }
      ] : []
      secrets: union(
        [
          { name: 'api-key',   value: apiKey   }
          { name: 'mongo-uri', value: mongoUri }
          { name: 'hf-token',  value: hfToken  }
        ],
        useGhcrAuth ? [ { name: 'ghcr-pull-token', value: ghcrPullToken } ] : []
      )
    }
    template: {
      containers: [
        {
          name: 'api'
          image: image
          // 0.5 vCPU / 1 GiB keeps us well inside ACA's free vCPU-seconds.
          // Bump to 1.0/2Gi if BGE-base feels slow.
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            { name: 'API_KEY',                  secretRef: 'api-key' }
            { name: 'MONGO_URI',                secretRef: 'mongo-uri' }
            { name: 'HF_TOKEN',                 secretRef: 'hf-token' }
            { name: 'HF_MODEL',                 value: hfModel }
            { name: 'RAG_GENERATION_PROVIDER',  value: 'huggingface' }
            { name: 'RAG_CONFIG',               value: 'config/config.yaml' }
            { name: 'LOG_LEVEL',                value: 'INFO' }
            { name: 'WORKERS',                  value: '1' }
            { name: 'FORCE_CPU',                value: '1' }
            { name: 'CORS_ORIGINS',             value: '*' }
            { name: 'DEFAULT_TOP_K',            value: '5' }
          ]
          probes: [
            {
              type: 'liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 60
              periodSeconds: 30
              timeoutSeconds: 5
              failureThreshold: 3
            }
            {
              type: 'readiness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 30
              periodSeconds: 15
              timeoutSeconds: 5
              failureThreshold: 3
            }
          ]
        }
      ]
      scale: {
        minReplicas: minReplicas
        maxReplicas: maxReplicas
        rules: [
          {
            name: 'http-scale'
            http: {
              metadata: {
                concurrentRequests: '10'
              }
            }
          }
        ]
      }
    }
  }
}

// ── Outputs ─────────────────────────────────────────────────────────────────
output apiFqdn         string = api.properties.configuration.ingress.fqdn
output keyVaultName    string = keyVault.name
output logAnalyticsId  string = logAnalytics.id
