// ────────────────────────────────────────────────────────────────────────────
//  Azure Container Apps deployment for the Support RAG API.
//
//  Provisions:
//    - Log Analytics workspace (required by ACA)
//    - Container Apps environment
//    - Azure Container Registry (private)
//    - Container App: support-rag-api (1 replica, scale 1–3 on HTTP)
//    - Azure Key Vault to hold the API key
//
//  Out of scope (deliberately): the Mongo + Ollama backends. In production
//  swap to MongoDB Atlas (vector search) and either bundle Ollama in a
//  second container app or call Azure OpenAI. See docs/INTEGRATION_SECURITY.md.
//
//  Deploy from deploy/azure/:
//      az group create --name $RG --location $LOCATION
//      az deployment group create \
//        --resource-group $RG \
//        --template-file main.bicep \
//        --parameters apiKey=$(openssl rand -hex 32)
// ────────────────────────────────────────────────────────────────────────────

@description('Azure region (e.g. westeurope, eastus).')
param location string = resourceGroup().location

@description('Short name prefix used for all resources.')
@minLength(3)
@maxLength(12)
param namePrefix string = 'supportrag'

@description('Container image — pushed to the ACR created here.')
param imageName string = 'support-rag-api'

@description('Image tag to deploy.')
param imageTag string = 'latest'

@description('External MongoDB URI (e.g. MongoDB Atlas). Stored as a Container App secret.')
@secure()
param mongoUri string

@description('Ollama URL the API should hit (e.g. internal Container Apps DNS, or Azure OpenAI shim).')
param ollamaUrl string = 'http://ollama:11434/api/generate'

@description('Secret value for X-API-Key. Generate with: openssl rand -hex 32')
@secure()
param apiKey string

@description('Min replica count (1 keeps cold-starts away for chatbots).')
@minValue(0)
@maxValue(10)
param minReplicas int = 1

@description('Max replica count for HTTP-scale rule.')
@minValue(1)
@maxValue(30)
param maxReplicas int = 3

// ── Log Analytics (ACA requirement) ─────────────────────────────────────────
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${namePrefix}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ── Azure Container Registry ────────────────────────────────────────────────
resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: toLower('${namePrefix}acr${uniqueString(resourceGroup().id)}')
  location: location
  sku: { name: 'Basic' }
  properties: {
    adminUserEnabled: true
  }
}

// ── Key Vault (holds the API key as a secret) ───────────────────────────────
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
  properties: {
    value: apiKey
  }
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
      registries: [
        {
          server: acr.properties.loginServer
          username: acr.listCredentials().username
          passwordSecretRef: 'acr-password'
        }
      ]
      secrets: [
        { name: 'api-key',       value: apiKey }
        { name: 'mongo-uri',     value: mongoUri }
        { name: 'acr-password',  value: acr.listCredentials().passwords[0].value }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: '${acr.properties.loginServer}/${imageName}:${imageTag}'
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          env: [
            { name: 'API_KEY',           secretRef: 'api-key' }
            { name: 'MONGO_URI',         secretRef: 'mongo-uri' }
            { name: 'OLLAMA_URL',        value: ollamaUrl }
            { name: 'RAG_CONFIG',        value: 'config/config.yaml' }
            { name: 'LOG_LEVEL',         value: 'INFO' }
            { name: 'WORKERS',           value: '1' }
            { name: 'FORCE_CPU',         value: '1' }
            { name: 'CORS_ORIGINS',      value: '*' }
            { name: 'DEFAULT_TOP_K',     value: '5' }
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
                concurrentRequests: '20'
              }
            }
          }
        ]
      }
    }
  }
}

// ── Outputs ─────────────────────────────────────────────────────────────────
output apiFqdn          string = api.properties.configuration.ingress.fqdn
output acrLoginServer   string = acr.properties.loginServer
output keyVaultName     string = keyVault.name
output logAnalyticsId   string = logAnalytics.id
