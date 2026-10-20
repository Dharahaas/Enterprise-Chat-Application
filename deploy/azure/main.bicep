# Enterprise Knowledge Assistant — Azure Deployment (Phase 3)
#
# This stack deploys:
#   - Azure Container Apps for the backend (FastAPI + LangChain + Chroma)
#   - Azure Blob Storage for the frontend static files
#   - Azure CDN for HTTPS + global delivery
#   - Azure Log Analytics for centralized logging
#
# Usage:
#   az deployment group create \
#     --resource-group eka-prod-rg \
#     --template-file deploy/azure/main.bicep \
#     --parameters environmentName=prod openaiApiKey=sk-...
#
# Prerequisites:
#   - az CLI configured: az login
#   - Resource group created: az group create -n eka-prod-rg -l eastus

param environmentName string = 'prod'
param location string = resourceGroup().location
param openaiApiKey string = ''
param langsmithApiKey string = ''
param cpu float = 1.0
param memory string = '2.0Gi'
param minReplicas int = 1
param maxReplicas int = 10

// ---------- Log Analytics Workspace ----------
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${environmentName}-eka-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

// ---------- Container Apps Environment ----------
resource containerAppsEnv 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: '${environmentName}-eka-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: listKeys(logAnalytics.id, '2023-09-01').primarySharedKey
      }
    }
  }
}

// ---------- Storage Account for Frontend ----------
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: toLower('${environmentName}ekafrontend${uniqueString(resourceGroup().id)}')
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: true
    staticWebsite: {
      enabled: true
      indexDocument: 'index.html'
      error404Document: 'index.html'
    }
  }
}

// ---------- Container App: Backend ----------
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: '${environmentName}-eka-backend'
  location: location
  properties: {
    managedEnvironmentId: containerAppsEnv.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        traffic: [{ weight: 100, latestRevision: true }]
        allowInsecure: false
      }
      secrets: [
        { name: 'openai-api-key', value: openaiApiKey }
        { name: 'langsmith-api-key', value: langsmithApiKey }
      ]
      env: [
        { name: 'ENVIRONMENT', value: environmentName }
        { name: 'OPENAI_API_KEY', secretRef: 'openai-api-key' }
        { name: 'LANGSMITH_API_KEY', secretRef: 'langsmith-api-key' }
        { name: 'LANGSMITH_PROJECT', value: '${environmentName}-eka' }
        { name: 'CORS_ORIGINS', value: '*' }
      ]
    }
    template: {
      containers: [
        {
          name: 'eka-backend'
          image: 'mcr.microsoft.com/mcr/placeholder/eka-backend:latest'  // replace with your ACR image
          resources: {
            cpu: cpu
            memory: memory
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/api/health', port: 8000 }
              periodSeconds: 30
              timeoutSeconds: 10
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
            name: 'cpu-based-scaling'
            custom: {
              type: 'cpu'
              metadata: { type: 'Utilization', value: '70' }
            }
          }
        ]
      }
    }
  }
}

// ---------- CDN Profile + Endpoint (Frontend) ----------
resource cdnProfile 'Microsoft.Cdn/profiles@2024-02-01' = {
  name: '${environmentName}-eka-cdn'
  location: location
  sku: { name: 'Standard_Microsoft' }
}

resource cdnEndpoint 'Microsoft.Cdn/profiles/endpoints@2024-02-01' = {
  name: '${environmentName}-eka-endpoint'
  parent: cdnProfile
  location: location
  properties: {
    originHostHeader: storageAccount.properties.primaryEndpoints.blob
    origins: [
      {
        name: 'storage-origin'
        properties: {
          hostName: replace(storageAccount.properties.primaryEndpoints.blob, 'https://', '')
        }
      }
    ]
  }
}

// ---------- Outputs ----------
output backendUrl string = containerApp.properties.configuration.ingress.fqdn
output frontendUrl string = storageAccount.properties.primaryEndpoints.web
output logAnalyticsWorkspaceId string = logAnalytics.id
