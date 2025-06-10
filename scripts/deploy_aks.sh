#!/bin/bash

# Exit on error
set -e

# Variables
RESOURCE_GROUP="goldensignals-rg"
CLUSTER_NAME="goldensignals-aks"
ACR_NAME="goldensignalsacr"
LOCATION="eastus"
CERT_MANAGER_VERSION="v1.11.0"
AGIC_VERSION="1.7.0"
CDN_PROFILE="goldensignals-cdn"
CDN_ENDPOINT="goldensignals-cdn-endpoint"

# Colors for output
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Starting GoldenSignalsAI deployment to AKS...${NC}"

# Login to Azure (if not already logged in)
echo -e "${GREEN}Checking Azure login...${NC}"
az account show > /dev/null 2>&1 || az login

# Create resource group if it doesn't exist
echo -e "${GREEN}Creating resource group...${NC}"
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create AKS cluster using Terraform
echo -e "${GREEN}Applying Terraform configuration...${NC}"
cd terraform
terraform init
terraform apply -auto-approve

# Get AKS credentials
echo -e "${GREEN}Getting AKS credentials...${NC}"
az aks get-credentials --resource-group $RESOURCE_GROUP --name $CLUSTER_NAME --overwrite-existing

# Install AGIC
echo -e "${GREEN}Installing Application Gateway Ingress Controller...${NC}"
helm repo add application-gateway-kubernetes-ingress https://appgwingress.blob.core.windows.net/ingress-azure-helm-package/
helm repo update

helm upgrade --install agic application-gateway-kubernetes-ingress/ingress-azure \
    --namespace agic \
    --create-namespace \
    --set appgw.name=goldensignals-agw \
    --set appgw.resourceGroup=$RESOURCE_GROUP \
    --set appgw.subscriptionId=$(az account show --query id -o tsv) \
    --set appgw.usePrivateIP=false \
    --set appgw.shared=false \
    --set armAuth.type=aadPodIdentity \
    --set armAuth.identityResourceID=$(terraform output -raw agw_identity_id) \
    --set armAuth.identityClientID=$(terraform output -raw agw_identity_client_id)

# Create frontend optimization ConfigMap
echo -e "${GREEN}Creating frontend optimization ConfigMap...${NC}"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: frontend-config
  namespace: goldensignals
data:
  nginx.conf: |
    worker_processes auto;
    events {
      worker_connections 1024;
    }
    http {
      gzip on;
      gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;
      gzip_min_length 1000;
      gzip_proxied any;
      
      client_max_body_size 100M;
      client_body_buffer_size 128k;
      
      proxy_buffer_size 128k;
      proxy_buffers 4 256k;
      proxy_busy_buffers_size 256k;
      
      fastcgi_buffers 16 16k;
      fastcgi_buffer_size 32k;
      
      # Browser cache - static files
      location ~* \.(jpg|jpeg|png|gif|ico|css|js|woff|woff2)$ {
        expires 7d;
        add_header Cache-Control "public, no-transform";
      }
      
      # Security headers
      add_header X-Content-Type-Options "nosniff" always;
      add_header X-Frame-Options "SAMEORIGIN" always;
      add_header X-XSS-Protection "1; mode=block" always;
      add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
      add_header Content-Security-Policy "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:;" always;
      
      # Enable HTTP/2
      listen 443 ssl http2;
      
      # OCSP Stapling
      ssl_stapling on;
      ssl_stapling_verify on;
      resolver 8.8.8.8 8.8.4.4 valid=300s;
      resolver_timeout 5s;
    }
EOF

# Install cert-manager
echo -e "${GREEN}Installing cert-manager...${NC}"
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/${CERT_MANAGER_VERSION}/cert-manager.yaml

# Wait for cert-manager to be ready
echo -e "${GREEN}Waiting for cert-manager to be ready...${NC}"
kubectl wait --for=condition=ready pod -l app.kubernetes.io/instance=cert-manager -n cert-manager --timeout=120s

# Create ClusterIssuer for Let's Encrypt
echo -e "${GREEN}Creating Let's Encrypt ClusterIssuer...${NC}"
cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: admin@goldensignals.ai
    privateKeySecretRef:
      name: letsencrypt-private-key
    solvers:
    - http01:
        ingress:
          class: azure-application-gateway
EOF

# Create namespaces
echo -e "${GREEN}Creating Kubernetes namespaces...${NC}"
kubectl create namespace goldensignals --dry-run=client -o yaml | kubectl apply -f -
kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Label namespaces for network policies
echo -e "${GREEN}Labeling namespaces...${NC}"
kubectl label namespace monitoring name=monitoring --overwrite
kubectl label namespace agic name=agic --overwrite

# Create secrets from environment variables
echo -e "${GREEN}Creating Kubernetes secrets...${NC}"
kubectl create secret generic api-secrets \
  --namespace goldensignals \
  --from-literal=alpha-vantage-api-key="$ALPHA_VANTAGE_API_KEY" \
  --from-literal=news-api-key="$NEWS_API_KEY" \
  --from-literal=twitter-bearer-token="$TWITTER_BEARER_TOKEN" \
  --dry-run=client -o yaml | kubectl apply -f -

# Apply network policies
echo -e "${GREEN}Applying network policies...${NC}"
kubectl apply -f ../k8s/network-policies.yaml

# Install Prometheus and Grafana using Helm
echo -e "${GREEN}Installing monitoring stack...${NC}"
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  -f ../k8s/monitoring/prometheus-values.yaml

# Deploy application
echo -e "${GREEN}Deploying application...${NC}"
helm upgrade --install goldensignals ../helm/goldensignals \
  --namespace goldensignals \
  --set global.environment=production \
  --set global.imageRegistry=$ACR_NAME.azurecr.io

# Wait for deployments to be ready
echo -e "${GREEN}Waiting for deployments to be ready...${NC}"
kubectl wait --for=condition=available --timeout=300s deployment/goldensignalsai-backend -n goldensignals
kubectl wait --for=condition=available --timeout=300s deployment/goldensignalsai-frontend -n goldensignals
kubectl wait --for=condition=available --timeout=300s deployment/goldensignalsai-dashboard -n goldensignals

# Configure Azure Monitor
echo -e "${GREEN}Configuring Azure Monitor...${NC}"
az monitor diagnostic-settings create \
  --name aks-diagnostics \
  --resource $CLUSTER_NAME \
  --resource-group $RESOURCE_GROUP \
  --resource-type Microsoft.ContainerService/managedClusters \
  --workspace $(terraform output -raw log_analytics_workspace_id) \
  --logs '[{"category": "kube-apiserver","enabled": true},{"category": "kube-audit","enabled": true}]' \
  --metrics '[{"category": "AllMetrics","enabled": true}]'

# Verify Private Link connections
echo -e "${GREEN}Verifying Private Link connections...${NC}"
az network private-endpoint-connection list \
  --resource-group $RESOURCE_GROUP \
  --name $ACR_NAME \
  --type Microsoft.ContainerRegistry/registries

az network private-endpoint-connection list \
  --resource-group $RESOURCE_GROUP \
  --name goldensignals-kv \
  --type Microsoft.KeyVault/vaults

# Purge CDN cache
echo -e "${GREEN}Purging CDN cache...${NC}"
az cdn endpoint purge \
  --resource-group $RESOURCE_GROUP \
  --profile-name $CDN_PROFILE \
  --name $CDN_ENDPOINT \
  --content-paths "/*"

# Pre-warm CDN cache for critical paths
echo -e "${GREEN}Pre-warming CDN cache...${NC}"
ENDPOINTS=(
  "/static/css/main.css"
  "/static/js/main.js"
  "/static/js/vendor.js"
  "/static/images/logo.svg"
  "/favicon.ico"
)

for endpoint in "${ENDPOINTS[@]}"; do
  curl -s -o /dev/null -w "%{http_code}" "https://trading.goldensignals.ai${endpoint}"
done

# Get service endpoints
echo -e "${GREEN}Getting service endpoints...${NC}"
echo "Application Gateway Public IP: $(terraform output -raw agw_public_ip)"
echo "CDN Endpoint: $(terraform output -raw cdn_endpoint_hostname)"
echo "Application URLs:"
echo "Frontend: https://trading.goldensignals.ai"
echo "Dashboard: https://dashboard.goldensignals.ai"
echo "Grafana: https://dashboard.goldensignals.ai/grafana"

echo -e "${GREEN}Deployment complete!${NC}"

# Display cluster information
echo -e "${GREEN}Cluster Information:${NC}"
kubectl cluster-info
kubectl get nodes
kubectl get pods -A
kubectl get services -A
kubectl get networkpolicies -A

# Display security information
echo -e "${GREEN}Security Information:${NC}"
echo "Network Policies: Enabled"
echo "Pod Security Policies: Enabled"
echo "Azure Policy: Enabled"
echo "TLS: Enabled with Let's Encrypt"
echo "WAF: Enabled (Prevention Mode)"
echo "Private Link: Enabled for ACR and Key Vault"
echo "AGIC: Enabled"
echo "CDN: Enabled with Premium Verizon"

# Display monitoring URLs
echo -e "${GREEN}Monitoring URLs:${NC}"
echo "Azure Monitor: https://portal.azure.com/#resource/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.ContainerService/managedClusters/$CLUSTER_NAME/monitoring"
echo "Application Gateway WAF: https://portal.azure.com/#resource/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Network/applicationGateways/goldensignals-agw/wafConfiguration"
echo "CDN Analytics: https://portal.azure.com/#resource/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Cdn/profiles/$CDN_PROFILE/endpoints/$CDN_ENDPOINT/analytics"
echo "Grafana: https://dashboard.goldensignals.ai/grafana"
echo "Prometheus: https://dashboard.goldensignals.ai/prometheus"
echo "Alert Manager: https://dashboard.goldensignals.ai/alertmanager" 