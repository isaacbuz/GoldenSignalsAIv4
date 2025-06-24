#!/bin/bash

# Production Deployment Script for GoldenSignals AI V2
# Issue #196: Production Deployment and Monitoring

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="goldensignals"
ENVIRONMENT="${1:-production}"
DRY_RUN="${2:-false}"

echo -e "${BLUE}🚀 GoldenSignals AI V2 Production Deployment${NC}"
echo -e "${BLUE}===========================================${NC}"
echo ""

# Check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}📋 Checking prerequisites...${NC}"
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ kubectl found${NC}"
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Connected to Kubernetes cluster${NC}"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker is not installed${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker found${NC}"
    
    echo ""
}

# Create namespace
create_namespace() {
    echo -e "${YELLOW}📦 Creating namespace...${NC}"
    
    if kubectl get namespace $NAMESPACE &> /dev/null; then
        echo -e "${GREEN}✅ Namespace $NAMESPACE already exists${NC}"
    else
        kubectl create namespace $NAMESPACE
        echo -e "${GREEN}✅ Created namespace $NAMESPACE${NC}"
    fi
    
    echo ""
}

# Create secrets
create_secrets() {
    echo -e "${YELLOW}🔐 Creating secrets...${NC}"
    
    # Check if secrets exist
    if kubectl get secret db-credentials -n $NAMESPACE &> /dev/null; then
        echo -e "${YELLOW}⚠️  Secrets already exist, skipping...${NC}"
    else
        # Database credentials
        kubectl create secret generic db-credentials \
            --from-literal=postgres-url="postgresql://goldensignals:${POSTGRES_PASSWORD}@postgres-service:5432/goldensignals" \
            -n $NAMESPACE
        
        # Broker credentials
        kubectl create secret generic broker-credentials \
            --from-literal=api-key="${BROKER_API_KEY}" \
            --from-literal=api-secret="${BROKER_API_SECRET}" \
            -n $NAMESPACE
        
        # JWT secret
        kubectl create secret generic jwt-secret \
            --from-literal=secret="${JWT_SECRET}" \
            -n $NAMESPACE
        
        # Grafana secret
        kubectl create secret generic grafana-secret \
            --from-literal=admin-password="${GRAFANA_PASSWORD}" \
            -n $NAMESPACE
        
        # OpenAI secret
        kubectl create secret generic openai-secret \
            --from-literal=api-key="${OPENAI_API_KEY}" \
            -n $NAMESPACE
        
        # PostgreSQL secret
        kubectl create secret generic postgres-secret \
            --from-literal=password="${POSTGRES_PASSWORD}" \
            -n $NAMESPACE
        
        echo -e "${GREEN}✅ Secrets created${NC}"
    fi
    
    echo ""
}

# Deploy infrastructure
deploy_infrastructure() {
    echo -e "${YELLOW}🏗️  Deploying infrastructure components...${NC}"
    
    if [ "$DRY_RUN" == "true" ]; then
        kubectl apply -f k8s/production/infrastructure.yaml --dry-run=client
    else
        kubectl apply -f k8s/production/infrastructure.yaml
    fi
    
    echo -e "${GREEN}✅ Infrastructure deployed${NC}"
    echo ""
    
    # Wait for infrastructure to be ready
    echo -e "${YELLOW}⏳ Waiting for infrastructure to be ready...${NC}"
    kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
    kubectl wait --for=condition=ready pod -l app=kafka -n $NAMESPACE --timeout=300s
    echo -e "${GREEN}✅ Infrastructure is ready${NC}"
    echo ""
}

# Deploy services
deploy_services() {
    echo -e "${YELLOW}🚀 Deploying services...${NC}"
    
    if [ "$DRY_RUN" == "true" ]; then
        kubectl apply -f k8s/production/services.yaml --dry-run=client
    else
        kubectl apply -f k8s/production/services.yaml
    fi
    
    echo -e "${GREEN}✅ Services deployed${NC}"
    echo ""
}

# Deploy applications
deploy_applications() {
    echo -e "${YELLOW}📱 Deploying applications...${NC}"
    
    if [ "$DRY_RUN" == "true" ]; then
        kubectl apply -f k8s/production/deployment.yaml --dry-run=client
    else
        kubectl apply -f k8s/production/deployment.yaml
    fi
    
    echo -e "${GREEN}✅ Applications deployed${NC}"
    echo ""
    
    # Wait for deployments to be ready
    echo -e "${YELLOW}⏳ Waiting for deployments to be ready...${NC}"
    for deployment in market-data-mcp rag-query-mcp agent-comm-mcp risk-analytics-mcp execution-mcp api-gateway frontend; do
        kubectl rollout status deployment/$deployment -n $NAMESPACE --timeout=600s
    done
    echo -e "${GREEN}✅ All deployments are ready${NC}"
    echo ""
}

# Deploy monitoring
deploy_monitoring() {
    echo -e "${YELLOW}📊 Deploying monitoring stack...${NC}"
    
    if [ "$DRY_RUN" == "true" ]; then
        kubectl apply -f k8s/production/monitoring.yaml --dry-run=client
    else
        kubectl apply -f k8s/production/monitoring.yaml
    fi
    
    echo -e "${GREEN}✅ Monitoring deployed${NC}"
    echo ""
}

# Deploy ingress
deploy_ingress() {
    echo -e "${YELLOW}🌐 Deploying ingress...${NC}"
    
    if [ "$DRY_RUN" == "true" ]; then
        kubectl apply -f k8s/production/ingress.yaml --dry-run=client
    else
        kubectl apply -f k8s/production/ingress.yaml
    fi
    
    echo -e "${GREEN}✅ Ingress deployed${NC}"
    echo ""
}

# Run health checks
run_health_checks() {
    echo -e "${YELLOW}🏥 Running health checks...${NC}"
    
    # Check pod status
    echo -e "\n${BLUE}Pod Status:${NC}"
    kubectl get pods -n $NAMESPACE
    
    # Check service endpoints
    echo -e "\n${BLUE}Service Endpoints:${NC}"
    kubectl get endpoints -n $NAMESPACE
    
    # Check ingress
    echo -e "\n${BLUE}Ingress Status:${NC}"
    kubectl get ingress -n $NAMESPACE
    
    echo ""
}

# Print access information
print_access_info() {
    echo -e "${GREEN}✨ Deployment Complete!${NC}"
    echo -e "${GREEN}=====================${NC}"
    echo ""
    echo -e "${BLUE}Access Information:${NC}"
    echo -e "  Frontend: https://goldensignals.ai"
    echo -e "  API: https://api.goldensignals.ai"
    echo -e "  Grafana: https://api.goldensignals.ai/grafana"
    echo -e "  Prometheus: https://api.goldensignals.ai/prometheus"
    echo ""
    echo -e "${BLUE}MCP Server Endpoints:${NC}"
    echo -e "  Market Data: https://api.goldensignals.ai/mcp/market-data"
    echo -e "  RAG Query: https://api.goldensignals.ai/mcp/rag-query"
    echo -e "  Agent Comm: https://api.goldensignals.ai/mcp/agent-comm"
    echo -e "  Risk Analytics: https://api.goldensignals.ai/mcp/risk-analytics"
    echo -e "  Execution: https://api.goldensignals.ai/mcp/execution"
    echo ""
    echo -e "${YELLOW}⚠️  Note: DNS propagation may take up to 24 hours${NC}"
}

# Main deployment flow
main() {
    check_prerequisites
    
    if [ "$DRY_RUN" == "true" ]; then
        echo -e "${YELLOW}🔍 Running in DRY RUN mode${NC}"
        echo ""
    fi
    
    # Check for required environment variables
    if [ -z "${POSTGRES_PASSWORD:-}" ] || [ -z "${JWT_SECRET:-}" ] || [ -z "${OPENAI_API_KEY:-}" ]; then
        echo -e "${RED}❌ Required environment variables are not set${NC}"
        echo -e "${YELLOW}Please set: POSTGRES_PASSWORD, JWT_SECRET, OPENAI_API_KEY, etc.${NC}"
        exit 1
    fi
    
    create_namespace
    create_secrets
    deploy_infrastructure
    deploy_services
    deploy_applications
    deploy_monitoring
    deploy_ingress
    run_health_checks
    
    if [ "$DRY_RUN" != "true" ]; then
        print_access_info
    fi
}

# Run main function
main 