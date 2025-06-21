#!/bin/bash
# ML Service Deployment Script

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}
NAMESPACE="goldensignals"
REGISTRY="ghcr.io/goldensignals"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl not found. Please install kubectl."
        exit 1
    fi
    
    # Check helm
    if ! command -v helm &> /dev/null; then
        log_error "helm not found. Please install helm."
        exit 1
    fi
    
    # Check cluster connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create namespace if not exists
create_namespace() {
    if ! kubectl get namespace $NAMESPACE &> /dev/null; then
        log_info "Creating namespace $NAMESPACE..."
        kubectl create namespace $NAMESPACE
        kubectl label namespace $NAMESPACE environment=$ENVIRONMENT
    fi
}

# Deploy secrets
deploy_secrets() {
    log_info "Deploying secrets..."
    
    # Check if secrets exist
    if ! kubectl get secret goldensignals-secrets -n $NAMESPACE &> /dev/null; then
        log_warning "Secrets not found. Creating from environment variables..."
        
        kubectl create secret generic goldensignals-secrets \
            --from-literal=database-url="$DATABASE_URL" \
            --from-literal=redis-url="$REDIS_URL" \
            --from-literal=secret-key="$SECRET_KEY" \
            --from-literal=ml-encryption-key="$ML_ENCRYPTION_KEY" \
            -n $NAMESPACE
    fi
    
    log_success "Secrets deployed"
}

# Deploy storage
deploy_storage() {
    log_info "Deploying storage configurations..."
    kubectl apply -f k8s/production/storage.yaml
    
    # Wait for PVCs to be bound
    log_info "Waiting for PVCs to be bound..."
    kubectl wait --for=condition=Bound pvc/ml-models-pvc -n $NAMESPACE --timeout=300s
    
    log_success "Storage deployed"
}

# Deploy ConfigMaps
deploy_configmaps() {
    log_info "Deploying ConfigMaps..."
    kubectl apply -f k8s/production/configmaps.yaml
    log_success "ConfigMaps deployed"
}

# Deploy ML Service
deploy_ml_service() {
    log_info "Deploying ML Service..."
    
    # Update image tag
    sed -i.bak "s|image: goldensignals/ml-service:.*|image: $REGISTRY/ml-service:$VERSION|g" \
        k8s/production/ml-service.yaml
    
    kubectl apply -f k8s/production/ml-service.yaml
    
    # Wait for deployment
    log_info "Waiting for ML Service deployment..."
    kubectl rollout status deployment/ml-service -n $NAMESPACE --timeout=600s
    
    log_success "ML Service deployed"
}

# Deploy ML Worker
deploy_ml_worker() {
    log_info "Deploying ML Worker..."
    
    # Update image tag
    sed -i.bak "s|image: goldensignals/ml-worker:.*|image: $REGISTRY/ml-worker:$VERSION|g" \
        k8s/production/ml-worker.yaml
    
    kubectl apply -f k8s/production/ml-worker.yaml
    
    # Wait for deployment
    log_info "Waiting for ML Worker deployment..."
    kubectl rollout status deployment/ml-worker -n $NAMESPACE --timeout=600s
    
    log_success "ML Worker deployed"
}

# Deploy monitoring
deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Add Prometheus Helm repo
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace $NAMESPACE \
        --values helm/monitoring/prometheus-values.yaml \
        --wait
    
    # Install Grafana dashboards
    kubectl apply -f k8s/monitoring/grafana-dashboards.yaml
    
    log_success "Monitoring deployed"
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    # Check ML Service
    ML_SERVICE_POD=$(kubectl get pod -n $NAMESPACE -l app=ml-service -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec -n $NAMESPACE $ML_SERVICE_POD -- curl -f http://localhost:8001/health; then
        log_success "ML Service health check passed"
    else
        log_error "ML Service health check failed"
        return 1
    fi
    
    # Check ML Worker
    ML_WORKER_POD=$(kubectl get pod -n $NAMESPACE -l app=ml-worker -o jsonpath='{.items[0].metadata.name}')
    if kubectl exec -n $NAMESPACE $ML_WORKER_POD -- celery -A src.workers.ml_worker inspect ping; then
        log_success "ML Worker health check passed"
    else
        log_error "ML Worker health check failed"
        return 1
    fi
    
    return 0
}

# Run smoke tests
run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Get service endpoint
    if [ "$ENVIRONMENT" = "production" ]; then
        ENDPOINT="https://api.goldensignals.ai"
    else
        ENDPOINT=$(kubectl get service ml-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
        ENDPOINT="http://$ENDPOINT:8001"
    fi
    
    # Test endpoints
    log_info "Testing ML Service endpoints..."
    
    # Health check
    if curl -f "$ENDPOINT/health"; then
        log_success "Health endpoint OK"
    else
        log_error "Health endpoint failed"
        return 1
    fi
    
    # Backtest endpoint
    if curl -f -X POST "$ENDPOINT/api/v1/backtest" \
        -H "Content-Type: application/json" \
        -d '{"symbol": "AAPL", "start_date": "2023-01-01", "end_date": "2023-12-31"}'; then
        log_success "Backtest endpoint OK"
    else
        log_error "Backtest endpoint failed"
        return 1
    fi
    
    return 0
}

# Main deployment function
main() {
    log_info "Starting ML Service deployment to $ENVIRONMENT..."
    
    check_prerequisites
    create_namespace
    deploy_secrets
    deploy_storage
    deploy_configmaps
    deploy_ml_service
    deploy_ml_worker
    
    if [ "$ENVIRONMENT" = "production" ]; then
        deploy_monitoring
    fi
    
    # Wait for services to be ready
    sleep 30
    
    if run_health_checks; then
        log_success "Health checks passed"
    else
        log_error "Health checks failed"
        exit 1
    fi
    
    if run_smoke_tests; then
        log_success "Smoke tests passed"
    else
        log_error "Smoke tests failed"
        exit 1
    fi
    
    log_success "ML Service deployment completed successfully!"
    
    # Display service information
    echo ""
    log_info "Service Information:"
    kubectl get services -n $NAMESPACE
    echo ""
    log_info "Pods:"
    kubectl get pods -n $NAMESPACE
    echo ""
    log_info "Access Grafana at: http://localhost:3001 (kubectl port-forward -n $NAMESPACE svc/prometheus-grafana 3001:80)"
}

# Run main function
main "$@" 