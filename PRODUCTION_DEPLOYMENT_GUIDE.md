# GoldenSignals AI V2 Production Deployment Guide

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Deployment Process](#deployment-process)
4. [Monitoring Setup](#monitoring-setup)
5. [Security Configuration](#security-configuration)
6. [Scaling Guidelines](#scaling-guidelines)
7. [Troubleshooting](#troubleshooting)
8. [Rollback Procedures](#rollback-procedures)

## Prerequisites

### Required Tools
- **kubectl** v1.28+
- **Docker** v20.10+
- **Helm** v3.12+ (optional but recommended)
- **gcloud** CLI (for GKE)
- **aws** CLI (for EKS)
- **az** CLI (for AKS)

### Required Access
- Kubernetes cluster admin access
- Docker registry push access
- DNS management access
- SSL certificate management

### Environment Variables
```bash
# Required secrets
export POSTGRES_PASSWORD="your-secure-password"
export JWT_SECRET="your-jwt-secret"
export OPENAI_API_KEY="your-openai-key"
export BROKER_API_KEY="your-broker-key"
export BROKER_API_SECRET="your-broker-secret"
export GRAFANA_PASSWORD="your-grafana-password"

# Optional configuration
export SLACK_WEBHOOK="your-slack-webhook"
export PAGERDUTY_SERVICE_KEY="your-pagerduty-key"
```

## Infrastructure Setup

### 1. Kubernetes Cluster Setup

#### Google Kubernetes Engine (GKE)
```bash
# Create cluster
gcloud container clusters create goldensignals-prod \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n2-standard-4 \
  --enable-autoscaling \
  --min-nodes 3 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials goldensignals-prod \
  --zone us-central1-a
```

#### Amazon EKS
```bash
# Create cluster
eksctl create cluster \
  --name goldensignals-prod \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.xlarge \
  --nodes 3 \
  --nodes-min 3 \
  --nodes-max 10 \
  --managed
```

#### Azure AKS
```bash
# Create resource group
az group create --name goldensignals-rg --location eastus

# Create cluster
az aks create \
  --resource-group goldensignals-rg \
  --name goldensignals-prod \
  --node-count 3 \
  --vm-set-type VirtualMachineScaleSets \
  --enable-cluster-autoscaler \
  --min-count 3 \
  --max-count 10
```

### 2. Install Ingress Controller
```bash
# Install NGINX Ingress Controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.2/deploy/static/provider/cloud/deploy.yaml

# Install cert-manager for SSL
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.1/cert-manager.yaml
```

### 3. Setup DNS
Configure your DNS provider to point to the ingress controller's external IP:
```bash
# Get ingress external IP
kubectl get svc -n ingress-nginx

# Configure DNS records
goldensignals.ai -> INGRESS_EXTERNAL_IP
api.goldensignals.ai -> INGRESS_EXTERNAL_IP
app.goldensignals.ai -> INGRESS_EXTERNAL_IP
```

## Deployment Process

### 1. Build and Push Docker Images
```bash
# Build all images
make build-all

# Push to registry
make push-all
```

### 2. Deploy to Production
```bash
# Dry run first
./scripts/deploy-production.sh production true

# Actual deployment
./scripts/deploy-production.sh production
```

### 3. Verify Deployment
```bash
# Check all pods are running
kubectl get pods -n goldensignals

# Check services
kubectl get svc -n goldensignals

# Check ingress
kubectl get ingress -n goldensignals

# Run health checks
./scripts/health-check.sh production
```

## Monitoring Setup

### 1. Access Grafana
- URL: https://api.goldensignals.ai/grafana
- Default username: admin
- Password: Set via GRAFANA_PASSWORD environment variable

### 2. Import Dashboards
```bash
# Import pre-configured dashboards
kubectl apply -f k8s/production/grafana-dashboards.yaml
```

### 3. Configure Alerts
1. Access AlertManager: https://api.goldensignals.ai/alertmanager
2. Configure Slack webhook in alertmanager-config
3. Set up PagerDuty integration for critical alerts

### 4. Key Metrics to Monitor
- **Service Health**
  - Request rate
  - Error rate
  - Response time (p50, p95, p99)
  
- **Resource Usage**
  - CPU utilization
  - Memory usage
  - Disk I/O
  
- **Business Metrics**
  - Active users
  - Trading signals generated
  - Order execution success rate
  - RAG query performance

## Security Configuration

### 1. Network Policies
```yaml
# Apply network policies
kubectl apply -f k8s/production/network-policies.yaml
```

### 2. Pod Security Policies
```yaml
# Apply security policies
kubectl apply -f k8s/production/pod-security-policies.yaml
```

### 3. RBAC Configuration
```bash
# Create service accounts
kubectl apply -f k8s/production/rbac.yaml

# Bind roles
kubectl apply -f k8s/production/role-bindings.yaml
```

### 4. Secrets Management
- Use Kubernetes secrets for sensitive data
- Consider using HashiCorp Vault or AWS Secrets Manager
- Rotate secrets regularly

## Scaling Guidelines

### 1. Horizontal Pod Autoscaling
```yaml
# Apply HPA configurations
kubectl apply -f k8s/production/hpa.yaml

# Monitor HPA status
kubectl get hpa -n goldensignals
```

### 2. Service-Specific Scaling

#### Market Data MCP
- Scale based on WebSocket connections
- Recommended: 3-10 replicas
- CPU target: 70%

#### RAG Query MCP
- Scale based on query latency
- Recommended: 2-8 replicas
- Memory target: 80%

#### Execution MCP
- Scale based on order volume
- Recommended: 3-15 replicas
- Response time target: <100ms

### 3. Database Scaling
```bash
# PostgreSQL read replicas
kubectl scale statefulset postgres-read --replicas=3

# Redis clustering
kubectl apply -f k8s/production/redis-cluster.yaml
```

## Troubleshooting

### Common Issues

#### 1. Pods Not Starting
```bash
# Check pod events
kubectl describe pod POD_NAME -n goldensignals

# Check logs
kubectl logs POD_NAME -n goldensignals

# Check resource constraints
kubectl top pods -n goldensignals
```

#### 2. Service Discovery Issues
```bash
# Check endpoints
kubectl get endpoints -n goldensignals

# Test DNS resolution
kubectl run -it --rm debug --image=busybox --restart=Never -- nslookup service-name
```

#### 3. Ingress Not Working
```bash
# Check ingress controller logs
kubectl logs -n ingress-nginx deployment/ingress-nginx-controller

# Verify SSL certificates
kubectl describe certificate -n goldensignals
```

### Debug Commands
```bash
# Enter pod shell
kubectl exec -it POD_NAME -n goldensignals -- /bin/bash

# Port forward for local testing
kubectl port-forward svc/SERVICE_NAME 8080:80 -n goldensignals

# Check cluster events
kubectl get events -n goldensignals --sort-by='.lastTimestamp'
```

## Rollback Procedures

### 1. Quick Rollback
```bash
# Rollback deployment
kubectl rollout undo deployment/DEPLOYMENT_NAME -n goldensignals

# Check rollback status
kubectl rollout status deployment/DEPLOYMENT_NAME -n goldensignals
```

### 2. Full Environment Rollback
```bash
# Use the rollback script
./scripts/rollback-production.sh

# Or manually rollback all deployments
for deployment in $(kubectl get deployments -n goldensignals -o jsonpath='{.items[*].metadata.name}'); do
    kubectl rollout undo deployment/$deployment -n goldensignals
done
```

### 3. Database Rollback
```bash
# Restore from backup
kubectl exec -it postgres-0 -n goldensignals -- psql -U goldensignals -c "
SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = 'goldensignals';
DROP DATABASE goldensignals;
CREATE DATABASE goldensignals;
"

# Restore backup
kubectl exec -i postgres-0 -n goldensignals -- psql -U goldensignals goldensignals < backup.sql
```

## Maintenance Procedures

### 1. Regular Updates
```bash
# Update dependencies
make update-deps

# Rebuild and redeploy
make build-all push-all
./scripts/deploy-production.sh production
```

### 2. Certificate Renewal
Certificates are auto-renewed by cert-manager, but verify:
```bash
kubectl get certificates -n goldensignals
kubectl describe certificate goldensignals-tls -n goldensignals
```

### 3. Backup Procedures
```bash
# Database backup
./scripts/backup-database.sh

# Configuration backup
kubectl get all,cm,secret,ing -n goldensignals -o yaml > backup-config.yaml
```

## Performance Optimization

### 1. Enable Caching
```yaml
# Redis caching is enabled by default
# Adjust cache TTL in ConfigMaps
```

### 2. Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX idx_signals_timestamp ON signals(created_at);
CREATE INDEX idx_orders_status ON orders(status, created_at);
```

### 3. CDN Configuration
```bash
# Configure CloudFlare or AWS CloudFront for static assets
# Update frontend deployment with CDN URLs
```

## Support and Contact

- **Technical Support**: support@goldensignals.ai
- **Emergency Hotline**: +1-555-GOLD-SIG
- **Slack Channel**: #goldensignals-ops
- **Documentation**: https://docs.goldensignals.ai

---

**Last Updated**: December 2024
**Version**: 2.0.0
**Issue**: #196 