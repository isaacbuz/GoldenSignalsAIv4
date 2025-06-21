# Production ML Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the GoldenSignals ML services to production using Docker Compose, Kubernetes, and CI/CD pipelines.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Docker Compose Deployment](#docker-compose-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Monitoring & Observability](#monitoring--observability)
7. [Scaling & Performance](#scaling--performance)
8. [Security Best Practices](#security-best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance & Updates](#maintenance--updates)

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   API Gateway   │────▶│   ML Service    │────▶│   ML Workers    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                         │
         │                       ▼                         ▼
         │              ┌─────────────────┐      ┌─────────────────┐
         └─────────────▶│     Redis       │      │   PostgreSQL    │
                        └─────────────────┘      └─────────────────┘
                                 │                         │
                        ┌─────────────────┐      ┌─────────────────┐
                        │   Prometheus    │      │     Grafana     │
                        └─────────────────┘      └─────────────────┘
```

## Prerequisites

### Local Development
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- kubectl 1.25+
- Helm 3.10+

### Cloud Resources
- Kubernetes cluster (EKS/GKE/AKS)
- Container registry (ECR/GCR/ACR)
- PostgreSQL database
- Redis instance
- Load balancer
- SSL certificates

### Required Secrets
```bash
# Database
DATABASE_URL=postgresql://user:password@host:5432/goldensignals

# Redis
REDIS_URL=redis://redis:6379

# API Keys
SECRET_KEY=your-secret-key
ML_ENCRYPTION_KEY=your-encryption-key
ALPHA_VANTAGE_API_KEY=your-api-key
POLYGON_API_KEY=your-api-key
```

## Docker Compose Deployment

### 1. Build Images

```bash
# Build all services
docker-compose -f docker-compose.prod.ml.yml build

# Build specific service
docker-compose -f docker-compose.prod.ml.yml build ml-service
```

### 2. Start Services

```bash
# Start all services
docker-compose -f docker-compose.prod.ml.yml up -d

# Start with specific scale
docker-compose -f docker-compose.prod.ml.yml up -d --scale ml-worker=4

# View logs
docker-compose -f docker-compose.prod.ml.yml logs -f ml-service
```

### 3. Health Checks

```bash
# Check service health
curl http://localhost:8001/health

# Check worker status
docker-compose -f docker-compose.prod.ml.yml exec ml-worker celery -A src.workers.ml_worker inspect active

# Check metrics
curl http://localhost:9090/metrics
```

## Kubernetes Deployment

### 1. Setup Namespace

```bash
# Create namespace
kubectl apply -f k8s/production/namespace.yaml

# Set default namespace
kubectl config set-context --current --namespace=goldensignals
```

### 2. Deploy Secrets

```bash
# Create secrets from environment
kubectl create secret generic goldensignals-secrets \
  --from-literal=database-url="$DATABASE_URL" \
  --from-literal=redis-url="$REDIS_URL" \
  --from-literal=secret-key="$SECRET_KEY" \
  --from-literal=ml-encryption-key="$ML_ENCRYPTION_KEY"

# Or apply from file
kubectl apply -f k8s/production/secrets.yaml
```

### 3. Deploy Storage

```bash
# Create persistent volumes
kubectl apply -f k8s/production/storage.yaml

# Check PVC status
kubectl get pvc
```

### 4. Deploy Services

```bash
# Deploy ConfigMaps
kubectl apply -f k8s/production/configmaps.yaml

# Deploy ML Service
kubectl apply -f k8s/production/ml-service.yaml

# Deploy ML Workers
kubectl apply -f k8s/production/ml-worker.yaml

# Check deployment status
kubectl rollout status deployment/ml-service
kubectl rollout status deployment/ml-worker
```

### 5. Using Helm

```bash
# Add Helm repository
helm repo add goldensignals https://charts.goldensignals.ai
helm repo update

# Install with default values
helm install goldensignals-ml goldensignals/goldensignals-ml

# Install with custom values
helm install goldensignals-ml goldensignals/goldensignals-ml \
  --values custom-values.yaml \
  --set image.tag=v1.2.3 \
  --set postgresql.auth.password=secretpassword
```

### 6. Verify Deployment

```bash
# Check pods
kubectl get pods -l app=ml-service
kubectl get pods -l app=ml-worker

# Check services
kubectl get svc

# Check logs
kubectl logs -f deployment/ml-service
kubectl logs -f deployment/ml-worker

# Access service
kubectl port-forward svc/ml-service 8001:8001
curl http://localhost:8001/health
```

## CI/CD Pipeline

### 1. GitHub Actions Setup

The pipeline is triggered on:
- Push to main/develop branches
- Pull requests
- Manual workflow dispatch

### 2. Pipeline Stages

1. **Test Stage**
   - Run unit tests
   - Run integration tests
   - Generate coverage reports
   - Run performance benchmarks

2. **Model Validation**
   - Validate model performance
   - Check for model drift
   - Generate validation reports

3. **Build Stage**
   - Build Docker images
   - Push to container registry
   - Generate SBOM
   - Scan for vulnerabilities

4. **Deploy Stage**
   - Deploy to staging (develop branch)
   - Deploy to production (main branch)
   - Run smoke tests
   - Update model registry

### 3. Manual Deployment

```bash
# Deploy to staging
./scripts/deploy_ml_service.sh staging v1.2.3

# Deploy to production
./scripts/deploy_ml_service.sh production v1.2.3
```

## Monitoring & Observability

### 1. Prometheus Metrics

```yaml
# Available metrics
goldensignals_ml_predictions_total
goldensignals_ml_prediction_duration_seconds
goldensignals_ml_model_accuracy
goldensignals_ml_backtest_duration_seconds
goldensignals_ml_worker_tasks_total
```

### 2. Grafana Dashboards

Access Grafana:
```bash
kubectl port-forward svc/grafana 3001:80
# Open http://localhost:3001
# Default: admin/changeme
```

Available dashboards:
- ML Service Overview
- Model Performance
- Worker Queue Status
- Resource Utilization

### 3. Logging

```bash
# View logs
kubectl logs -f deployment/ml-service --tail=100

# View logs with labels
kubectl logs -l app=ml-service --tail=100

# Export logs
kubectl logs deployment/ml-service > ml-service.log
```

### 4. Alerts

Configure alerts in `k8s/monitoring/alerts.yaml`:
- High error rate (>1%)
- Low model accuracy (<50%)
- High response time (>1s)
- Worker queue backlog (>1000)
- Resource exhaustion

## Scaling & Performance

### 1. Horizontal Scaling

```bash
# Manual scaling
kubectl scale deployment ml-service --replicas=5
kubectl scale deployment ml-worker --replicas=10

# Check HPA status
kubectl get hpa
kubectl describe hpa ml-service-hpa
```

### 2. Vertical Scaling

Update resource limits in `values.yaml`:
```yaml
resources:
  mlService:
    limits:
      cpu: 4000m
      memory: 8Gi
```

### 3. Performance Tuning

```yaml
# Worker concurrency
CELERY_WORKER_CONCURRENCY: "8"

# Connection pooling
DATABASE_POOL_SIZE: "20"
DATABASE_MAX_OVERFLOW: "40"

# Redis optimization
REDIS_MAX_CONNECTIONS: "100"
```

## Security Best Practices

### 1. Image Security

```bash
# Scan images
trivy image goldensignals/ml-service:latest

# Sign images
cosign sign goldensignals/ml-service:latest
```

### 2. Network Policies

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-service-netpol
spec:
  podSelector:
    matchLabels:
      app: ml-service
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - port: 8001
```

### 3. RBAC

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ml-service-role
rules:
- apiGroups: [""]
  resources: ["configmaps", "secrets"]
  verbs: ["get", "list"]
```

### 4. Secrets Management

```bash
# Use sealed secrets
kubeseal --format=yaml < secrets.yaml > sealed-secrets.yaml

# Use external secrets operator
kubectl apply -f external-secrets.yaml
```

## Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff**
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name> --previous
```

2. **Service Unavailable**
```bash
kubectl get endpoints
kubectl describe svc ml-service
```

3. **Model Loading Errors**
```bash
kubectl exec -it <pod-name> -- ls -la /app/models
kubectl exec -it <pod-name> -- python -c "import joblib; joblib.load('/app/models/model.pkl')"
```

4. **Database Connection Issues**
```bash
kubectl exec -it <pod-name> -- nc -zv postgres 5432
kubectl exec -it <pod-name> -- psql $DATABASE_URL -c "SELECT 1"
```

### Debug Commands

```bash
# Get pod details
kubectl get pod <pod-name> -o yaml

# Check events
kubectl get events --sort-by='.lastTimestamp'

# Debug container
kubectl debug <pod-name> -it --image=busybox

# Copy files from pod
kubectl cp <pod-name>:/app/logs/error.log ./error.log
```

## Maintenance & Updates

### 1. Model Updates

```bash
# Update model files
kubectl cp new_model.pkl <pod-name>:/app/models/

# Trigger model reload
kubectl exec -it <pod-name> -- kill -HUP 1
```

### 2. Rolling Updates

```bash
# Update image
kubectl set image deployment/ml-service ml-service=goldensignals/ml-service:v1.2.4

# Check rollout status
kubectl rollout status deployment/ml-service

# Rollback if needed
kubectl rollout undo deployment/ml-service
```

### 3. Database Migrations

```bash
# Run migrations
kubectl run --rm -it alembic --image=goldensignals/ml-service:latest \
  --env="DATABASE_URL=$DATABASE_URL" \
  -- alembic upgrade head
```

### 4. Backup & Restore

```bash
# Backup models
kubectl exec -it <pod-name> -- tar -czf /tmp/models-backup.tar.gz /app/models
kubectl cp <pod-name>:/tmp/models-backup.tar.gz ./models-backup.tar.gz

# Backup database
kubectl exec -it postgres-pod -- pg_dump -U goldensignals goldensignals > backup.sql

# Restore database
kubectl exec -i postgres-pod -- psql -U goldensignals goldensignals < backup.sql
```

## Best Practices Summary

1. **Always use health checks and readiness probes**
2. **Implement proper resource limits and requests**
3. **Use horizontal pod autoscaling for dynamic load**
4. **Monitor model performance and drift continuously**
5. **Implement canary deployments for production**
6. **Keep secrets encrypted and rotate regularly**
7. **Use persistent volumes for model storage**
8. **Implement proper logging and monitoring**
9. **Regular backups of models and data**
10. **Test disaster recovery procedures**

## Support

For issues or questions:
- Check logs and events first
- Review monitoring dashboards
- Consult troubleshooting section
- Contact: devops@goldensignals.ai 