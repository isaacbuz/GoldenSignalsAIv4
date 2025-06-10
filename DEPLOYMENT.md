# GoldenSignalsAI AKS Deployment Guide

This guide explains how to deploy GoldenSignalsAI to Azure Kubernetes Service (AKS).

## Prerequisites

1. Azure CLI installed and configured
2. Terraform installed
3. Helm installed
4. kubectl installed
5. Azure subscription with necessary permissions

## Environment Variables

Set the following environment variables:

```bash
export ALPHA_VANTAGE_API_KEY="your_key"
export NEWS_API_KEY="your_key"
export TWITTER_BEARER_TOKEN="your_token"
export REDIS_PASSWORD="your_password"
export GRAFANA_ADMIN_PASSWORD="your_password"
export SLACK_WEBHOOK_URL="your_webhook_url"
```

## Deployment Steps

1. **Initialize Terraform**

```bash
cd terraform
terraform init
```

2. **Deploy Infrastructure**

```bash
# Deploy using the provided script
./scripts/deploy_aks.sh
```

This script will:
- Create Azure resource group
- Deploy AKS cluster
- Set up Azure Container Registry
- Configure monitoring
- Deploy the application

3. **Verify Deployment**

```bash
# Check pods
kubectl get pods -n goldensignals

# Check services
kubectl get services -n goldensignals

# Check ingress
kubectl get ingress -n goldensignals
```

## Accessing the Application

- Trading Platform: https://trading.goldensignals.ai
- Dashboard: https://dashboard.goldensignals.ai
- Grafana: https://dashboard.goldensignals.ai/grafana

## Monitoring

The deployment includes:
- Prometheus for metrics collection
- Grafana for visualization
- Azure Monitor integration
- Custom dashboards for trading metrics

## Scaling

The application uses Horizontal Pod Autoscaling:
- Backend: 2-5 replicas
- Frontend: 2-4 replicas
- Dashboard: 1 replica (stateful)

## Troubleshooting

1. **Check pod logs**
```bash
kubectl logs -f deployment/goldensignalsai-backend -n goldensignals
```

2. **Check events**
```bash
kubectl get events -n goldensignals
```

3. **Check metrics**
```bash
kubectl top pods -n goldensignals
```

## Maintenance

1. **Update Application**
```bash
# Update using Helm
helm upgrade goldensignals ./helm/goldensignals
```

2. **Rollback**
```bash
# Rollback to previous version
helm rollback goldensignals
```

3. **Backup**
```bash
# Backup persistent volumes
velero backup create goldensignals-backup
```

## Security

- All secrets are stored in Azure Key Vault
- TLS enabled for all ingress endpoints
- Network policies restrict pod communication
- RBAC configured for minimal privileges

## Support

For issues:
1. Check pod logs
2. Check Azure Monitor
3. Review Grafana dashboards
4. Contact DevOps team

## Cleanup

To remove the deployment:
```bash
terraform destroy
``` 