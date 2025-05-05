#!/bin/bash
# Deploy to Azure AKS
echo "Deploying GoldenSignalsAI to Azure AKS..."
kubectl apply -f k8s/api-deployment.yaml
