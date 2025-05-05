#!/bin/bash
# Deploy to AWS EKS
echo "Deploying GoldenSignalsAI to AWS EKS..."
kubectl apply -f k8s/api-deployment.yaml
