# GoldenSignalsAI V2 Deployment Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development Setup](#local-development-setup)
3. [Production Deployment](#production-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Provider Deployments](#cloud-provider-deployments)
7. [Configuration Management](#configuration-management)
8. [Database Setup](#database-setup)
9. [Monitoring & Logging](#monitoring--logging)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Python**: 3.9 or higher
- **Node.js**: 16.x or higher
- **RAM**: Minimum 4GB, 8GB+ recommended
- **Storage**: 10GB+ free space
- **CPU**: 2+ cores recommended

### Required Software

```bash
# Python and pip
python --version  # Should be 3.9+
pip --version

# Node.js and npm
node --version   # Should be 16.x+
npm --version

# Docker (for containerized deployment)
docker --version
docker-compose --version

# Git
git --version
```

### API Keys Required

Create accounts and obtain API keys for:
- Alpha Vantage (optional): https://www.alphavantage.co/
- IEX Cloud (optional): https://iexcloud.io/
- Polygon.io (optional): https://polygon.io/
- Finnhub (optional): https://finnhub.io/

## Local Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/GoldenSignalsAI_V2.git
cd GoldenSignalsAI_V2
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-test.txt  # For development
```

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Build frontend
npm run build

# Return to root
cd ..
```

### 4. Environment Configuration

Create `.env` file in the root directory:

```bash
# Copy example environment file
cp env.example .env
```

Edit `.env` with your configuration:

```env
# Application Settings
DEBUG=False
APP_NAME=GoldenSignalsAI
VERSION=2.0.0
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=sqlite:///data/goldensignals.db
# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost:5432/goldensignals

# Redis (optional)
REDIS_URL=redis://localhost:6379/0

# API Keys (optional - for fallback data sources)
ALPHA_VANTAGE_API_KEY=your-key-here
IEX_CLOUD_API_KEY=your-key-here
POLYGON_API_KEY=your-key-here
FINNHUB_API_KEY=your-key-here

# Email Configuration (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
EMAIL_FROM=noreply@goldensignals.ai

# Monitoring
SENTRY_DSN=your-sentry-dsn  # Optional
```

### 5. Database Initialization

```bash
# Create database directory
mkdir -p data

# Initialize database
python -c "from src.services.signal_monitoring_service import SignalMonitoringService; SignalMonitoringService()"

# For PostgreSQL setup:
createdb goldensignals
python manage.py migrate  # If using Django
```

### 6. Start Development Server

```bash
# Start backend
python standalone_backend_optimized.py

# In another terminal, start frontend (if developing frontend)
cd frontend
npm run dev
```

## Production Deployment

### 1. System Preparation

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv nginx supervisor postgresql redis-server

# Create application user
sudo useradd -m -s /bin/bash goldensignals
sudo su - goldensignals
```

### 2. Application Setup

```bash
# Clone repository
git clone https://github.com/yourusername/GoldenSignalsAI_V2.git
cd GoldenSignalsAI_V2

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install production dependencies
pip install -r requirements.txt
pip install gunicorn

# Build frontend
cd frontend
npm install --production
npm run build
cd ..
```

### 3. Gunicorn Configuration

Create `gunicorn_config.py`:

```python
bind = "127.0.0.1:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
timeout = 30
keepalive = 2
preload_app = True
accesslog = "/var/log/goldensignals/access.log"
errorlog = "/var/log/goldensignals/error.log"
loglevel = "info"
```

### 4. Supervisor Configuration

Create `/etc/supervisor/conf.d/goldensignals.conf`:

```ini
[program:goldensignals]
command=/home/goldensignals/GoldenSignalsAI_V2/.venv/bin/gunicorn standalone_backend_optimized:app -c gunicorn_config.py
directory=/home/goldensignals/GoldenSignalsAI_V2
user=goldensignals
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/var/log/goldensignals/supervisor.log
environment=PATH="/home/goldensignals/GoldenSignalsAI_V2/.venv/bin",PYTHONPATH="/home/goldensignals/GoldenSignalsAI_V2"
```

### 5. Nginx Configuration

Create `/etc/nginx/sites-available/goldensignals`:

```nginx
upstream goldensignals {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Frontend static files
    location / {
        root /home/goldensignals/GoldenSignalsAI_V2/frontend/build;
        try_files $uri $uri/ /index.html;
    }
    
    # API proxy
    location /api {
        proxy_pass http://goldensignals;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
    
    # WebSocket support
    location /ws {
        proxy_pass http://goldensignals;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 6. SSL Setup with Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Test automatic renewal
sudo certbot renew --dry-run
```

### 7. Start Services

```bash
# Enable and start services
sudo systemctl enable nginx
sudo systemctl enable supervisor
sudo systemctl enable redis-server
sudo systemctl enable postgresql

sudo systemctl start nginx
sudo systemctl start supervisor
sudo systemctl start redis-server
sudo systemctl start postgresql

# Enable Nginx site
sudo ln -s /etc/nginx/sites-available/goldensignals /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx

# Start application
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start goldensignals
```

## Docker Deployment

### 1. Docker Setup

The repository includes Docker configuration files:

```bash
# Build and run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 2. Production Docker Compose

Use `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://goldensignals:password@db:5432/goldensignals
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./data:/app/data
    restart: always
    
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    depends_on:
      - backend
    restart: always
    
  nginx:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "80:80"
      - "443:443"
    depends_on:
      - backend
      - frontend
    restart: always
    
  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=goldensignals
      - POSTGRES_USER=goldensignals
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: always
    
  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    restart: always

volumes:
  postgres_data:
  redis_data:
```

### 3. Deploy with Docker

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Scale backend workers
docker-compose -f docker-compose.prod.yml scale backend=3
```

## Kubernetes Deployment

### 1. Prepare Kubernetes Manifests

The repository includes Kubernetes configurations in the `k8s/` directory.

### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace goldensignals

# Apply configurations
kubectl apply -f k8s/production/ -n goldensignals

# Check deployment status
kubectl get pods -n goldensignals
kubectl get services -n goldensignals
```

### 3. Helm Deployment (Alternative)

```bash
# Add Helm repository
helm repo add goldensignals https://charts.goldensignals.ai

# Install with Helm
helm install goldensignals goldensignals/goldensignals \
  --namespace goldensignals \
  --create-namespace \
  --values helm/goldensignals/values.yaml
```

## Cloud Provider Deployments

### AWS Deployment

```bash
# Using AWS CLI and Terraform
cd terraform
terraform init
terraform plan -var-file=aws.tfvars
terraform apply -var-file=aws.tfvars
```

### Google Cloud Platform

```bash
# Deploy to GKE
gcloud container clusters create goldensignals-cluster \
  --num-nodes=3 \
  --zone=us-central1-a

gcloud container clusters get-credentials goldensignals-cluster
kubectl apply -f k8s/production/
```

### Azure Deployment

```bash
# Deploy to AKS
az aks create \
  --resource-group goldensignals-rg \
  --name goldensignals-aks \
  --node-count 3

az aks get-credentials \
  --resource-group goldensignals-rg \
  --name goldensignals-aks

kubectl apply -f k8s/production/
```

## Configuration Management

### Environment Variables

All configuration should be managed through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DEBUG` | Enable debug mode | `False` |
| `DATABASE_URL` | Database connection string | `sqlite:///data/goldensignals.db` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379/0` |
| `SECRET_KEY` | Application secret key | Generated |
| `ALLOWED_HOSTS` | Comma-separated list of allowed hosts | `*` |
| `CORS_ORIGINS` | Comma-separated list of CORS origins | `http://localhost:3000` |

### Secrets Management

For production, use a secrets management solution:

1. **Kubernetes Secrets**:
```bash
kubectl create secret generic goldensignals-secrets \
  --from-literal=SECRET_KEY=your-secret-key \
  --from-literal=DATABASE_PASSWORD=your-db-password
```

2. **AWS Secrets Manager**:
```bash
aws secretsmanager create-secret \
  --name goldensignals/production \
  --secret-string file://secrets.json
```

3. **HashiCorp Vault**:
```bash
vault kv put secret/goldensignals \
  secret_key="your-secret-key" \
  database_password="your-db-password"
```

## Database Setup

### PostgreSQL Production Setup

```sql
-- Create database and user
CREATE DATABASE goldensignals;
CREATE USER goldensignals WITH ENCRYPTED PASSWORD 'your-password';
GRANT ALL PRIVILEGES ON DATABASE goldensignals TO goldensignals;

-- Performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
```

### Database Migrations

```bash
# Run migrations
python manage.py migrate

# Create indexes for performance
python manage.py create_indexes
```

### Backup Strategy

```bash
# Automated daily backups
0 2 * * * pg_dump goldensignals | gzip > /backups/goldensignals_$(date +\%Y\%m\%d).sql.gz

# Backup to S3
0 3 * * * aws s3 cp /backups/goldensignals_$(date +\%Y\%m\%d).sql.gz s3://goldensignals-backups/
```

## Monitoring & Logging

### 1. Application Monitoring

**Prometheus Configuration** (`prometheus.yml`):
```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'goldensignals'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### 2. Log Aggregation

**Filebeat Configuration**:
```yaml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/goldensignals/*.log
  
output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "goldensignals-%{+yyyy.MM.dd}"
```

### 3. Health Checks

```bash
# Application health check
curl http://localhost:8000/health

# Database health check
curl http://localhost:8000/health/db

# Redis health check
curl http://localhost:8000/health/redis
```

### 4. Alerting

Create alerts for critical metrics:

```yaml
# Prometheus alert rules
groups:
  - name: goldensignals
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
          
      - alert: SlowResponseTime
        expr: http_request_duration_seconds{quantile="0.95"} > 1
        for: 5m
        annotations:
          summary: "Slow response times detected"
```

## Troubleshooting

### Common Issues

1. **API Authentication Errors (HTTP 401)**
   - Check if API keys are properly set in environment variables
   - Verify yfinance is working: `python -c "import yfinance as yf; print(yf.Ticker('AAPL').info)"`
   - Use fallback data sources if primary fails

2. **Database Connection Issues**
   - Check DATABASE_URL is correct
   - Verify database service is running
   - Check firewall rules for database port

3. **High Memory Usage**
   - Adjust worker count in gunicorn_config.py
   - Enable memory limits in Docker/Kubernetes
   - Check for memory leaks with `memory_profiler`

4. **WebSocket Connection Failures**
   - Verify Nginx WebSocket configuration
   - Check CORS settings
   - Ensure WebSocket upgrade headers are passed

### Debug Mode

Enable debug mode for detailed error messages:

```bash
# Set in environment
export DEBUG=True

# Or in .env file
DEBUG=True

# Run with debug logging
python standalone_backend_optimized.py --log-level=DEBUG
```

### Performance Optimization

1. **Enable Redis Caching**:
   ```bash
   # Install Redis
   sudo apt install redis-server
   
   # Configure in .env
   REDIS_URL=redis://localhost:6379/0
   ```

2. **Database Query Optimization**:
   ```sql
   -- Add indexes
   CREATE INDEX idx_signals_symbol ON signals(symbol);
   CREATE INDEX idx_signals_timestamp ON signals(timestamp);
   CREATE INDEX idx_signal_outcomes_signal_id ON signal_outcomes(signal_id);
   ```

3. **API Response Caching**:
   ```python
   # Configure cache TTL in settings
   CACHE_TTL = {
       'quotes': 300,      # 5 minutes
       'historical': 600,  # 10 minutes
       'signals': 60,      # 1 minute
   }
   ```

### Logs Location

- **Application logs**: `/var/log/goldensignals/app.log`
- **Access logs**: `/var/log/goldensignals/access.log`
- **Error logs**: `/var/log/goldensignals/error.log`
- **Nginx logs**: `/var/log/nginx/`
- **Supervisor logs**: `/var/log/supervisor/`

### Support

For additional help:
1. Check the [API Documentation](API_DOCUMENTATION.md)
2. Review the [Architecture Documentation](ARCHITECTURE.md)
3. Submit issues to the GitHub repository
4. Contact support at support@goldensignals.ai 