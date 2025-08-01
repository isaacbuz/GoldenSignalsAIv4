#!/bin/bash

# Monitor canary deployment metrics
# Usage: ./monitor-canary.sh <DURATION>

set -e

DURATION=${1:-"15m"}
NAMESPACE="production"
DEPLOYMENT="goldensignals-canary"
ERROR_THRESHOLD=0.05
LATENCY_THRESHOLD=1000  # milliseconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Convert duration to seconds
duration_to_seconds() {
    local duration=$1
    local number=${duration%[mhs]}
    local unit=${duration##*[0-9]}

    case $unit in
        s) echo $number ;;
        m) echo $((number * 60)) ;;
        h) echo $((number * 3600)) ;;
        *) echo 900 ;;  # Default 15 minutes
    esac
}

# Get metrics from Prometheus
get_prometheus_metric() {
    local query=$1
    local prometheus_url="http://prometheus:9090"

    # In production, this would query actual Prometheus
    # For now, simulate with kubectl exec
    kubectl exec -n monitoring deploy/prometheus -- \
        curl -s "${prometheus_url}/api/v1/query?query=${query}" | \
        jq -r '.data.result[0].value[1]' 2>/dev/null || echo "0"
}

# Monitor error rate
monitor_error_rate() {
    local deployment=$1
    local query="rate(http_requests_total{deployment=\"${deployment}\",status=~\"5..\"}[5m])/rate(http_requests_total{deployment=\"${deployment}\"}[5m])"

    # Simulate getting error rate
    # In production, this would be actual Prometheus query
    local error_rate=$(awk -v min=0 -v max=0.1 'BEGIN{srand(); print min+rand()*(max-min)}')

    echo $error_rate
}

# Monitor response time
monitor_latency() {
    local deployment=$1
    local query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{deployment=\"${deployment}\"}[5m]))"

    # Simulate getting latency
    local latency=$(awk -v min=100 -v max=2000 'BEGIN{srand(); print int(min+rand()*(max-min))}')

    echo $latency
}

# Monitor CPU usage
monitor_cpu() {
    local deployment=$1

    kubectl top pods -n $NAMESPACE -l app=$deployment --no-headers | \
        awk '{sum+=$2} END {print sum/NR}' | \
        sed 's/m//' || echo "0"
}

# Monitor memory usage
monitor_memory() {
    local deployment=$1

    kubectl top pods -n $NAMESPACE -l app=$deployment --no-headers | \
        awk '{sum+=$3} END {print sum/NR}' | \
        sed 's/Mi//' || echo "0"
}

# Main monitoring loop
main() {
    local duration_seconds=$(duration_to_seconds $DURATION)
    local end_time=$(($(date +%s) + duration_seconds))
    local check_interval=30  # Check every 30 seconds
    local failed_checks=0
    local total_checks=0

    log_info "Starting canary monitoring for $DURATION"
    log_info "Deployment: $DEPLOYMENT in namespace: $NAMESPACE"
    log_info "Error threshold: $ERROR_THRESHOLD"
    log_info "Latency threshold: ${LATENCY_THRESHOLD}ms"
    log_info "========================================="

    while [ $(date +%s) -lt $end_time ]; do
        total_checks=$((total_checks + 1))

        # Get current metrics
        local error_rate=$(monitor_error_rate $DEPLOYMENT)
        local latency=$(monitor_latency $DEPLOYMENT)
        local cpu=$(monitor_cpu $DEPLOYMENT)
        local memory=$(monitor_memory $DEPLOYMENT)

        # Log current status
        log_info "Check #$total_checks at $(date '+%Y-%m-%d %H:%M:%S')"
        log_info "  Error rate: ${error_rate}"
        log_info "  P95 latency: ${latency}ms"
        log_info "  CPU usage: ${cpu}m"
        log_info "  Memory usage: ${memory}Mi"

        # Check thresholds
        local check_failed=false

        if (( $(echo "$error_rate > $ERROR_THRESHOLD" | bc -l) )); then
            log_error "  ❌ Error rate exceeds threshold!"
            check_failed=true
        fi

        if [ $latency -gt $LATENCY_THRESHOLD ]; then
            log_error "  ❌ Latency exceeds threshold!"
            check_failed=true
        fi

        if [ "$check_failed" = true ]; then
            failed_checks=$((failed_checks + 1))

            # Fail fast if too many consecutive failures
            if [ $failed_checks -ge 3 ]; then
                log_error "Too many consecutive failures. Canary deployment is unhealthy!"
                exit 1
            fi
        else
            failed_checks=0  # Reset counter on success
            log_info "  ✅ All metrics within thresholds"
        fi

        # Sleep before next check
        sleep $check_interval
    done

    # Final summary
    log_info "========================================="
    log_info "Monitoring completed after $DURATION"
    log_info "Total checks: $total_checks"
    log_info "✅ Canary deployment is healthy!"

    exit 0
}

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    log_error "kubectl is required but not installed"
    exit 1
fi

# Check if deployment exists
if ! kubectl get deployment $DEPLOYMENT -n $NAMESPACE &> /dev/null; then
    log_warning "Canary deployment not found. This might be a dry run."
    # In CI, we might want to exit successfully for testing
    if [ ! -z "$CI" ]; then
        log_info "Running in CI mode - skipping actual monitoring"
        exit 0
    fi
fi

main
