#!/bin/bash

# 🚀 GoldenSignalsAI V3 - Startup Script
# Complete AI Trading Platform Launch System

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logo
echo -e "${BLUE}"
cat << "EOF"
   ▄████  ▒█████   ██▓    ▓█████▄ ▓█████  ███▄    █   ██████  ██▓  ▄████  ███▄    █  ▄▄▄       ██▓      ██████    
 ██▒ ▀█▒▒██▒  ██▒▓██▒    ▒██▀ ██▌▓█   ▀  ██ ▀█   █ ▒██    ▒ ▓██▒ ██▒ ▀█▒ ██ ▀█   █ ▒████▄    ▓██▒    ▒██    ▒    
▒██░▄▄▄░▒██░  ██▒▒██░    ░██   █▌▒███   ▓██  ▀█ ██▒░ ▓██▄   ▒██▒▒██░▄▄▄░▓██  ▀█ ██▒▒██  ▀█▄  ▒██░    ░ ▓██▄      
░▓█  ██▓▒██   ██░▒██░    ░▓█▄   ▌▒▓█  ▄ ▓██▒  ▐▌██▒  ▒   ██▒░██░░▓█  ██▓▓██▒  ▐▌██▒░██▄▄▄▄██ ▒██░      ▒   ██▒   
░▒▓███▀▒░ ████▓▒░░██████▒░▒████▓ ░▒████▒▒██░   ▓██░▒██████▒▒░██░░▒▓███▀▒▒██░   ▓██░ ▓█   ▓██▒░██████▒▒██████▒▒   
 ░▒   ▒ ░ ▒░▒░▒░ ░ ▒░▓  ░ ▒▒▓  ▒ ░░ ▒░ ░░ ▒░   ▒ ▒ ▒ ▒▓▒ ▒ ░░▓   ░▒   ▒ ░ ▒░   ▒ ▒  ▒▒   ▓▒█░░ ▒░▓  ░▒ ▒▓▒ ▒ ░   
  ░   ░   ░ ▒ ▒░ ░ ░ ▒  ░ ░ ▒  ▒  ░ ░  ░░ ░░   ░ ▒░░ ░▒  ░ ░ ▒ ░  ░   ░ ░ ░░   ░ ▒░  ▒   ▒▒ ░░ ░ ▒  ░░ ░▒  ░ ░   
░ ░   ░ ░ ░ ░ ▒    ░ ░    ░ ░  ░    ░      ░   ░ ░ ░  ░  ░   ▒ ░░ ░   ░    ░   ░ ░   ░   ▒     ░ ░   ░  ░  ░     
      ░     ░ ░      ░  ░   ░       ░  ░         ░       ░   ░        ░          ░       ░  ░    ░  ░      ░     
                          ░                                                                                        

        🚀 GoldenSignalsAI V3 - Next-Generation AI Trading Platform 🚀
EOF
echo -e "${NC}"

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}[$(date +'%Y-%m-%d %H:%M:%S')] ${message}${NC}"
}

print_success() {
    print_message $GREEN "✅ $1"
}

print_info() {
    print_message $BLUE "ℹ️  $1"
}

print_warning() {
    print_message $YELLOW "⚠️  $1"
}

print_error() {
    print_message $RED "❌ $1"
}

print_header() {
    echo -e "\n${PURPLE}=== $1 ===${NC}\n"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_requirements() {
    print_header "Checking System Requirements"
    
    local all_good=true
    
    # Check Python
    if command_exists python3; then
        python_version=$(python3 --version | cut -d' ' -f2)
        print_success "Python $python_version found"
    else
        print_error "Python 3.11+ is required"
        all_good=false
    fi
    
    # Check Node.js
    if command_exists node; then
        node_version=$(node --version)
        print_success "Node.js $node_version found"
    else
        print_error "Node.js 18+ is required"
        all_good=false
    fi
    
    # Check Docker
    if command_exists docker; then
        docker_version=$(docker --version | cut -d' ' -f3 | cut -d',' -f1)
        print_success "Docker $docker_version found"
    else
        print_error "Docker is required"
        all_good=false
    fi
    
    # Check Docker Compose
    if command_exists docker-compose || docker compose version >/dev/null 2>&1; then
        print_success "Docker Compose found"
    else
        print_error "Docker Compose is required"
        all_good=false
    fi
    
    if [ "$all_good" = false ]; then
        print_error "Please install missing requirements and try again"
        exit 1
    fi
}

# Function to setup environment
setup_environment() {
    print_header "Setting Up Environment"
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        print_info "Creating .env file from template..."
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success ".env file created from template"
            print_warning "Please edit .env file with your API keys before continuing"
            
            # Ask user if they want to continue
            read -p "Have you configured your .env file with API keys? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                print_info "Please configure your .env file and run this script again"
                exit 0
            fi
        else
            print_error ".env.example not found. Creating basic .env file..."
            cat > .env << EOL
# GoldenSignalsAI V3 Configuration
SECRET_KEY=your_ultra_secure_secret_key_change_this_in_production
DEBUG=false
ENVIRONMENT=production

# Database
DB_PASSWORD=goldensignals_secure_password
REDIS_PASSWORD=goldensignals_redis_password

# AI APIs
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here

# Trading APIs
ALPHA_VANTAGE_KEY=your_alpha_vantage_key
POLYGON_KEY=your_polygon_key
ALPACA_KEY=your_alpaca_key
ALPACA_SECRET=your_alpaca_secret

# Monitoring
SENTRY_DSN=your_sentry_dsn_optional
GRAFANA_PASSWORD=admin
EOL
            print_warning "Basic .env file created. Please edit it with your actual API keys!"
            exit 1
        fi
    else
        print_success ".env file already exists"
    fi
    
    # Create necessary directories
    print_info "Creating necessary directories..."
    mkdir -p logs data models monitoring/grafana/dashboards monitoring/prometheus ssl nginx/conf.d scripts
    print_success "Directories created"
}

# Function to start services with Docker Compose
start_docker_services() {
    print_header "Starting Docker Services"
    
    # Check if docker-compose.v3.yml exists
    if [ ! -f docker-compose.v3.yml ]; then
        print_error "docker-compose.v3.yml not found!"
        exit 1
    fi
    
    print_info "Building and starting all services..."
    
    # Pull latest images
    print_info "Pulling latest base images..."
    docker-compose -f docker-compose.v3.yml pull --ignore-pull-failures
    
    # Build and start services
    print_info "Building and starting services (this may take a few minutes)..."
    docker-compose -f docker-compose.v3.yml up -d --build
    
    if [ $? -eq 0 ]; then
        print_success "All services started successfully!"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Function to wait for services to be healthy
wait_for_services() {
    print_header "Waiting for Services to be Ready"
    
    local max_attempts=60
    local attempt=0
    
    print_info "Checking service health (timeout: ${max_attempts}s)..."
    
    while [ $attempt -lt $max_attempts ]; do
        # Check database
        if docker-compose -f docker-compose.v3.yml exec -T database pg_isready -U goldensignals -d goldensignals >/dev/null 2>&1; then
            db_ready=true
        else
            db_ready=false
        fi
        
        # Check Redis
        if docker-compose -f docker-compose.v3.yml exec -T redis redis-cli ping >/dev/null 2>&1; then
            redis_ready=true
        else
            redis_ready=false
        fi
        
        # Check backend API
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            api_ready=true
        else
            api_ready=false
        fi
        
        # Check frontend
        if curl -f http://localhost:3000 >/dev/null 2>&1; then
            frontend_ready=true
        else
            frontend_ready=false
        fi
        
        if [ "$db_ready" = true ] && [ "$redis_ready" = true ] && [ "$api_ready" = true ] && [ "$frontend_ready" = true ]; then
            print_success "All services are healthy!"
            break
        fi
        
        printf "."
        sleep 1
        ((attempt++))
    done
    
    echo ""
    
    if [ $attempt -eq $max_attempts ]; then
        print_warning "Services may still be starting up. Check logs if issues persist."
    fi
}

# Function to show service status
show_service_status() {
    print_header "Service Status"
    
    # Show running containers
    print_info "Running containers:"
    docker-compose -f docker-compose.v3.yml ps
    
    echo ""
    print_info "Service URLs:"
    echo -e "${CYAN}🌐 Frontend Dashboard:${NC}     http://localhost:3000"
    echo -e "${CYAN}🔧 API Documentation:${NC}     http://localhost:8000/docs"
    echo -e "${CYAN}📊 Grafana Monitoring:${NC}    http://localhost:3001 (admin/admin)"
    echo -e "${CYAN}📈 Prometheus Metrics:${NC}    http://localhost:9090"
    echo -e "${CYAN}🔍 Health Check:${NC}          http://localhost:8000/health"
    
    echo ""
    print_info "WebSocket endpoints:"
    echo -e "${CYAN}📡 Real-time Signals:${NC}     ws://localhost:8000/ws/signals/{symbol}"
    echo -e "${CYAN}📊 Market Data Stream:${NC}    ws://localhost:8000/ws/market-data/{symbol}"
}

# Function to show logs
show_logs() {
    print_header "Service Logs"
    
    print_info "Showing last 50 lines of backend logs..."
    docker-compose -f docker-compose.v3.yml logs --tail=50 backend
    
    echo ""
    print_info "To follow logs in real-time, use:"
    echo "docker-compose -f docker-compose.v3.yml logs -f [service_name]"
    
    echo ""
    print_info "Available services: backend, frontend, database, redis, prometheus, grafana"
}

# Function to run health checks
run_health_checks() {
    print_header "Running Health Checks"
    
    # API Health Check
    print_info "Checking API health..."
    if curl -s http://localhost:8000/health | jq . >/dev/null 2>&1; then
        print_success "API is responding correctly"
    else
        print_warning "API health check failed or jq not installed"
    fi
    
    # Database connectivity
    print_info "Checking database connectivity..."
    if docker-compose -f docker-compose.v3.yml exec -T database pg_isready -U goldensignals -d goldensignals >/dev/null 2>&1; then
        print_success "Database is accepting connections"
    else
        print_warning "Database connectivity check failed"
    fi
    
    # Redis connectivity
    print_info "Checking Redis connectivity..."
    if docker-compose -f docker-compose.v3.yml exec -T redis redis-cli ping >/dev/null 2>&1; then
        print_success "Redis is responding"
    else
        print_warning "Redis connectivity check failed"
    fi
    
    # Frontend check
    print_info "Checking frontend..."
    if curl -f http://localhost:3000 >/dev/null 2>&1; then
        print_success "Frontend is accessible"
    else
        print_warning "Frontend check failed"
    fi
}

# Function to stop services
stop_services() {
    print_header "Stopping Services"
    
    print_info "Stopping all services..."
    docker-compose -f docker-compose.v3.yml down
    
    if [ $? -eq 0 ]; then
        print_success "All services stopped"
    else
        print_warning "Some services may still be running"
    fi
}

# Function to restart services
restart_services() {
    print_header "Restarting Services"
    
    stop_services
    sleep 3
    start_docker_services
    wait_for_services
}

# Function to show help
show_help() {
    echo -e "${BLUE}GoldenSignalsAI V3 Startup Script${NC}\n"
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  start     Start all services (default)"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    Show service status"
    echo "  logs      Show service logs"
    echo "  health    Run health checks"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start    # Start the platform"
    echo "  $0 logs     # View logs"
    echo "  $0 health   # Check service health"
}

# Main execution
main() {
    local action=${1:-start}
    
    case $action in
        start)
            check_requirements
            setup_environment
            start_docker_services
            wait_for_services
            show_service_status
            run_health_checks
            print_success "🎉 GoldenSignalsAI V3 is ready! Visit http://localhost:3000 to get started."
            ;;
        stop)
            stop_services
            ;;
        restart)
            restart_services
            show_service_status
            ;;
        status)
            show_service_status
            ;;
        logs)
            show_logs
            ;;
        health)
            run_health_checks
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $action"
            show_help
            exit 1
            ;;
    esac
}

# Trap to handle Ctrl+C gracefully
trap 'print_warning "Script interrupted. To stop services, run: $0 stop"; exit 1' INT

# Run main function
main "$@" 