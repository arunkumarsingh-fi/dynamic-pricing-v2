#!/bin/bash

# Full Circle Exchange V2 - Podman Deployment Script
# End-to-End Asset Optimization Platform

set -e

# Add common Python bin paths to PATH
export PATH="$PATH:/Users/$(whoami)/Library/Python/3.9/bin:/Users/$(whoami)/.local/bin"

echo "🔄 Full Circle Exchange V2: Podman Deployment"
echo "============================================"

# Function to check if podman is installed
check_podman() {
    if ! command -v podman &> /dev/null; then
        echo "❌ Podman is not installed. Please install Podman first:"
        echo "   macOS: brew install podman"
        echo "   Linux: Check your package manager"
        exit 1
    fi
    echo "✅ Podman found"
}

# Function to check if podman-compose is available
check_podman_compose() {
    if ! command -v podman-compose &> /dev/null; then
        echo "❌ podman-compose is not installed. Please install it:"
        echo "   pip install podman-compose"
        echo "   Note: Make sure ~/.local/bin is in your PATH"
        exit 1
    fi
    echo "✅ podman-compose found"
}

# Function to generate demo data if needed
generate_demo_data() {
    echo "📊 Setting up data pipeline..."
    
    # Create data directory
    mkdir -p data
    
    # Step 1: Generate synthetic transaction data
    if [ ! -f "data/analytics_data.csv" ]; then
        echo "📈 Generating synthetic transaction data (10k records)..."
        if command -v python3 &> /dev/null; then
            python3 data_simulator.py
        else
            python data_simulator.py
        fi
        echo "✅ Transaction data generated"
    else
        echo "✅ Transaction data already exists"
    fi
    
    # Note: ML data files will be generated inside containers during startup
    echo "✅ Data pipeline setup complete"
}

# Main deployment function
deploy() {
    echo "🚀 Starting Full Circle Exchange V2 deployment..."
    
    check_podman
    check_podman_compose
    
    # Generate demo data
    generate_demo_data
    
    # Stop any existing containers
    echo "🛑 Stopping existing containers..."
    podman-compose -f podman-compose.yml down 2>/dev/null || true
    
    # Build and start containers
    echo "🔨 Building and starting containers..."
    podman-compose -f podman-compose.yml up --build -d
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to start..."
    sleep 10
    
    # Check if services are running
    if podman ps | grep -q "full-circle-ml-v2" && podman ps | grep -q "full-circle-ui-v2"; then
        echo ""
        echo "🎉 Full Circle Exchange V2 deployed successfully!"
        echo ""
        echo "📱 Access your application:"
        echo "   🌐 Web Interface V2: http://localhost:8503"
        echo "   🔧 ML API V2: http://localhost:5003"
        echo ""
        echo "🛠️ Management commands:"
        echo "   Stop:    ./deploy.sh stop"
        echo "   Restart: ./deploy.sh restart" 
        echo "   Logs:    ./deploy.sh logs"
        echo "   Status:  ./deploy.sh status"
    else
        echo "❌ Deployment failed. Check logs with: podman-compose logs"
        exit 1
    fi
}

# Handle different commands
case "${1:-deploy}" in
    "deploy"|"start")
        deploy
        ;;
    "stop")
        echo "🛑 Stopping Full Circle Exchange V2..."
        podman-compose -f podman-compose.yml down
        echo "✅ Stopped successfully"
        ;;
    "restart")
        echo "🔄 Restarting Full Circle Exchange V2..."
        podman-compose -f podman-compose.yml down
        sleep 2
        podman-compose -f podman-compose.yml up -d
        echo "✅ Restarted successfully"
        ;;
    "logs")
        echo "📋 Showing logs..."
        podman-compose -f podman-compose.yml logs -f
        ;;
    "status")
        echo "📊 Container status:"
        podman ps --filter "name=full-circle"
        ;;
    "clean")
        echo "🧹 Cleaning up..."
        podman-compose -f podman-compose.yml down
        podman system prune -f
        echo "✅ Cleanup complete"
        ;;
    *)
        echo "Usage: $0 {deploy|start|stop|restart|logs|status|clean}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Build and deploy the application"
        echo "  start    - Same as deploy"  
        echo "  stop     - Stop all containers"
        echo "  restart  - Restart all containers"
        echo "  logs     - Show container logs"
        echo "  status   - Show container status"
        echo "  clean    - Clean up containers and images"
        exit 1
        ;;
esac
