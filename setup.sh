#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[BiasLens]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[Warning]${NC} $1"
}

print_error() {
    echo -e "${RED}[Error]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    }
}

# Check for required environment variables
check_env() {
    if [ ! -f .env ]; then
        print_message "Creating .env file from template..."
        cp .env.example .env
        print_warning "Please edit .env file with your configuration values"
        exit 1
    fi
}

# Create necessary directories
create_directories() {
    print_message "Creating necessary directories..."
    mkdir -p .cache/embeddings
    mkdir -p data/neo4j
}

# Build and start the services
start_services() {
    print_message "Building and starting services..."
    docker-compose build
    docker-compose up -d
}

# Initialize Neo4j database
init_neo4j() {
    print_message "Waiting for Neo4j to start..."
    sleep 10 # Wait for Neo4j to be ready
    
    print_message "Initializing Neo4j database..."
    docker-compose exec backend python -c "
from src.driver import BiasLens
from src.utils.config import BiasLensConfig
config = BiasLensConfig()
biaslens = BiasLens(config)
biaslens.initialize_database()
"
}

# Main setup process
main() {
    print_message "Starting BiasLens setup..."
    
    # Check prerequisites
    check_docker
    check_env
    
    # Create directories
    create_directories
    
    # Start services
    start_services
    
    # Initialize database
    init_neo4j
    
    print_message "Setup complete! BiasLens is now running."
    print_message "Frontend: http://localhost:3000"
    print_message "Backend API: http://localhost:5000"
    print_message "Neo4j Browser: http://localhost:7474"
}

# Helper functions
stop() {
    print_message "Stopping BiasLens services..."
    docker-compose down
}

restart() {
    print_message "Restarting BiasLens services..."
    docker-compose restart
}

logs() {
    print_message "Showing logs..."
    docker-compose logs -f
}

# Command line interface
case "$1" in
    "start")
        main
        ;;
    "stop")
        stop
        ;;
    "restart")
        restart
        ;;
    "logs")
        logs
        ;;
    *)
        print_message "Usage: $0 {start|stop|restart|logs}"
        print_message "  start    - Set up and start BiasLens"
        print_message "  stop     - Stop all services"
        print_message "  restart  - Restart all services"
        print_message "  logs     - Show service logs"
        exit 1
        ;;
esac

exit 0
