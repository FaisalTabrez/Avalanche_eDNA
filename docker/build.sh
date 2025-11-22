#!/bin/bash
# Docker Build and Deployment Script for Avalanche eDNA

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}  Avalanche eDNA - Docker Build${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}Warning: docker-compose not found, trying docker compose${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Parse arguments
TARGET="${1:-application}"  # Default to application stage
ACTION="${2:-build}"        # Default to build

echo -e "${GREEN}Target:${NC} $TARGET"
echo -e "${GREEN}Action:${NC} $ACTION\n"

# Navigate to docker directory
cd "$(dirname "$0")"

if [ "$ACTION" == "build" ]; then
    echo -e "${BLUE}Building Docker image...${NC}"
    
    # Build the image
    docker build \
        --target $TARGET \
        --tag avalanche-edna:$TARGET \
        --tag avalanche-edna:latest \
        --file Dockerfile \
        --progress=plain \
        ..
    
    echo -e "\n${GREEN}✓ Build complete!${NC}"
    echo -e "Image: ${BLUE}avalanche-edna:$TARGET${NC}\n"
    
elif [ "$ACTION" == "run" ]; then
    echo -e "${BLUE}Running Docker container...${NC}"
    
    docker run -it --rm \
        --name avalanche-edna-dev \
        -p 8501:8501 \
        -v "$(pwd)/../data:/app/data" \
        -v "$(pwd)/../reference:/app/reference" \
        avalanche-edna:$TARGET
    
elif [ "$ACTION" == "compose" ]; then
    echo -e "${BLUE}Starting services with docker-compose...${NC}"
    
    $DOCKER_COMPOSE up -d
    
    echo -e "\n${GREEN}✓ Services started!${NC}"
    echo -e "Streamlit: ${BLUE}http://localhost:8501${NC}"
    echo -e "Prometheus: ${BLUE}http://localhost:9090${NC}"
    echo -e "Grafana: ${BLUE}http://localhost:3000${NC}\n"
    
elif [ "$ACTION" == "stop" ]; then
    echo -e "${BLUE}Stopping services...${NC}"
    
    $DOCKER_COMPOSE down
    
    echo -e "${GREEN}✓ Services stopped${NC}\n"
    
elif [ "$ACTION" == "logs" ]; then
    echo -e "${BLUE}Showing logs...${NC}\n"
    
    $DOCKER_COMPOSE logs -f avalanche-app
    
else
    echo -e "${RED}Unknown action: $ACTION${NC}"
    echo -e "Usage: $0 [target] [action]"
    echo -e "  target: application (default), development, dependencies"
    echo -e "  action: build (default), run, compose, stop, logs"
    exit 1
fi
