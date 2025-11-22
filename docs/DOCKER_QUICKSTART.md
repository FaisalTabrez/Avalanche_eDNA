# Docker Quick Start Guide
## Avalanche eDNA Biodiversity Assessment System

This guide will help you get the Avalanche eDNA system running with Docker in minutes.

---

## Prerequisites

### Required Software
- **Docker Desktop** 20.10+ or Docker Engine
  - Windows: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Mac: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
  - Linux: Install via package manager
    ```bash
    # Ubuntu/Debian
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    ```
- **Docker Compose** 2.0+ (included with Docker Desktop)

### System Requirements
- **Minimum:** 4GB RAM, 10GB disk space
- **Recommended:** 8GB RAM, 50GB disk space
- **For production:** 16GB+ RAM, 100GB+ disk space

---

## Development Setup (Local Testing)

### Step 1: Clone and Configure

```bash
# Clone the repository
git clone https://github.com/FaisalTabrez/Avalanche_eDNA.git
cd Avalanche_eDNA

# Copy environment template (optional for dev)
cp .env.example .env
```

### Step 2: Start Services

```bash
# Start all services in development mode
docker-compose up

# Or run in background (detached mode)
docker-compose up -d
```

This will start:
- ✅ Streamlit application (port 8501)
- ✅ PostgreSQL database (port 5432)
- ✅ Redis cache (port 6379)

### Step 3: Access the Application

Open your browser and navigate to:
```
http://localhost:8501
```

You should see the Avalanche eDNA homepage!

### Step 4: Optional - Start pgAdmin

```bash
# Start with database management UI
docker-compose --profile dev-tools up -d
```

Access pgAdmin at `http://localhost:5050`:
- Email: `admin@avalanche.local`
- Password: `admin`

---

## Production Deployment

### Step 1: Configure Environment

```bash
# Copy production environment template
cp .env.example .env

# Edit with production settings
nano .env  # or your preferred editor
```

**Important variables to change:**
```env
DB_PASSWORD=STRONG_RANDOM_PASSWORD_HERE
SECRET_KEY=RANDOM_SECRET_KEY_32_CHARS_MIN
ENVIRONMENT=production
LOG_LEVEL=INFO

# Optional: Enable monitoring
SENTRY_DSN=https://your-sentry-dsn

# Optional: Enable cloud backups
BACKUP_S3_BUCKET=your-backup-bucket
AWS_ACCESS_KEY_ID=your-aws-key
AWS_SECRET_ACCESS_KEY=your-aws-secret
```

### Step 2: Build Production Image

```bash
# Build the production image
docker-compose -f docker-compose.prod.yml build

# Or pull pre-built image (if available)
docker pull ghcr.io/faisaltabrez/avalanche-edna:latest
```

### Step 3: Start Production Services

```bash
# Start production stack
docker-compose -f docker-compose.prod.yml up -d

# Check status
docker-compose -f docker-compose.prod.yml ps

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

### Step 4: Scale Application (Optional)

```bash
# Run 3 instances of the application behind load balancer
docker-compose -f docker-compose.prod.yml up -d --scale streamlit=3

# Enable nginx load balancer (if configured)
docker-compose -f docker-compose.prod.yml --profile loadbalancer up -d
```

---

## Common Operations

### View Logs

```bash
# All services
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# Specific service
docker-compose logs streamlit
docker-compose logs postgres

# Last 100 lines
docker-compose logs --tail=100 streamlit
```

### Restart Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart streamlit

# Rebuild and restart after code changes
docker-compose up -d --build
```

### Stop Services

```bash
# Stop all services (preserves data)
docker-compose down

# Stop and remove volumes (DELETES ALL DATA!)
docker-compose down -v

# Force stop
docker-compose kill
```

### Execute Commands in Container

```bash
# Access shell in running container
docker-compose exec streamlit bash

# Run Python script
docker-compose exec streamlit python scripts/run_pipeline.py --help

# Run database migrations
docker-compose exec streamlit python scripts/migrate_database.py

# Check Python packages
docker-compose exec streamlit pip list
```

---

## Data Management

### Backup Database

```bash
# Create backup
docker-compose exec postgres pg_dump -U avalanche avalanche_edna > backup_$(date +%Y%m%d).sql

# Compressed backup
docker-compose exec postgres pg_dump -U avalanche avalanche_edna | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore Database

```bash
# Restore from backup
docker-compose exec -T postgres psql -U avalanche avalanche_edna < backup_20250101.sql

# From compressed backup
gunzip < backup_20250101.sql.gz | docker-compose exec -T postgres psql -U avalanche avalanche_edna
```

### Access Data Volumes

```bash
# List volumes
docker volume ls | grep avalanche

# Inspect volume
docker volume inspect avalanche_postgres-data

# Backup volume to archive
docker run --rm -v avalanche_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-data-backup.tar.gz -C /data .

# Restore volume from archive
docker run --rm -v avalanche_postgres-data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres-data-backup.tar.gz -C /data
```

---

## Monitoring & Health Checks

### Check Container Health

```bash
# View health status
docker-compose ps

# Detailed health check
docker inspect --format='{{json .State.Health}}' avalanche-streamlit | jq

# View all container stats
docker stats
```

### Application Health Endpoints

```bash
# Check if application is running
curl http://localhost:8501/_stcore/health

# Check database connection
docker-compose exec streamlit python -c "from src.database.manager import DatabaseManager; print(DatabaseManager().health_check())"

# Check Redis
docker-compose exec redis redis-cli ping
```

---

## Troubleshooting

### Application Won't Start

```bash
# Check logs
docker-compose logs streamlit

# Check for port conflicts
lsof -i :8501  # Mac/Linux
netstat -ano | findstr :8501  # Windows

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Database Connection Errors

```bash
# Verify PostgreSQL is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Test connection manually
docker-compose exec postgres psql -U avalanche -d avalanche_edna -c "SELECT 1;"

# Reset database
docker-compose down
docker volume rm avalanche_postgres-data
docker-compose up -d
```

### Out of Memory

```bash
# Check resource usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → Increase to 8GB+

# Reduce number of replicas
docker-compose -f docker-compose.prod.yml up -d --scale streamlit=1
```

### Port Already in Use

```bash
# Change ports in docker-compose.yml
# Example: Change 8501:8501 to 8502:8501

# Or stop conflicting service
# Mac/Linux
lsof -ti:8501 | xargs kill -9

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### Permission Errors

```bash
# Fix volume permissions (Linux)
sudo chown -R $USER:$USER data/ analysis_outputs/ logs/

# Or run with proper user in docker-compose.yml
user: "${UID}:${GID}"
```

---

## Updating the Application

### Pull Latest Changes

```bash
# Stop services
docker-compose down

# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose build
docker-compose up -d

# Run migrations if needed
docker-compose exec streamlit python scripts/migrate_database.py
```

### Update to New Docker Image

```bash
# Pull latest image
docker pull ghcr.io/faisaltabrez/avalanche-edna:latest

# Restart with new image
docker-compose -f docker-compose.prod.yml up -d
```

---

## Advanced Configuration

### Custom Network Configuration

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  streamlit:
    networks:
      - custom-network
      - avalanche-network

networks:
  custom-network:
    external: true
```

### Volume Mounting for Development

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  streamlit:
    volumes:
      - ./src:/app/src:ro  # Mount source as read-only
      - ./config:/app/config:ro
```

### Using External Database

```yaml
# .env
DB_HOST=external-postgres.example.com
DB_PORT=5432
DB_NAME=avalanche_prod
DB_USER=avalanche_prod_user
DB_PASSWORD=secure_password

# docker-compose.override.yml
version: '3.8'
services:
  postgres:
    profiles:
      - disable  # Don't start local postgres
```

---

## Performance Tuning

### PostgreSQL Optimization

Edit `docker-compose.prod.yml` to tune PostgreSQL:

```yaml
postgres:
  command:
    - "postgres"
    - "-c"
    - "max_connections=200"
    - "-c"
    - "shared_buffers=512MB"  # 25% of RAM
    - "-c"
    - "effective_cache_size=2GB"  # 50-75% of RAM
    - "-c"
    - "work_mem=10MB"
```

### Redis Optimization

```yaml
redis:
  command:
    - redis-server
    - --maxmemory 2gb
    - --maxmemory-policy allkeys-lru
    - --save 60 1000
```

### Application Scaling

```bash
# Horizontal scaling (multiple instances)
docker-compose -f docker-compose.prod.yml up -d --scale streamlit=5

# Resource limits per container
docker-compose -f docker-compose.prod.yml up -d --scale streamlit=2 \
  --compatibility \
  --memory=4g \
  --cpus=2
```

---

## Security Best Practices

### 1. Change Default Passwords
```bash
# Generate secure random password
openssl rand -base64 32

# Update .env file
DB_PASSWORD=<generated-password>
SECRET_KEY=<generated-secret>
```

### 2. Use Secrets (Docker Swarm)
```yaml
# docker-compose.prod.yml
services:
  streamlit:
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password
    secrets:
      - db_password

secrets:
  db_password:
    external: true
```

### 3. Enable SSL/TLS
```yaml
# nginx/nginx.conf
server {
    listen 443 ssl;
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    # ... rest of config
}
```

### 4. Regular Updates
```bash
# Update base images regularly
docker-compose pull
docker-compose up -d

# Scan for vulnerabilities
docker scan avalanche-edna:latest
```

---

## Getting Help

### Documentation
- 📖 [Full Documentation](../docs/)
- 🚀 [Deployment Roadmap](../DEPLOYMENT_ROADMAP.md)
- 🔧 [Installation Guide](../docs/installation.md)

### Logs
```bash
# Export logs for troubleshooting
docker-compose logs > logs_$(date +%Y%m%d).txt
```

### Community
- 🐛 [Report Issues](https://github.com/FaisalTabrez/Avalanche_eDNA/issues)
- 💬 [Discussions](https://github.com/FaisalTabrez/Avalanche_eDNA/discussions)

---

**Last Updated:** November 22, 2025  
**Version:** 1.0  
**Next:** Proceed to [Phase 1.2: Authentication](../DEPLOYMENT_ROADMAP.md#12-authentication--authorization-week-1-2)
