"""
Production performance optimization configuration
Gunicorn/uWSGI settings, worker management, and performance tuning
"""
import multiprocessing
import os

# ============================================================================
# Gunicorn Configuration
# ============================================================================

# Server Socket
bind = f"0.0.0.0:{os.getenv('PORT', '8501')}"
backlog = 2048

# Worker Processes
workers = int(os.getenv('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = 'sync'  # or 'gevent', 'eventlet' for async
worker_connections = 1000
max_requests = 10000  # Restart workers after this many requests
max_requests_jitter = 1000  # Add randomness to prevent all workers restarting at once
timeout = 300  # 5 minutes for long-running requests
graceful_timeout = 120  # Time to wait for graceful shutdown
keepalive = 5  # Keep-alive connections

# Process Naming
proc_name = 'avalanche_edna'

# Logging
accesslog = os.getenv('GUNICORN_ACCESS_LOG', '-')  # stdout
errorlog = os.getenv('GUNICORN_ERROR_LOG', '-')  # stdout
loglevel = os.getenv('LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Server Mechanics
daemon = False  # Don't daemonize (let Docker/systemd handle this)
pidfile = os.getenv('GUNICORN_PID', '/tmp/gunicorn.pid')
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed)
keyfile = os.getenv('SSL_KEY_FILE')
certfile = os.getenv('SSL_CERT_FILE')
ca_certs = os.getenv('SSL_CA_CERTS')

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8190

# Preload app for faster worker startup
preload_app = True


# ============================================================================
# Worker Lifecycle Hooks
# ============================================================================

def on_starting(server):
    """Called just before the master process is initialized"""
    print("Starting Gunicorn server...")


def on_reload(server):
    """Called when configuration is reloaded"""
    print("Reloading configuration...")


def when_ready(server):
    """Called just after the server is started"""
    print(f"Server ready. Listening on {bind}")
    print(f"Workers: {workers}")


def pre_fork(server, worker):
    """Called just before a worker is forked"""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked"""
    print(f"Worker spawned (pid: {worker.pid})")
    
    # Initialize per-worker resources
    # e.g., database connections, cache connections
    from src.utils.cache import cache
    from src.database.manager import db_manager
    
    # Test connections
    try:
        cache.client.ping()
        print(f"Worker {worker.pid}: Redis connection OK")
    except Exception as e:
        print(f"Worker {worker.pid}: Redis connection failed: {e}")


def pre_exec(server):
    """Called just before a new master process is forked"""
    print("Forked child, re-executing...")


def worker_int(worker):
    """Called when a worker receives the SIGINT or SIGQUIT signal"""
    print(f"Worker received INT or QUIT signal (pid: {worker.pid})")


def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal"""
    print(f"Worker received ABRT signal (pid: {worker.pid})")


def pre_request(worker, req):
    """Called just before a worker processes the request"""
    worker.log.debug(f"{req.method} {req.path}")


def post_request(worker, req, environ, resp):
    """Called after a worker processes the request"""
    pass


def child_exit(server, worker):
    """Called just after a worker has been exited"""
    print(f"Worker exited (pid: {worker.pid})")


def worker_exit(server, worker):
    """Called just after a worker has been exited"""
    print(f"Worker shutting down (pid: {worker.pid})")


def nworkers_changed(server, new_value, old_value):
    """Called when the number of workers changes"""
    print(f"Number of workers changed from {old_value} to {new_value}")


# ============================================================================
# uWSGI Configuration (alternative to Gunicorn)
# ============================================================================

# This configuration can be used in a uwsgi.ini file:
"""
[uwsgi]
# Application
module = streamlit_app:app
callable = app

# Process Management
master = true
processes = %(cpu_count * 2 + 1)
threads = 2
enable-threads = true

# Socket
http-socket = :8501
socket = /tmp/avalanche.sock
chmod-socket = 660
vacuum = true

# Resource Limits
max-requests = 5000
max-worker-lifetime = 3600
reload-on-rss = 512
worker-reload-mercy = 60

# Logging
logto = /var/log/uwsgi/avalanche.log
log-maxsize = 100000000
log-backupname = /var/log/uwsgi/avalanche.log.old

# Performance
lazy-apps = true
cheaper = 2
cheaper-initial = 4
cheaper-step = 1
cheaper-algo = busyness

# Stats
stats = 127.0.0.1:9191
stats-http = true

# Monitoring
memory-report = true
harakiri = 300
harakiri-verbose = true

# Buffer size
buffer-size = 32768
post-buffering = 8192
"""


# ============================================================================
# Nginx Configuration Template
# ============================================================================

NGINX_CONFIG = """
upstream avalanche_backend {
    # Least connections load balancing
    least_conn;
    
    # Backend servers
    server 127.0.0.1:8501 max_fails=3 fail_timeout=30s;
    # Add more servers for horizontal scaling:
    # server 127.0.0.1:8502 max_fails=3 fail_timeout=30s;
    # server 127.0.0.1:8503 max_fails=3 fail_timeout=30s;
    
    # Keep-alive connections
    keepalive 32;
}

server {
    listen 80;
    server_name avalanche-edna.example.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name avalanche-edna.example.com;
    
    # SSL Configuration
    ssl_certificate /etc/ssl/certs/avalanche.crt;
    ssl_certificate_key /etc/ssl/private/avalanche.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    
    # Security Headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    
    # Client Body Size (for file uploads)
    client_max_body_size 500M;
    client_body_buffer_size 128k;
    
    # Timeouts
    client_body_timeout 300s;
    client_header_timeout 60s;
    keepalive_timeout 65s;
    send_timeout 300s;
    
    # Proxy Timeouts
    proxy_connect_timeout 300s;
    proxy_send_timeout 300s;
    proxy_read_timeout 300s;
    
    # Gzip Compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/rss+xml
        application/atom+xml
        image/svg+xml;
    
    # Rate Limiting
    limit_req_zone $binary_remote_addr zone=general:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=api:10m rate=20r/s;
    limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/m;
    
    # Static Files
    location /static/ {
        alias /var/www/avalanche/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
        access_log off;
    }
    
    # Media Files
    location /media/ {
        alias /var/www/avalanche/media/;
        expires 30d;
        add_header Cache-Control "public";
    }
    
    # API Endpoints
    location /api/ {
        limit_req zone=api burst=50 nodelay;
        
        proxy_pass http://avalanche_backend;
        proxy_http_version 1.1;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        
        # CORS (if needed)
        add_header Access-Control-Allow-Origin "*";
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
        add_header Access-Control-Allow-Headers "Content-Type, Authorization";
        
        # Handle OPTIONS for CORS preflight
        if ($request_method = OPTIONS) {
            return 204;
        }
    }
    
    # Authentication Endpoints (stricter rate limit)
    location /api/auth/ {
        limit_req zone=auth burst=5 nodelay;
        
        proxy_pass http://avalanche_backend;
        proxy_http_version 1.1;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # WebSocket Support (if needed for Streamlit)
    location /stream {
        proxy_pass http://avalanche_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
    
    # Main Application
    location / {
        limit_req zone=general burst=20 nodelay;
        
        proxy_pass http://avalanche_backend;
        proxy_http_version 1.1;
        
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Connection "";
        
        # Proxy buffering
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 24 4k;
        proxy_busy_buffers_size 8k;
        proxy_max_temp_file_size 2048m;
        proxy_temp_file_write_size 32k;
    }
    
    # Health Check Endpoint
    location /health {
        access_log off;
        proxy_pass http://avalanche_backend;
    }
    
    # Deny access to hidden files
    location ~ /\. {
        deny all;
        access_log off;
        log_not_found off;
    }
}
"""


# ============================================================================
# Systemd Service File
# ============================================================================

SYSTEMD_SERVICE = """
[Unit]
Description=Avalanche eDNA Analysis Platform
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=avalanche
Group=avalanche
WorkingDirectory=/opt/avalanche
Environment="PATH=/opt/avalanche/venv/bin"
Environment="PYTHONPATH=/opt/avalanche"
ExecStart=/opt/avalanche/venv/bin/gunicorn \\
    --config /opt/avalanche/config/gunicorn_config.py \\
    streamlit_app:app
ExecReload=/bin/kill -s HUP $MAINPID
KillMode=mixed
TimeoutStopSec=120
PrivateTmp=true
Restart=on-failure
RestartSec=10

# Resource Limits
LimitNOFILE=65536
LimitNPROC=4096

# Security
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/avalanche/data /opt/avalanche/logs

[Install]
WantedBy=multi-user.target
"""


# ============================================================================
# Docker Compose Production Override
# ============================================================================

DOCKER_COMPOSE_PROD = """
version: '3.8'

services:
  streamlit:
    command: gunicorn --config /app/config/gunicorn_config.py streamlit_app:app
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./config/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./static:/var/www/avalanche/static:ro
      - ./media:/var/www/avalanche/media:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - streamlit
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
"""
