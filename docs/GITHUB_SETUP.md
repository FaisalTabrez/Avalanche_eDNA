# GitHub Repository Setup Guide

This guide covers the configuration needed for the CI/CD pipeline to work properly.

## Table of Contents

1. [Repository Secrets](#repository-secrets)
2. [Environment Configuration](#environment-configuration)
3. [Branch Protection Rules](#branch-protection-rules)
4. [GitHub Security Features](#github-security-features)
5. [Third-Party Integrations](#third-party-integrations)
6. [Testing the Pipeline](#testing-the-pipeline)

## Repository Secrets

Navigate to **Settings → Secrets and variables → Actions** and add the following secrets:

### Deployment Secrets

#### Staging Environment
```
STAGING_SSH_KEY       # Private SSH key for staging server
STAGING_HOST          # Staging server hostname (e.g., staging.avalanche-edna.com)
STAGING_USER          # SSH username (e.g., deploy)
```

#### Production Environment
```
PRODUCTION_SSH_KEY    # Private SSH key for production server
PRODUCTION_HOST       # Production server hostname (e.g., avalanche-edna.com)
PRODUCTION_USER       # SSH username (e.g., deploy)
```

### Notification Secrets
```
SLACK_WEBHOOK         # Slack webhook URL for deployment notifications
```

### Optional Cloud Storage Secrets (for backup system)
```
AWS_ACCESS_KEY_ID     # AWS credentials for S3 backup
AWS_SECRET_ACCESS_KEY
AWS_REGION
AWS_BUCKET_NAME

AZURE_STORAGE_CONNECTION_STRING  # Azure Blob Storage
AZURE_CONTAINER_NAME

GCS_PROJECT_ID        # Google Cloud Storage
GCS_BUCKET_NAME
GCS_CREDENTIALS       # JSON credentials file content
```

## Generating SSH Keys

### For Deployment Servers

1. **Generate SSH key pair**:
   ```bash
   ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/avalanche_deploy
   ```

2. **Copy public key to servers**:
   ```bash
   # For staging
   ssh-copy-id -i ~/.ssh/avalanche_deploy.pub deploy@staging.avalanche-edna.com
   
   # For production
   ssh-copy-id -i ~/.ssh/avalanche_deploy.pub deploy@production.avalanche-edna.com
   ```

3. **Add private key to GitHub**:
   ```bash
   # Display private key
   cat ~/.ssh/avalanche_deploy
   
   # Copy the entire output (including BEGIN/END lines) to:
   # GitHub → Settings → Secrets → STAGING_SSH_KEY / PRODUCTION_SSH_KEY
   ```

4. **Test SSH connection**:
   ```bash
   ssh -i ~/.ssh/avalanche_deploy deploy@staging.avalanche-edna.com
   ```

## Environment Configuration

### Create GitHub Environments

1. Go to **Settings → Environments**
2. Create two environments:
   - `staging`
   - `production`

### Staging Environment Settings

- **Deployment branches**: `develop` only
- **Environment secrets**: Use the STAGING_* secrets
- **Protection rules**: Optional (can enable required reviewers if needed)

### Production Environment Settings

- **Deployment branches**: `main` and tags matching `v*.*.*`
- **Environment secrets**: Use the PRODUCTION_* secrets
- **Protection rules**:
  - ✅ Required reviewers (recommended: 1-2 team members)
  - ✅ Wait timer: 5 minutes (to allow pre-deployment review)

## Branch Protection Rules

### Main Branch (`main`)

Navigate to **Settings → Branches → Branch protection rules** → Add rule for `main`:

- **Branch name pattern**: `main`
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: 1
  - ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ **Require status checks to pass before merging**
  - ✅ Require branches to be up to date before merging
  - Required status checks:
    - `test (ubuntu-latest, 3.11)`
    - `test (ubuntu-latest, 3.10)`
    - `lint`
    - `dependency-scan`
    - `code-scan`
- ✅ **Require conversation resolution before merging**
- ✅ **Do not allow bypassing the above settings**

### Develop Branch (`develop`)

- **Branch name pattern**: `develop`
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: 1
- ✅ **Require status checks to pass before merging**
  - Required status checks:
    - `test (ubuntu-latest, 3.11)`
    - `lint`
- ✅ **Require conversation resolution before merging**

## GitHub Security Features

### 1. Dependabot

Enable Dependabot for automated dependency updates:

1. Go to **Settings → Code security and analysis**
2. Enable:
   - ✅ **Dependency graph**
   - ✅ **Dependabot alerts**
   - ✅ **Dependabot security updates**
   - ✅ **Dependabot version updates**

3. Create `.github/dependabot.yml`:
   ```yaml
   version: 2
   updates:
     - package-ecosystem: "pip"
       directory: "/"
       schedule:
         interval: "weekly"
       open-pull-requests-limit: 10
       reviewers:
         - "FaisalTabrez"
       labels:
         - "dependencies"
         - "python"
   
     - package-ecosystem: "docker"
       directory: "/"
       schedule:
         interval: "weekly"
       reviewers:
         - "FaisalTabrez"
       labels:
         - "dependencies"
         - "docker"
   
     - package-ecosystem: "github-actions"
       directory: "/"
       schedule:
         interval: "weekly"
       reviewers:
         - "FaisalTabrez"
       labels:
         - "dependencies"
         - "ci"
   ```

### 2. Code Scanning

Enable GitHub Advanced Security features:

1. Go to **Settings → Code security and analysis**
2. Enable:
   - ✅ **Code scanning** (using GitHub CodeQL)
   - ✅ **Secret scanning**
   - ✅ **Secret scanning push protection**

The security-scan workflow will upload SARIF results to GitHub Security tab.

### 3. Security Policy

Create `SECURITY.md` in repository root with vulnerability reporting guidelines.

## Third-Party Integrations

### 1. Codecov Setup

1. **Sign up at [codecov.io](https://codecov.io)**:
   - Login with GitHub
   - Select the Avalanche_eDNA repository

2. **Get Codecov token**:
   - Navigate to repository settings in Codecov
   - Copy the upload token

3. **Add to GitHub Secrets**:
   ```
   CODECOV_TOKEN    # Codecov upload token
   ```

4. **Add badge to README.md**:
   ```markdown
   [![codecov](https://codecov.io/gh/FaisalTabrez/Avalanche_eDNA/branch/main/graph/badge.svg)](https://codecov.io/gh/FaisalTabrez/Avalanche_eDNA)
   ```

### 2. Slack Notifications

1. **Create Slack App**:
   - Go to [api.slack.com/apps](https://api.slack.com/apps)
   - Create new app
   - Enable "Incoming Webhooks"

2. **Add webhook to workspace**:
   - Select channel (e.g., #deployments)
   - Copy webhook URL

3. **Add to GitHub Secrets**:
   ```
   SLACK_WEBHOOK    # Slack webhook URL (https://hooks.slack.com/services/...)
   ```

### 3. Docker Registry (GitHub Container Registry)

The pipeline uses GitHub Container Registry (ghcr.io) by default, which requires no additional setup. Images will be available at:

```
ghcr.io/faisaltabrez/avalanche_edna:latest
ghcr.io/faisaltabrez/avalanche_edna:v1.0.0
ghcr.io/faisaltabrez/avalanche_edna:main-abc123
```

**Package visibility**:
1. Go to package page: https://github.com/FaisalTabrez/Avalanche_eDNA/pkgs/container/avalanche_edna
2. Settings → Change visibility → Public (if desired)

## Server Setup

### Staging Server Setup

SSH into staging server and run:

```bash
# 1. Create deployment user
sudo useradd -m -s /bin/bash deploy
sudo usermod -aG docker deploy

# 2. Create application directory
sudo mkdir -p /opt/avalanche
sudo chown deploy:deploy /opt/avalanche

# 3. Switch to deploy user
sudo -u deploy bash

# 4. Clone repository
cd /opt/avalanche
git clone https://github.com/FaisalTabrez/Avalanche_eDNA.git .
git checkout develop

# 5. Create environment file
cp .env.example .env
# Edit .env with staging-specific values

# 6. Create docker-compose.staging.yml
cat > docker-compose.staging.yml <<EOF
version: '3.8'

services:
  app:
    image: ghcr.io/faisaltabrez/avalanche_edna:develop
    container_name: avalanche_staging
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=staging
    env_file:
      - .env
    volumes:
      - ./data:/app/data
      - ./consolidated_data:/app/consolidated_data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:15
    container_name: avalanche_db_staging
    environment:
      POSTGRES_DB: avalanche
      POSTGRES_USER: avalanche
      POSTGRES_PASSWORD: \${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
EOF

# 7. Start services
docker-compose -f docker-compose.staging.yml up -d

# 8. Configure nginx (optional)
sudo apt install nginx
sudo nano /etc/nginx/sites-available/avalanche-staging

# Nginx config:
server {
    listen 80;
    server_name staging.avalanche-edna.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }

    location /health {
        proxy_pass http://localhost:8501/health;
    }
}

sudo ln -s /etc/nginx/sites-available/avalanche-staging /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### Production Server Setup

Follow same steps as staging, but:
- Use `main` branch
- Use `docker-compose.prod.yml`
- Use production domain
- Enable SSL/TLS (Let's Encrypt recommended):

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d avalanche-edna.com
```

## Testing the Pipeline

### 1. Test Workflows Locally

Before pushing, test workflows locally using the Makefile:

```bash
# Run all CI checks locally
make ci-test       # Runs test suite
make ci-lint       # Runs linting
make ci-security   # Runs security scans

# Or run all at once
make ci-test && make ci-lint && make ci-security
```

### 2. Install Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

### 3. Test Workflows on GitHub

#### Test Build Workflow
```bash
git checkout -b test/ci-build
git push origin test/ci-build
```
This triggers: test.yml, lint.yml, security-scan.yml, build.yml

#### Test Staging Deployment
```bash
git checkout develop
git merge test/ci-build
git push origin develop
```
This triggers: deploy-staging.yml (if staging server configured)

#### Test Production Deployment
```bash
# Create release tag
git checkout main
git merge develop
git tag v1.0.0
git push origin main --tags
```
This triggers: deploy-production.yml (requires production approval)

### 4. Monitor Workflow Runs

1. Go to **Actions** tab
2. Monitor running workflows
3. Check for failures
4. Review logs for each step

### 5. Verify Deployment

**Staging**:
```bash
curl -f https://staging.avalanche-edna.com/health
```

**Production**:
```bash
curl -f https://avalanche-edna.com/health
```

## Troubleshooting

### Workflow Failures

1. **SSH Connection Failed**:
   - Verify SSH key is correct
   - Test SSH connection manually
   - Check firewall rules

2. **Docker Build Failed**:
   - Check Dockerfile syntax
   - Verify base image availability
   - Check disk space on runner

3. **Tests Failed**:
   - Run tests locally first: `make test`
   - Check test environment setup
   - Review test logs

4. **Deployment Failed**:
   - Check server logs: `ssh deploy@server "docker logs avalanche"`
   - Verify health endpoint works
   - Check rollback was successful

### Common Issues

**Issue**: `permission denied (publickey)`
```bash
# Solution: Verify SSH key format
cat ~/.ssh/avalanche_deploy | head -1
# Should show: -----BEGIN OPENSSH PRIVATE KEY-----
# If shows "-----BEGIN RSA PRIVATE KEY-----", regenerate with ed25519
```

**Issue**: `dial tcp: lookup staging.avalanche-edna.com: no such host`
```bash
# Solution: Update DNS records or use IP address
# Or add to /etc/hosts for testing
```

**Issue**: `Docker image not found`
```bash
# Solution: Check GHCR package visibility
# Ensure package is public or authenticate:
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

## Security Checklist

Before going live, verify:

- [ ] All secrets are configured in GitHub
- [ ] SSH keys use ed25519 algorithm
- [ ] SSH keys are properly secured (600 permissions)
- [ ] Branch protection rules are enabled
- [ ] Required approvals are configured for production
- [ ] Dependabot is enabled
- [ ] Secret scanning is enabled
- [ ] Code scanning is enabled
- [ ] SSL/TLS is configured on production
- [ ] Firewall rules allow only necessary traffic
- [ ] Backup system is tested and working
- [ ] Rollback procedure is tested
- [ ] Monitoring/alerts are configured

## Next Steps

After completing this setup:

1. **Test the full pipeline** with a dummy change
2. **Set up monitoring** (Phase 2.3)
3. **Configure job queue** (Phase 2.2)
4. **Add comprehensive tests** (Phase 2.4)
5. **Document runbooks** for common operations

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Security Features](https://docs.github.com/en/code-security)
- [Docker Documentation](https://docs.docker.com/)
- [Codecov Documentation](https://docs.codecov.com/)
- [Slack API Documentation](https://api.slack.com/)
