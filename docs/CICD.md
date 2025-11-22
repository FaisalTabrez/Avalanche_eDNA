# CI/CD Pipeline Documentation

This document describes the Continuous Integration and Continuous Deployment (CI/CD) pipeline for Avalanche.

## Overview

The CI/CD pipeline is implemented using **GitHub Actions** and provides:
- Automated testing across multiple platforms
- Code quality and security checks
- Docker image building and publishing
- Automated deployments to staging and production
- Rollback capabilities

## Workflows

### 1. Test Workflow (`test.yml`)

**Triggers:**
- Push to `main`, `develop`, `chore/reorg-codebase` branches
- Pull requests to `main`, `develop`

**What it does:**
- Runs tests on Ubuntu, Windows, and macOS
- Tests Python 3.9, 3.10, and 3.11
- Generates code coverage reports
- Uploads coverage to Codecov
- Fails if coverage < 70%

**Usage:**
```bash
# Run tests locally
pytest tests/ -v --cov=src --cov-report=term-missing

# Check coverage threshold
pytest tests/ --cov=src --cov-report=xml
python -c "import xml.etree.ElementTree as ET; tree = ET.parse('coverage.xml'); root = tree.getroot(); print(float(root.attrib['line-rate']) * 100)"
```

### 2. Lint & Format Workflow (`lint.yml`)

**Triggers:**
- Push to `main`, `develop`, `chore/reorg-codebase` branches
- Pull requests to `main`, `develop`

**What it does:**
- Checks code formatting with **Black**
- Checks import sorting with **isort**
- Lints code with **flake8**
- Type checks with **mypy** (non-blocking)
- Additional linting with **pylint** (non-blocking)
- Security checks with **bandit**

**Tools:**

| Tool | Purpose | Blocking |
|------|---------|----------|
| Black | Code formatting | Yes |
| isort | Import sorting | Yes |
| flake8 | Linting (syntax errors) | Yes |
| flake8 | Linting (style warnings) | No |
| mypy | Type checking | No |
| pylint | Code quality | No |
| bandit | Security issues | Yes |

**Fix issues locally:**
```bash
# Format code
black src/ scripts/ tests/

# Sort imports
isort src/ scripts/ tests/

# Check linting
flake8 src/ scripts/ tests/

# Type check
mypy src/ --ignore-missing-imports

# Security scan
bandit -r src/ -ll
```

### 3. Security Scan Workflow (`security-scan.yml`)

**Triggers:**
- Push to `main`, `develop`
- Pull requests to `main`, `develop`
- Weekly schedule (Mondays at 00:00 UTC)

**What it does:**
- **Dependency Scan**: Checks for vulnerabilities in dependencies
  - `safety check` - Python package vulnerabilities
  - `pip-audit` - Additional vulnerability scanning
- **Code Scan**: Static analysis for security issues
  - `bandit` - Python code security scan
- **Secrets Detection**: Scans for leaked secrets
  - `gitleaks` - Detects secrets in git history
- **Container Scan**: Scans Docker images for vulnerabilities
  - `trivy` - Container vulnerability scanner

**Run locally:**
```bash
# Check dependencies
pip install safety pip-audit
safety check
pip-audit

# Scan code
pip install bandit
bandit -r src/ scripts/ -ll

# Scan Docker image
docker run --rm -v $(pwd):/src aquasec/trivy:latest image avalanche-edna:latest
```

### 4. Build Docker Image Workflow (`build.yml`)

**Triggers:**
- Push to `main`, `develop`
- Tags matching `v*.*.*`
- Pull requests to `main`, `develop`

**What it does:**
- Builds Docker image with BuildKit caching
- Pushes to GitHub Container Registry (ghcr.io)
- Tags images appropriately:
  - `main` → `latest`
  - `develop` → `develop`
  - `v1.2.3` → `v1.2.3`, `v1.2`, `v1`, `1.2.3`
  - PR → `pr-123`
  - Commit → `main-sha123abc`
- Scans image with Trivy
- Uploads scan results to GitHub Security

**Image tags:**
```
ghcr.io/faisaltabrez/avalanche_edna:latest
ghcr.io/faisaltabrez/avalanche_edna:develop
ghcr.io/faisaltabrez/avalanche_edna:v1.0.0
ghcr.io/faisaltabrez/avalanche_edna:v1.0
ghcr.io/faisaltabrez/avalanche_edna:v1
ghcr.io/faisaltabrez/avalanche_edna:main-abc123
```

**Pull image:**
```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull latest
docker pull ghcr.io/faisaltabrez/avalanche_edna:latest

# Pull specific version
docker pull ghcr.io/faisaltabrez/avalanche_edna:v1.0.0
```

### 5. Deploy to Staging Workflow (`deploy-staging.yml`)

**Triggers:**
- Push to `develop` branch
- Manual trigger via workflow_dispatch

**What it does:**
- Builds and pushes Docker image tagged with `develop`
- Connects to staging server via SSH
- Pulls latest image
- Creates pre-deployment database backup
- Performs zero-downtime deployment
- Runs health checks
- Notifies via Slack

**Deployment strategy:**
```bash
# 1. Pull new image
docker pull ghcr.io/faisaltabrez/avalanche_edna:develop

# 2. Backup database
pg_dump > backup_pre_deploy.sql

# 3. Update containers (no downtime)
docker-compose up -d --no-deps --build app

# 4. Health check
curl -f https://staging.avalanche-edna.example.com/health
```

**Required secrets:**
- `STAGING_SSH_KEY` - SSH private key for staging server
- `STAGING_HOST` - Staging server hostname
- `STAGING_USER` - SSH username
- `SLACK_WEBHOOK` - Slack webhook URL for notifications

### 6. Deploy to Production Workflow (`deploy-production.yml`)

**Triggers:**
- Push tags matching `v*.*.*` (e.g., `v1.0.0`)
- Manual trigger with version input

**What it does:**
- Creates GitHub deployment
- Performs full system backup
- Pulls specific version image
- Performs blue-green deployment (rolling update)
- Runs smoke tests
- Updates deployment status
- Rollback automatically on failure
- Notifies via Slack
- Creates GitHub release

**Deployment strategy:**
```bash
# 1. Full backup
./scripts/backup/full_backup.sh

# 2. Pull versioned image
docker pull ghcr.io/faisaltabrez/avalanche_edna:v1.0.0

# 3. Blue-green deployment
docker-compose up -d --scale app=2 app  # Start new container
sleep 30  # Wait for health check
docker-compose up -d --scale app=1 app  # Remove old container

# 4. Smoke tests
curl -f https://avalanche-edna.example.com/health
curl -f https://avalanche-edna.example.com/api/status

# 5. Rollback on failure
python scripts/backup/restore_manager.py database --backup-id <latest> --force
docker-compose up -d --force-recreate app
```

**Required secrets:**
- `PRODUCTION_SSH_KEY` - SSH private key for production server
- `PRODUCTION_HOST` - Production server hostname
- `PRODUCTION_USER` - SSH username
- `SLACK_WEBHOOK` - Slack webhook URL for notifications

## Configuration Files

### `pyproject.toml`

Centralizes tool configuration:
- **Black**: Code formatter (line length: 127)
- **isort**: Import sorter (Black-compatible profile)
- **pylint**: Code linter with disabled rules
- **mypy**: Type checker configuration
- **pytest**: Test runner configuration
- **coverage**: Coverage reporting
- **bandit**: Security scanner

### `.flake8`

Configuration for flake8 linter:
- Max line length: 127
- Ignored rules: E203, E266, E501, W503 (Black-compatible)
- Max complexity: 10

### `.gitleaksignore`

Patterns to ignore for secret scanning:
- Test files
- Example/template files
- Documentation

## Badges

Add to README.md:

```markdown
![Tests](https://github.com/FaisalTabrez/Avalanche_eDNA/workflows/Tests/badge.svg)
![Lint](https://github.com/FaisalTabrez/Avalanche_eDNA/workflows/Lint%20&%20Format/badge.svg)
![Security](https://github.com/FaisalTabrez/Avalanche_eDNA/workflows/Security%20Scan/badge.svg)
![Build](https://github.com/FaisalTabrez/Avalanche_eDNA/workflows/Build%20Docker%20Image/badge.svg)
[![codecov](https://codecov.io/gh/FaisalTabrez/Avalanche_eDNA/branch/main/graph/badge.svg)](https://codecov.io/gh/FaisalTabrez/Avalanche_eDNA)
```

## Development Workflow

### Feature Development

```bash
# 1. Create feature branch
git checkout -b feature/my-feature develop

# 2. Make changes
# ... code ...

# 3. Format and lint locally
black src/ scripts/ tests/
isort src/ scripts/ tests/
flake8 src/ scripts/ tests/

# 4. Run tests
pytest tests/ -v --cov=src

# 5. Commit and push
git add .
git commit -m "feat: add new feature"
git push origin feature/my-feature

# 6. Create pull request to develop
# GitHub Actions will run tests, lint, and security scans

# 7. Merge to develop
# Automatic deployment to staging

# 8. Test on staging
# https://staging.avalanche-edna.example.com
```

### Release Process

```bash
# 1. Ensure develop is tested on staging
# 2. Merge develop to main
git checkout main
git merge develop
git push origin main

# 3. Create release tag
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# 4. Automatic deployment to production
# GitHub Actions will deploy tagged version

# 5. Monitor deployment
# Check GitHub Actions logs
# Verify production: https://avalanche-edna.example.com

# 6. If issues arise, workflow auto-rollback
# Or manually rollback:
git revert HEAD
git push origin main
```

## Environment Setup

### Required GitHub Secrets

Navigate to: **Settings → Secrets and variables → Actions**

#### Staging Environment

| Secret | Description |
|--------|-------------|
| `STAGING_SSH_KEY` | SSH private key for staging server |
| `STAGING_HOST` | Staging server hostname (e.g., staging.example.com) |
| `STAGING_USER` | SSH username (e.g., deploy) |

#### Production Environment

| Secret | Description |
|--------|-------------|
| `PRODUCTION_SSH_KEY` | SSH private key for production server |
| `PRODUCTION_HOST` | Production server hostname (e.g., example.com) |
| `PRODUCTION_USER` | SSH username (e.g., deploy) |

#### Notifications

| Secret | Description |
|--------|-------------|
| `SLACK_WEBHOOK` | Slack incoming webhook URL |

#### Optional

| Secret | Description |
|--------|-------------|
| `GITLEAKS_LICENSE` | Gitleaks Pro license (for advanced features) |

### SSH Key Setup

```bash
# 1. Generate SSH key pair
ssh-keygen -t ed25519 -C "github-actions@avalanche" -f ~/.ssh/avalanche_deploy

# 2. Add public key to server
ssh-copy-id -i ~/.ssh/avalanche_deploy.pub deploy@server.com

# 3. Add private key to GitHub Secrets
cat ~/.ssh/avalanche_deploy | pbcopy  # macOS
cat ~/.ssh/avalanche_deploy | clip  # Windows
cat ~/.ssh/avalanche_deploy | xclip  # Linux

# Paste into GitHub Settings → Secrets → STAGING_SSH_KEY
```

### Server Setup

On staging/production servers:

```bash
# 1. Create deployment directory
sudo mkdir -p /opt/avalanche
sudo chown deploy:deploy /opt/avalanche

# 2. Install Docker & Docker Compose
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker deploy

# 3. Clone repository
cd /opt/avalanche
git clone https://github.com/FaisalTabrez/Avalanche_eDNA.git .

# 4. Configure environment
cp .env.example .env
nano .env  # Edit configuration

# 5. Create docker-compose.staging.yml or docker-compose.prod.yml
# (See Deployment section for examples)

# 6. Initial deployment
docker-compose up -d
```

## Monitoring & Alerts

### GitHub Actions Notifications

- **Slack**: Receive deployment notifications
- **Email**: GitHub sends emails on workflow failures
- **GitHub UI**: View workflow status in Actions tab

### Codecov Integration

1. **Sign up**: https://codecov.io
2. **Connect repository**: Avalanche_eDNA
3. **Get token**: Copy CODECOV_TOKEN
4. **Add to GitHub Secrets**: (Optional, public repos don't need it)

Coverage reports appear in PR comments automatically.

### Security Alerts

- **Dependabot**: Automatic dependency updates and vulnerability alerts
- **Code Scanning**: Trivy and Bandit results in Security tab
- **Secret Scanning**: GitHub detects committed secrets

Enable in: **Settings → Security → Code security and analysis**

## Troubleshooting

### Tests Failing

```bash
# Run tests locally first
pytest tests/ -v -x  # Stop on first failure

# Check specific test
pytest tests/test_specific.py::test_function -v

# Debug test
pytest tests/test_specific.py::test_function -v -s  # Show print statements
```

### Linting Errors

```bash
# Auto-fix formatting
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Check what needs fixing
flake8 src/ scripts/ tests/ --show-source
```

### Build Failures

```bash
# Test Docker build locally
docker build -t avalanche-edna:test .

# Check build logs
docker build --no-cache -t avalanche-edna:test .
```

### Deployment Failures

```bash
# Check GitHub Actions logs
# Go to Actions tab → Failed workflow → Click on job

# SSH to server and check
ssh deploy@server.com
cd /opt/avalanche
docker-compose ps
docker-compose logs app

# Manual rollback
./scripts/backup/restore_manager.py database --backup-id <backup-id> --force
docker-compose up -d --force-recreate
```

### Coverage Below Threshold

```bash
# Check coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Add tests for uncovered code
# Increase coverage above 70%
```

## Best Practices

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new feature
fix: fix bug in module
docs: update documentation
style: format code
refactor: refactor component
test: add tests
chore: update dependencies
ci: update workflow
```

### Pull Requests

- **Keep PRs small**: < 500 lines changed
- **Write descriptive titles**: "feat: add user authentication"
- **Add description**: Explain what and why
- **Link issues**: "Closes #123"
- **Wait for CI**: All checks must pass
- **Request review**: At least 1 reviewer

### Testing

- **Write tests first**: TDD approach
- **Test edge cases**: Not just happy path
- **Mock external services**: Don't hit real APIs
- **Use fixtures**: pytest fixtures for setup
- **Maintain coverage**: Keep above 70%

### Security

- **Never commit secrets**: Use environment variables
- **Review dependencies**: Check Dependabot alerts
- **Scan regularly**: Weekly security scans
- **Update dependencies**: Keep packages current
- **Use HTTPS**: Always encrypt in transit

## References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Conventional Commits](https://www.conventionalcommits.org/)
- [Python Testing Best Practices](https://docs.pytest.org/en/latest/goodpractices.html)
