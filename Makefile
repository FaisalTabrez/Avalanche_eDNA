.PHONY: help install install-dev test test-coverage lint format security clean docker-build docker-up docker-down backup

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev: ## Install development dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install pre-commit
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-coverage: ## Run tests with coverage report
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

test-unit: ## Run only unit tests
	pytest tests/ -v -m unit

test-integration: ## Run only integration tests
	pytest tests/ -v -m integration

lint: ## Run all linters
	@echo "Running Black..."
	black --check src/ scripts/ tests/
	@echo "\nRunning isort..."
	isort --check-only src/ scripts/ tests/
	@echo "\nRunning flake8..."
	flake8 src/ scripts/ tests/
	@echo "\nRunning mypy..."
	mypy src/ --ignore-missing-imports || true
	@echo "\nRunning pylint..."
	pylint src/ --disable=C0111,C0103,R0913,R0914,W0212,C0301 || true

format: ## Format code with Black and isort
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

security: ## Run security scans
	@echo "Running Bandit..."
	bandit -r src/ scripts/ -ll
	@echo "\nRunning Safety..."
	safety check || true
	@echo "\nRunning pip-audit..."
	pip-audit || true

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean: ## Clean up temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name "*.log" -delete 2>/dev/null || true
	@echo "Cleaned up temporary files"

docker-build: ## Build Docker image
	docker build -t avalanche-edna:latest .

docker-build-prod: ## Build production Docker image
	docker build -f Dockerfile -t avalanche-edna:prod .

docker-up: ## Start Docker containers
	docker-compose up -d

docker-down: ## Stop Docker containers
	docker-compose down

docker-logs: ## View Docker container logs
	docker-compose logs -f

docker-shell: ## Open shell in app container
	docker-compose exec app /bin/bash

docker-db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U avalanche avalanche_edna

backup-db: ## Backup database
	python scripts/backup/backup_manager.py database

backup-files: ## Backup files
	python scripts/backup/backup_manager.py files

backup-full: ## Full system backup
	python scripts/backup/backup_manager.py full

backup-list: ## List all backups
	python scripts/backup/backup_manager.py list

restore-db: ## Restore database (requires BACKUP_ID)
	@if [ -z "$(BACKUP_ID)" ]; then \
		echo "Error: BACKUP_ID not specified"; \
		echo "Usage: make restore-db BACKUP_ID=database_20241122_020000"; \
		exit 1; \
	fi
	python scripts/backup/restore_manager.py database --backup-id $(BACKUP_ID)

run-dev: ## Run development server
	streamlit run streamlit_app.py

run-prod: ## Run production server with gunicorn
	gunicorn -c gunicorn.conf.py streamlit_app:app

migrate-db: ## Run database migration to PostgreSQL
	python scripts/migrate_to_postgres.py --migrate

db-shell-sqlite: ## Open SQLite shell
	sqlite3 data/avalanche.db

db-shell-postgres: ## Open PostgreSQL shell
	PGPASSWORD=$(DB_PASSWORD) psql -h $(DB_HOST) -p $(DB_PORT) -U $(DB_USER) -d $(DB_NAME)

analysis: ## Run analysis on sample data
	python scripts/analyze_dataset.py data/sample/sample_edna_sequences.fasta analysis_output.txt

demo: ## Run demo analysis
	python scripts/run_demo.py

setup-env: ## Copy .env.example to .env
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "Created .env file. Please edit it with your configuration."; \
	else \
		echo ".env file already exists"; \
	fi

init: setup-env install-dev ## Initialize development environment
	@echo "Development environment initialized!"
	@echo "Next steps:"
	@echo "  1. Edit .env with your configuration"
	@echo "  2. Run 'make docker-up' to start services"
	@echo "  3. Run 'make test' to verify setup"

ci-test: ## Run CI test suite (same as GitHub Actions)
	pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing

ci-lint: ## Run CI linting (same as GitHub Actions)
	black --check --diff src/ scripts/ tests/
	isort --check-only --diff src/ scripts/ tests/
	flake8 src/ scripts/ tests/
	bandit -r src/ scripts/ -ll

ci-security: ## Run CI security scans (same as GitHub Actions)
	safety check
	pip-audit
	bandit -r src/ scripts/ -ll
