# Educational Chatbot - Development Utilities

.PHONY: help setup dev clean clean-all prisma-generate prisma-setup prisma-reset prisma-studio prisma-push infra-up infra-down infra-logs infra-status install test test-backend test-frontend lint lint-backend lint-frontend format format-backend format-frontend build quick-reset

# Default target
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Environment setup
setup: ## Set up the development environment
	@echo "Setting up development environment..."
	@echo "Creating symlinks for .env..."
	@ln -sf ../../.env infra/docker/.env
	@ln -sf ../.env backend/.env
	@ln -sf ../.env frontend/.env.local
	@echo "Installing dependencies..."
	@$(MAKE) install
	@echo "Setting up optional NLP models..."
	@echo "📚 Downloading SpaCy English model (required for SpaCy text splitter)..."
	@uv run python -m spacy download en_core_web_sm || echo "⚠️  SpaCy model download failed - SpaCy splitter will be skipped in tests"
	@echo "🤖 Pre-caching SentenceTransformers model (improves test performance)..."
	@uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2'); print('✅ SentenceTransformers model cached')" || echo "⚠️  SentenceTransformers model caching failed - will download on first use"
	@echo "📖 Downloading essential NLTK data..."
	@uv run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); print('✅ NLTK data downloaded')" || echo "⚠️  NLTK data download failed - NLTK splitter may not work fully"
	@echo "Setting up infrastructure..."
	@$(MAKE) infra-up
	@echo "Setting up database..."
	@$(MAKE) prisma-setup
	@echo "✅ Setup complete!"

install: ## Install all dependencies
	@echo "Installing Python dependencies with uv..."
	@uv sync
	@echo "Installing frontend dependencies..."
	@cd frontend && npm install

# Development
dev: ## Start development servers
	@echo "Starting development environment..."
	@echo "Make sure infrastructure is running: make infra-up"
	@echo ""
	@echo "Start backend (in new terminal): cd backend && uv run uvicorn main:app --reload"
	@echo "Start frontend (in new terminal): cd frontend && npm run dev"

# Infrastructure
infra-up: ## Start infrastructure services (Qdrant, PostgreSQL, MinIO)
	@echo "Starting infrastructure services..."
	@cd infra/docker && docker-compose up -d
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "✅ Infrastructure services started"
	@echo "  - Qdrant: http://localhost:6333/dashboard"
	@echo "  - MinIO Console: http://localhost:9001"
	@echo "  - PostgreSQL: localhost:5432"

infra-down: ## Stop infrastructure services
	@echo "Stopping infrastructure services..."
	@cd infra/docker && docker-compose down

infra-logs: ## View infrastructure service logs
	@cd infra/docker && docker-compose logs -f

infra-status: ## Check infrastructure service status
	@cd infra/docker && docker-compose ps

# Prisma Database Operations
prisma-generate: ## Generate Prisma clients
	@echo "Generating Prisma clients..."
	@npx prisma generate

prisma-setup: ## Set up Prisma (generate clients + run migrations)
	@echo "Setting up Prisma..."
	@echo "Generating Prisma clients..."
	@npx prisma generate
	@echo "Running database migrations..."
	@npx prisma migrate dev --name init
	@echo "✅ Prisma setup complete"

prisma-reset: ## Reset database and run migrations from scratch
	@echo "⚠️  This will delete all data in the database!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		npx prisma migrate reset --force; \
		echo "✅ Database reset complete"; \
	else \
		echo ""; \
		echo "Cancelled"; \
	fi

prisma-studio: ## Open Prisma Studio (database GUI)
	@echo "Opening Prisma Studio..."
	@npx prisma studio

prisma-push: ## Push schema changes to database (for development)
	@echo "Pushing schema changes to database..."
	@npx prisma db push

# Testing
test-backend: ## Run backend tests
	@echo "Running backend tests..."
	@echo "No tests found. Add test files to run tests."

test-frontend: ## Run frontend tests
	@echo "Running frontend tests..."
	@echo "No tests found. Add test files to run tests."

lint-backend: ## Run backend linting
	@echo "Running backend linting..."
	@cd backend && uv run ruff check --exclude clients/prisma

lint-frontend: ## Run frontend linting
	@echo "Running frontend linting..."
	@cd frontend && npm run lint

format-backend: ## Format backend code
	@echo "Formatting backend code..."
	@cd backend && uv run ruff format --exclude clients/prisma

format-frontend: ## Format frontend code
	@echo "Formatting frontend code..."
	@cd frontend && npm run lint -- --fix

# Cleanup
clean: ## Clean up generated files and caches
	@echo "Cleaning up..."
	@rm -rf frontend/.next/
	@rm -rf frontend/node_modules/.cache/
	@echo "✅ Cleanup complete"

clean-all: clean infra-down ## Clean everything and stop services
	@echo "Cleaning everything..."
	@cd infra/docker && docker-compose down -v
	@docker system prune -f
	@echo "✅ Full cleanup complete"

# Production
build: ## Build for production
	@echo "Building for production..."
	@echo "Building frontend..."
	@cd frontend && npm run build
	@echo "✅ Build complete"

# Quick development cycle
quick-reset: ## Quick reset for development (reset DB + restart)
	@echo "Quick development reset..."
	@$(MAKE) prisma-reset
	@$(MAKE) prisma-generate
	@echo "✅ Ready for development"

# Convenience commands
test: test-backend test-frontend ## Run all tests

lint: lint-backend lint-frontend ## Run all linting

format: format-backend format-frontend ## Format all code