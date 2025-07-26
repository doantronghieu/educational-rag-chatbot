# Setup Guide

## Tech Stack

### Backend
- **FastAPI** - Modern, fast (high-performance) web framework for building APIs with Python
- **UV** - Ultra-fast Python package installer and resolver for managing Python environments

### Frontend
- **Next.js** - React framework for production-grade applications
- **shadcn/ui** - Modern UI component library built on Radix UI and Tailwind CSS

### Infrastructure
- **Qdrant** - Vector database for semantic search and embeddings
- **PostgreSQL** - Relational database for chat history and metadata
- **MinIO** - S3-compatible object storage for document storage

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- UV package manager
- Docker and Docker Compose

### Environment Setup
Create symlinks for environment variables:
```bash
# Create symlinks for .env file (run from project root)
ln -sf ../../.env infra/docker/.env
ln -sf ../.env backend/.env  
ln -sf ../.env frontend/.env.local
```

### Infrastructure Setup
Start the required services (Qdrant, PostgreSQL, MinIO):
```bash
cd infra/docker
docker-compose up -d
```

### Backend Setup
```bash
cd backend
uv sync
uv run uvicorn main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Development Workflow

### Running the Full Stack
1. Start infrastructure services (runs Qdrant on :6333, PostgreSQL on :5432, MinIO on :9000-9001)
2. Start the backend server (runs on http://localhost:8000)
3. Start the frontend development server (runs on http://localhost:3000)
4. The frontend will proxy API requests to the backend

### Additional Commands

#### Infrastructure
```bash
# Start services
cd infra/docker && docker-compose up -d

# Stop services
cd infra/docker && docker-compose down

# View service logs
cd infra/docker && docker-compose logs -f

# Check service status
cd infra/docker && docker-compose ps

# Access service dashboards
# Qdrant Web UI: http://localhost:6333/dashboard
# MinIO Console: http://localhost:9001 (login with MINIO_ROOT_USER/MINIO_ROOT_PASSWORD)
```

#### Backend
```bash
# Install dependencies
uv sync

# Run with auto-reload
uv run uvicorn main:app --reload

# Run tests (if available)
uv run pytest

# Check code formatting
uv run ruff check
```

#### Frontend
```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint
```