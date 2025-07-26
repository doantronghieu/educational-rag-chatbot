# Setup Guide

## Tech Stack

### Backend
- **FastAPI** - Modern, fast (high-performance) web framework for building APIs with Python
- **UV** - Ultra-fast Python package installer and resolver for managing Python environments

### Frontend
- **Next.js** - React framework for production-grade applications
- **shadcn/ui** - Modern UI component library built on Radix UI and Tailwind CSS

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js 18+
- UV package manager

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
1. Start the backend server (runs on http://localhost:8000)
2. Start the frontend development server (runs on http://localhost:3000)
3. The frontend will proxy API requests to the backend

### Additional Commands

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