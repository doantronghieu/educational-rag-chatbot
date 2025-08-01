# Educational Chatbot

A RAG (Retrieval-Augmented Generation) chatbot designed for primary school students, built with a modern full-stack architecture.

## Documentation

- **[Product Requirements](docs/prd.md)** - Comprehensive project requirements and specifications
- **[Setup Guide](docs/setup.md)** - Tech stack details and development setup instructions

## Quick Start

For detailed setup instructions, see [docs/setup.md](docs/setup.md).

```bash
# Backend
cd backend && uv sync && uv run uvicorn main:app --reload

# Frontend  
cd frontend && npm install && npm run dev
```

## Project Structure
```
heyhi/
├── backend/         # FastAPI application
├── frontend/        # Next.js application  
├── docs/           # Project documentation
│   ├── prd.md      # Product Requirements Document
│   └── setup.md    # Setup and tech stack guide
├── infra/          # Infrastructure configuration
└── README.md       # This file
```