"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from clients.database import connect_db, disconnect_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await connect_db()
    yield
    # Shutdown
    await disconnect_db()


# Create FastAPI app
app = FastAPI(
    title="Educational Chatbot API",
    description="Educational Chatbot API for primary school students",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Educational Chatbot API", "status": "ready"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "backend", "version": "0.1.0"}


# TODO: Add routers when implementing features
# app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
# app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
