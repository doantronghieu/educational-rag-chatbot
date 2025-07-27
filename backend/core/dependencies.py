"""FastAPI dependency injection setup."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import Depends
from typing import Annotated

from clients.database import get_db, Prisma
from clients.vector_store import VectorStoreClient
from clients.minio import StorageClient
from clients.llm import LlmClient
from services.vector_store import VectorStoreService
from libs.langchain.embeddings import get_default_embeddings
from core.config import settings


# Database dependency
def get_database() -> Prisma:
    """Get database client."""
    return get_db()


def get_vector_store_client() -> VectorStoreClient:
    """Get vector store client instance."""
    return VectorStoreClient(
        url=settings.vector_store_url,
        api_key=settings.vector_store_api_key if settings.vector_store_api_key else None,
        timeout=settings.vector_store_timeout
    )


def get_storage_client() -> StorageClient:
    """Get storage client instance."""
    return StorageClient(
        endpoint=settings.storage_endpoint,
        access_key=settings.storage_access_key,
        secret_key=settings.storage_secret_key,
        bucket=settings.storage_bucket,
    )


def get_llm_client() -> LlmClient:
    """Get LLM client instance."""
    return LlmClient()


def get_vector_store_service() -> VectorStoreService:
    """Get vector store service instance."""
    vector_client = get_vector_store_client()
    embeddings = get_default_embeddings()
    return VectorStoreService(
        vector_store_client=vector_client,
        embedding_model=embeddings
    )


# Type annotations for dependency injection
DatabaseDep = Annotated[Prisma, Depends(get_database)]
VectorStoreDep = Annotated[VectorStoreClient, Depends(get_vector_store_client)]
VectorStoreServiceDep = Annotated[VectorStoreService, Depends(get_vector_store_service)]
StorageDep = Annotated[StorageClient, Depends(get_storage_client)]
LlmDep = Annotated[LlmClient, Depends(get_llm_client)]
