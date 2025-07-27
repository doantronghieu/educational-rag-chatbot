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
from core.config import settings


# Database dependency
def get_database() -> Prisma:
    """Get database client."""
    return get_db()


def get_vector_store_client() -> VectorStoreClient:
    """Get vector store client instance."""
    return VectorStoreClient(url=settings.vector_store_url)


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


# Type annotations for dependency injection
DatabaseDep = Annotated[Prisma, Depends(get_database)]
VectorStoreDep = Annotated[VectorStoreClient, Depends(get_vector_store_client)]
StorageDep = Annotated[StorageClient, Depends(get_storage_client)]
LlmDep = Annotated[LlmClient, Depends(get_llm_client)]
