"""FastAPI dependency injection setup."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi import Depends
from typing import Annotated

from clients.database import get_db, Prisma
from clients.vector_store import VectorStoreClient
from clients.minio import StorageClient
from libs.langchain.models import ChatModel
from libs.langchain.vector_stores import VectorStoreService
from libs.langchain.embeddings import EmbeddingService
from core.config import settings

# Global singleton instances
_llm_instance = None
_embeddings_instance = None
_vector_store_client_instance = None
_storage_client_instance = None
_vector_store_service_instance = None


# Database dependency
def get_database() -> Prisma:
    """Get database client."""
    return get_db()


def get_vector_store_client() -> VectorStoreClient:
    """Get vector store client instance (singleton)."""
    global _vector_store_client_instance
    if _vector_store_client_instance is None:
        _vector_store_client_instance = VectorStoreClient(
            url=settings.vector_store_url,
            api_key=settings.vector_store_api_key if settings.vector_store_api_key else None,
            timeout=settings.vector_store_timeout
        )
    return _vector_store_client_instance


def get_storage_client() -> StorageClient:
    """Get storage client instance (singleton)."""
    global _storage_client_instance
    if _storage_client_instance is None:
        _storage_client_instance = StorageClient(
            endpoint=settings.storage_endpoint,
            access_key=settings.storage_access_key,
            secret_key=settings.storage_secret_key,
            bucket=settings.storage_bucket,
        )
    return _storage_client_instance


def get_llm() -> ChatModel:
    """Get LangChain ChatModel instance (singleton)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = ChatModel()
    return _llm_instance


def get_embeddings() -> EmbeddingService:
    """Get embeddings service instance (singleton)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = EmbeddingService()
    return _embeddings_instance.embeddings


def get_vector_store_service() -> VectorStoreService:
    """Get vector store service instance (singleton)."""
    global _vector_store_service_instance
    if _vector_store_service_instance is None:
        vector_client = get_vector_store_client()
        embedding_service = get_embeddings()
        _vector_store_service_instance = VectorStoreService(
            vector_store_client=vector_client,
            embedding_model=embedding_service
        )
    return _vector_store_service_instance


# Type annotations for dependency injection
DatabaseDep = Annotated[Prisma, Depends(get_database)]
VectorStoreDep = Annotated[VectorStoreClient, Depends(get_vector_store_client)]
VectorStoreServiceDep = Annotated[VectorStoreService, Depends(get_vector_store_service)]
StorageDep = Annotated[StorageClient, Depends(get_storage_client)]
EmbeddingsDep = Annotated[EmbeddingService, Depends(get_embeddings)]
LlmDep = Annotated[ChatModel, Depends(get_llm)]
