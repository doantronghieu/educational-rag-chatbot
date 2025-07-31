"""
LangChain vector store implementation with unified service-level orchestration.
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import logging
import uuid
from typing import List, Optional, Dict, Any, Union, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client.models import Filter, FieldCondition, MatchValue
from clients.vector_store import VectorStoreClient
from utils.decorators import handle_errors

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Unified vector store service with LangChain integration and multi-collection management."""
    
    def __init__(
        self,
        vector_store_client: VectorStoreClient,
        embedding_model: Embeddings,
        default_collection: str = "documents"
    ):
        """Initialize vector store service.
        
        Args:
            vector_store_client: VectorStoreClient instance for collection management
            embedding_model: Embedding model for text vectorization
            default_collection: Default collection name
        """
        self.client = vector_store_client
        self.embedding_model = embedding_model
        self.default_collection = default_collection
        self._vector_stores: Dict[str, QdrantVectorStore] = {}
    
    def _convert_filter(self, filter_dict: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Convert simple dict filter to Qdrant Filter format.
        
        LangChain stores document metadata under the 'metadata' key in Qdrant's payload,
        so filters need to reference 'metadata.field_name' rather than just 'field_name'.
        
        Args:
            filter_dict: Simple key-value filter dict (e.g., {"type": "programming"})
                        This will be converted to metadata.type = "programming"
            
        Returns:
            Qdrant Filter object or None
        """
        if not filter_dict:
            return None
        
        conditions = []
        for key, value in filter_dict.items():
            # LangChain stores metadata under 'metadata' key in Qdrant payload
            field_key = f"metadata.{key}"
            conditions.append(
                FieldCondition(
                    key=field_key,
                    match=MatchValue(value=value)
                )
            )
        
        return Filter(must=conditions)
    
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from the embedding model.
        
        Returns:
            Vector dimension
        """
        
        # Default to 1536 for OpenAI
        return 1536
    
    async def get_vector_store(self, collection_name: Optional[str] = None) -> QdrantVectorStore:
        """Get or create LangChain vector store for collection.
        
        Args:
            collection_name: Collection name, defaults to default_collection
            
        Returns:
            QdrantVectorStore instance
        """
        collection_name = collection_name or self.default_collection
        
        if collection_name not in self._vector_stores:
            # Ensure collection exists before creating wrapper
            vector_size = self._get_embedding_dimension()
            await self.client.ensure_collection_exists(
                collection_name=collection_name,
                vector_size=vector_size
            )
            
            # Create LangChain vector store using sync client
            self._vector_stores[collection_name] = QdrantVectorStore(
                client=self.client.sync_client,
                collection_name=collection_name,
                embedding=self.embedding_model
            )
        
        return self._vector_stores[collection_name]
    
    # Document and text management methods
    @handle_errors("Add documents")
    async def add_documents(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to vector store with automatic ID generation.
        
        Args:
            documents: List of documents to add
            collection_name: Target collection name
            document_ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        collection_name = collection_name or self.default_collection
        vector_store = await self.get_vector_store(collection_name)
        
        # Generate IDs if not provided
        if not document_ids:
            document_ids = [str(uuid.uuid4()) for _ in documents]
        
        # Add documents to vector store
        result_ids = await vector_store.aadd_documents(documents, ids=document_ids)
        
        logger.info(
            f"Added {len(documents)} documents to collection '{collection_name}'"
        )
        return result_ids
    
    @handle_errors("Add texts")
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts with metadata to vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            collection_name: Target collection name
            document_ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        # Convert texts to documents
        documents = [
            Document(
                page_content=text,
                metadata=metadatas[i] if metadatas else {}
            )
            for i, text in enumerate(texts)
        ]
        
        return await self.add_documents(
            documents=documents,
            collection_name=collection_name,
            document_ids=document_ids
        )
    
    # Search methods
    @handle_errors("Similarity search")
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        collection_name: Optional[str] = None,
        filter_conditions: Optional[Union[Dict[str, Any], Filter]] = None,
        score_threshold: Optional[float] = None,
        with_scores: bool = False
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Unified similarity search method.
        
        Args:
            query: Search query
            k: Number of documents to return
            collection_name: Collection to search in
            filter_conditions: Optional metadata filter (dict or Qdrant Filter)
            score_threshold: Minimum similarity score
            with_scores: Whether to return scores with documents
            
        Returns:
            List of documents or (document, score) tuples based on with_scores
        """
        collection_name = collection_name or self.default_collection
        vector_store = await self.get_vector_store(collection_name)
        
        # Convert dict filter to Qdrant Filter if needed
        if isinstance(filter_conditions, dict):
            filter_conditions = self._convert_filter(filter_conditions)
        
        if with_scores:
            # Perform similarity search with scores
            results = await vector_store.asimilarity_search_with_score(
                query=query,
                k=k,
                filter=filter_conditions
            )
            
            # Filter by score threshold if provided
            if score_threshold is not None:
                results = [
                    (doc, score) for doc, score in results
                    if score >= score_threshold
                ]
            
            logger.info(
                f"Found {len(results)} similar documents with scores for query in '{collection_name}'"
            )
        else:
            # Perform similarity search without scores
            results = await vector_store.asimilarity_search(
                query=query,
                k=k,
                filter=filter_conditions
            )
            
            logger.info(
                f"Found {len(results)} similar documents for query in '{collection_name}'"
            )
        
        return results
    
    @handle_errors("Delete documents")
    async def delete_documents(
        self,
        document_ids: List[str],
        collection_name: Optional[str] = None
    ) -> bool:
        """Delete documents by IDs.
        
        Args:
            document_ids: List of document IDs to delete
            collection_name: Collection name
            
        Returns:
            True if successful
        """
        collection_name = collection_name or self.default_collection
        vector_store = await self.get_vector_store(collection_name)
        
        # Delete documents
        result = await vector_store.adelete(document_ids)
        
        logger.info(
            f"Deleted {len(document_ids)} documents from collection '{collection_name}'"
        )
        return result
    
    @handle_errors("Get retriever")
    async def get_retriever(
        self,
        collection_name: Optional[str] = None,
        search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Get a retriever for the vector store.
        
        Args:
            collection_name: Collection name
            search_kwargs: Search configuration
            
        Returns:
            VectorStoreRetriever instance
        """
        collection_name = collection_name or self.default_collection
        vector_store = await self.get_vector_store(collection_name)
        
        # Default search kwargs
        search_kwargs = search_kwargs or {"k": 4}
        
        retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
        
        logger.info(f"Created retriever for collection '{collection_name}'")
        return retriever
    
    # Class factory methods for convenience
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],  # noqa: F401 - kept for API compatibility
        vector_store_client,
        embedding_model: Embeddings,
        collection_name: str = "documents"
    ) -> "VectorStoreService":
        """Create service instance for adding documents.
        
        Args:
            documents: List of documents (for reference, actual addition needs async call)
            vector_store_client: VectorStoreClient instance
            embedding_model: Embedding model
            collection_name: Collection name
            
        Returns:
            VectorStoreService instance
        """
        # Note: documents parameter kept for API compatibility but not used directly
        # since add_documents is async and needs to be called separately
        return cls(
            vector_store_client=vector_store_client,
            embedding_model=embedding_model,
            default_collection=collection_name
        )
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        vector_store_client,
        embedding_model: Embeddings,
        collection_name: str = "documents",
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> "VectorStoreService":
        """Create service instance and add texts.
        
        Args:
            texts: List of text strings
            vector_store_client: VectorStoreClient instance
            embedding_model: Embedding model
            collection_name: Collection name
            metadatas: Optional list of metadata dicts
            
        Returns:
            VectorStoreService instance
        """
        documents = [
            Document(
                page_content=text,
                metadata=metadatas[i] if metadatas else {}
            )
            for i, text in enumerate(texts)
        ]
        return cls.from_documents(
            documents=documents,
            vector_store_client=vector_store_client,
            embedding_model=embedding_model,
            collection_name=collection_name
        )
