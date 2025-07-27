"""
Vector store service layer for document storage and retrieval operations.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
import uuid
from typing import List, Optional, Dict, Any, Tuple, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from clients.vector_store import VectorStoreClient
from libs.langchain.vector_stores import QdrantVectorStoreWrapper
from core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing vector store operations."""
    
    def __init__(
        self,
        vector_store_client: VectorStoreClient,
        embedding_model: Embeddings,
        default_collection: str = "documents"
    ):
        """Initialize vector store service.
        
        Args:
            vector_store_client: Qdrant client instance
            embedding_model: Embedding model for text vectorization
            default_collection: Default collection name
        """
        self.client = vector_store_client
        self.embedding_model = embedding_model
        self.default_collection = default_collection
        self._vector_stores: Dict[str, QdrantVectorStoreWrapper] = {}
    
    def get_vector_store(self, collection_name: Optional[str] = None) -> QdrantVectorStoreWrapper:
        """Get or create vector store for collection.
        
        Args:
            collection_name: Collection name, defaults to default_collection
            
        Returns:
            QdrantVectorStoreWrapper instance
        """
        collection_name = collection_name or self.default_collection
        
        if collection_name not in self._vector_stores:
            # Ensure collection exists before creating wrapper
            if not self.client.collection_exists(collection_name):
                vector_size = self._get_embedding_dimension()
                
                self.client.create_collection(
                    collection_name=collection_name,
                    vector_size=vector_size
                )
            
            self._vector_stores[collection_name] = QdrantVectorStoreWrapper(
                client=self.client.client,
                collection_name=collection_name,
                embedding=self.embedding_model
            )
        
        return self._vector_stores[collection_name]
    
    def _get_embedding_dimension(self) -> int:
        """Get embedding dimension from the embedding model.
        
        Returns:
            Vector dimension
        """
        # Try to get dimension from OpenAI embedding model
        if hasattr(self.embedding_model, 'model'):
            model_name = self.embedding_model.model
            model_dimensions = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            return model_dimensions.get(model_name, 1536)
        
        # Default to 1536 for OpenAI ada-002
        return 1536
    
    async def add_documents(
        self,
        documents: List[Document],
        collection_name: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add documents to vector store.
        
        Args:
            documents: List of documents to add
            collection_name: Target collection name
            document_ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        try:
            collection_name = collection_name or self.default_collection
            vector_store = self.get_vector_store(collection_name)
            
            # Generate IDs if not provided
            if not document_ids:
                document_ids = [str(uuid.uuid4()) for _ in documents]
            
            # Add documents to vector store
            result_ids = vector_store.add_documents(documents, ids=document_ids)
            
            logger.info(
                f"Added {len(documents)} documents to collection '{collection_name}'"
            )
            return result_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
        document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """Add texts to vector store.
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            collection_name: Target collection name
            document_ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        try:
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
            
        except Exception as e:
            logger.error(f"Failed to add texts to vector store: {e}")
            raise
    
    async def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
        collection_name: Optional[str] = None,
        filter_conditions: Optional[Union[Dict[str, Any], Any]] = None,
        score_threshold: Optional[float] = None,
        with_scores: bool = False
    ) -> Union[List[Document], List[Tuple[Document, float]]]:
        """Unified similarity search method.
        
        Args:
            query: Search query
            k: Number of documents to return
            collection_name: Collection to search in
            filter_conditions: Optional metadata filter
            score_threshold: Minimum similarity score
            with_scores: Whether to return scores with documents
            
        Returns:
            List of documents or (document, score) tuples based on with_scores
        """
        try:
            collection_name = collection_name or self.default_collection
            k = k or settings.default_search_k
            vector_store = self.get_vector_store(collection_name)
            
            if with_scores:
                # Perform similarity search with scores
                results = vector_store.similarity_search_with_score(
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
                results = vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_conditions
                )
                
                logger.info(
                    f"Found {len(results)} similar documents for query in '{collection_name}'"
                )
            
            return results
            
        except Exception as e:
            error_msg = f"Failed to perform similarity search{'with scores' if with_scores else ''}: {e}"
            logger.error(error_msg)
            raise
    
    async def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None,
        collection_name: Optional[str] = None,
        filter_conditions: Optional[Union[Dict[str, Any], Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            collection_name: Collection to search in
            filter_conditions: Optional metadata filter
            score_threshold: Minimum similarity score
            
        Returns:
            List of (document, score) tuples
        """
        return await self.similarity_search(
            query=query,
            k=k,
            collection_name=collection_name,
            filter_conditions=filter_conditions,
            score_threshold=score_threshold,
            with_scores=True
        )
    
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
        try:
            collection_name = collection_name or self.default_collection
            vector_store = self.get_vector_store(collection_name)
            
            # Delete documents
            result = vector_store.delete(document_ids)
            
            logger.info(
                f"Deleted {len(document_ids)} documents from collection '{collection_name}'"
            )
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise
    
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
        try:
            collection_name = collection_name or self.default_collection
            vector_store = self.get_vector_store(collection_name)
            
            # Default search kwargs
            if search_kwargs is None:
                search_kwargs = {"k": settings.default_search_k}
            
            retriever = vector_store.get_retriever(search_kwargs=search_kwargs)
            
            logger.info(f"Created retriever for collection '{collection_name}'")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create retriever: {e}")
            raise
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """Get collection statistics.
        
        Args:
            collection_name: Collection name
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection_name = collection_name or self.default_collection
            
            # Get collection info from client
            info = self.client.get_collection_info(collection_name)
            if not info:
                return {"exists": False}
            
            # Get point count
            point_count = self.client.count_points(collection_name)
            
            return {
                "exists": True,
                "name": collection_name,
                "points_count": point_count,
                "vectors_count": info.get("vectors_count", 0),
                "status": info.get("status"),
                "config": info.get("config")
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"exists": False, "error": str(e)}
    
    def list_collections(self) -> List[str]:
        """List all available collections.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: Optional[int] = None,
        overwrite: bool = False
    ) -> bool:
        """Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (auto-detects if None)
            overwrite: Whether to overwrite existing collection
            
        Returns:
            True if successful
        """
        try:
            # Auto-detect vector size if not provided
            if vector_size is None:
                vector_size = self._get_embedding_dimension()
            
            result = self.client.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                overwrite=overwrite
            )
            
            if result:
                logger.info(f"Created collection '{collection_name}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise
    
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful
        """
        try:
            result = self.client.delete_collection(collection_name)
            
            if result:
                # Remove from cache
                if collection_name in self._vector_stores:
                    del self._vector_stores[collection_name]
                logger.info(f"Deleted collection '{collection_name}'")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check vector store health.
        
        Returns:
            Health status dictionary
        """
        try:
            is_healthy = self.client.health_check()
            collections = self.list_collections()
            
            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "url": self.client.url,
                "collections_count": len(collections),
                "collections": collections
            }
            
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            return {
                "status": "unhealthy",
                "url": self.client.url,
                "error": str(e)
            }


# Example usage demonstrating integration of all components
"""
Complete Example Usage:

# 1. Initialize all components (typically handled by FastAPI dependency injection)
from core.dependencies import get_vector_store_service
from core.config import settings
from langchain_core.documents import Document

# Get service instance (uses VectorStoreClient + EmbeddingService + QdrantVectorStoreWrapper)
vector_service = get_vector_store_service()

# 2. Create collection and add documents
await vector_service.create_collection("documents", overwrite=True)

documents = [
    Document(page_content="Machine learning enables computers to learn without explicit programming"),
    Document(page_content="Vector databases store high-dimensional vectors for similarity search"),
    Document(page_content="RAG combines retrieval and generation for better AI responses")
]

doc_ids = await vector_service.add_documents(documents, collection_name="documents")

# 3. Semantic search (without scores) - using unified method
simple_results = await vector_service.similarity_search(
    query="What is machine learning?",
    k=2,
    collection_name="documents",
    with_scores=False  # Default behavior
)

# 4. Alternative: using convenience method
convenience_results = await vector_service.similarity_search_with_score(
    query="What is machine learning?",
    k=3,
    collection_name="documents",
    score_threshold=0.7  # Only return results with similarity >= 0.7
)

# 5. Unified method with scores and threshold filtering
scored_results = await vector_service.similarity_search(
    query="What is machine learning?",
    k=3,
    collection_name="documents",
    score_threshold=0.7,
    with_scores=True  # Get scores with documents
)

# 6. Add texts directly (alternative to documents)
text_ids = await vector_service.add_texts(
    texts=["Python is a programming language", "JavaScript is used for web development"],
    metadatas=[{"type": "programming"}, {"type": "web"}],
    collection_name="documents"
)

# 7. Search with metadata filtering (simple dict format - automatically converted to Qdrant Filter)
filtered_results = await vector_service.similarity_search(
    query="programming languages",
    k=2,
    collection_name="documents",
    filter_conditions={"type": "programming"},  # Simple dict automatically converted to Qdrant Filter
    with_scores=True
)

# 8. Get retriever for LangChain integration
retriever = await vector_service.get_retriever(
    collection_name="documents",
    search_kwargs={"k": 3, "score_threshold": 0.5}
)

# 9. Health monitoring and collection management
health = vector_service.health_check()
stats = vector_service.get_collection_stats("documents")
all_collections = vector_service.list_collections()

# 10. Document cleanup
await vector_service.delete_documents(doc_ids[:2], collection_name="documents")

print(f"Added {len(doc_ids)} documents + {len(text_ids)} texts")
print(f"Simple search found {len(simple_results)} documents")
print(f"Convenience method found {len(convenience_results)} documents with scores")
print(f"Unified method found {len(scored_results)} documents with scores")
print(f"Filtered search found {len(filtered_results)} programming docs")
print(f"Health: {health['status']}")
print(f"Collection has {stats['points_count']} points")
print(f"Available collections: {all_collections}")
"""