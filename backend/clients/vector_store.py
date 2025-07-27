"""Vector database client."""

import logging
from typing import Optional, List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter

logger = logging.getLogger(__name__)


class VectorStoreClient:
    """Qdrant vector store client for vector operations."""

    def __init__(self, url: str, api_key: Optional[str] = None, timeout: int = 60):
        """Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> QdrantClient:
        """Initialize and return Qdrant client."""
        try:
            if self.url.startswith("http"):
                # Remote Qdrant instance
                client = QdrantClient(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            else:
                # Local file-based instance
                client = QdrantClient(path=self.url)
            
            # Test connection
            client.get_collections()
            logger.info(f"Successfully connected to Qdrant at {self.url}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant at {self.url}: {e}")
            raise ConnectionError(f"Could not connect to Qdrant: {e}")
    
    def health_check(self) -> bool:
        """Check if Qdrant service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        overwrite: bool = False
    ) -> bool:
        """Create a new collection.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric
            overwrite: Whether to overwrite existing collection
            
        Returns:
            True if successful
        """
        try:
            if overwrite:
                self.delete_collection(collection_name)
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists
        """
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection information.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection info dict or None if not found
        """
        try:
            info = self.client.get_collection(collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": info.config.dict() if info.config else None
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {e}")
            return None
    
    def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct]
    ) -> bool:
        """Upsert points into collection.
        
        Args:
            collection_name: Name of the collection
            points: List of points to upsert
            
        Returns:
            True if successful
        """
        try:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )
            logger.info(f"Upserted {len(points)} points to '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert points to '{collection_name}': {e}")
            raise
    
    def search_points(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Filter] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar points.
        
        Args:
            collection_name: Name of the collection
            query_vector: Query vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            filter_conditions: Optional filter conditions
            
        Returns:
            List of search results
        """
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions
            )
            
            return [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "vector": result.vector
                }
                for result in results
            ]
        except Exception as e:
            logger.error(f"Failed to search in '{collection_name}': {e}")
            return []
    
    def delete_points(
        self,
        collection_name: str,
        point_ids: List[str]
    ) -> bool:
        """Delete points by IDs.
        
        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to delete
            
        Returns:
            True if successful
        """
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )
            logger.info(f"Deleted {len(point_ids)} points from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete points from '{collection_name}': {e}")
            raise
    
    def count_points(self, collection_name: str) -> int:
        """Count points in collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of points, -1 if error
        """
        try:
            result = self.client.count(collection_name=collection_name)
            return result.count
        except Exception as e:
            logger.error(f"Failed to count points in '{collection_name}': {e}")
            return -1
    
    def close(self) -> None:
        """Close the client connection."""
        try:
            if hasattr(self.client, 'close'):
                self.client.close()
            logger.info("Qdrant client connection closed")
        except Exception as e:
            logger.error(f"Error closing Qdrant client: {e}")
