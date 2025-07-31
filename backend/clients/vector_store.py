"""Vector database client."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
from typing import Optional, List, Dict, Any
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from utils.decorators import handle_errors

logger = logging.getLogger(__name__)


class VectorStoreClient:
    """Qdrant vector store client for vector operations."""

    def __init__(self, url: str, api_key: Optional[str] = None, timeout: int = 60):
        """Initialize Qdrant async and sync clients.
        
        Args:
            url: Qdrant server URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.url = url
        self.api_key = api_key
        self.timeout = timeout
        self.client = self._initialize_client(async_client=True)
        self.sync_client = self._initialize_client(async_client=False)
    
    def _initialize_client(self, async_client: bool = True):
        """Initialize and return Qdrant client (async or sync).
        
        Args:
            async_client: If True, returns AsyncQdrantClient, else QdrantClient
            
        Returns:
            AsyncQdrantClient or QdrantClient instance
        """
        client_type = "async" if async_client else "sync"
        client_class = AsyncQdrantClient if async_client else QdrantClient
        
        try:
            if self.url.startswith("http"):
                # Remote Qdrant instance
                client = client_class(
                    url=self.url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            else:
                # Local file-based instance
                client = client_class(path=self.url)
            
            logger.info(f"Successfully initialized {client_type} Qdrant client for {self.url}")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize {client_type} Qdrant client at {self.url}: {e}")
            raise ConnectionError(f"Could not connect to {client_type} Qdrant: {e}")
    
    @handle_errors("Health check", return_on_error=False, raise_on_error=False)
    async def health_check(self) -> bool:
        """Check if Qdrant service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        await self.client.get_collections()
        return True
    
    @handle_errors("Create collection")
    async def create_collection(
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
        if overwrite:
            await self.delete_collection(collection_name)
        
        await self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=distance
            )
        )
        logger.info(f"Created collection '{collection_name}' with vector size {vector_size}")
        return True
    
    @handle_errors("Delete collection")
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            True if successful
        """
        await self.client.delete_collection(collection_name)
        logger.info(f"Deleted collection '{collection_name}'")
        return True
    
    @handle_errors("Check collection existence", return_on_error=False, raise_on_error=False)
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if collection exists
        """
        collections = await self.client.get_collections()
        return any(col.name == collection_name for col in collections.collections)
    
    @handle_errors("Ensure collection exists")
    async def ensure_collection_exists(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        overwrite: bool = False
    ) -> bool:
        """Ensure collection exists, create if it doesn't.
        
        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance: Distance metric
            overwrite: Whether to overwrite existing collection
            
        Returns:
            True if collection exists or was created successfully
        """
        if overwrite or not await self.collection_exists(collection_name):
            await self.create_collection(
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance,
                overwrite=overwrite
            )
            logger.info(f"Ensured collection '{collection_name}' exists")
        return True
    
    @handle_errors("Get collection info", return_on_error=None, raise_on_error=False)
    async def get_collection_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get collection information.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Collection info dict or None if not found
        """
        info = await self.client.get_collection(collection_name)
        return {
            "name": collection_name,
            "vectors_count": info.vectors_count,
            "points_count": info.points_count,
            "status": info.status,
            "config": info.config.dict() if info.config else None
        }
    
    @handle_errors("Upsert points")
    async def upsert_points(
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
        await self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        logger.info(f"Upserted {len(points)} points to '{collection_name}'")
        return True
    
    @handle_errors("Search points", return_on_error=[], raise_on_error=False)
    async def search_points(
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
        results = await self.client.search(
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
    
    @handle_errors("Delete points")
    async def delete_points(
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
        await self.client.delete(
            collection_name=collection_name,
            points_selector=point_ids
        )
        logger.info(f"Deleted {len(point_ids)} points from '{collection_name}'")
        return True
    
    @handle_errors("Count points", return_on_error=-1, raise_on_error=False)
    async def count_points(self, collection_name: str) -> int:
        """Count points in collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of points, -1 if error
        """
        result = await self.client.count(collection_name=collection_name)
        return result.count
    
    @handle_errors("Close client connection", raise_on_error=False)
    async def close(self) -> None:
        """Close the async client connection."""
        await self.client.close()
        logger.info("Qdrant client connection closed")
