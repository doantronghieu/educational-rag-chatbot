"""
LangChain embedding utilities and wrappers.
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for managing embedding operations."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize embedding service.
        
        Args:
            api_key: OpenAI API key, defaults to settings
            model: Embedding model name
        """
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.embedding_model
        self._embeddings = None
    
    @property
    def embeddings(self) -> Embeddings:
        """Get embeddings instance."""
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                openai_api_key=self.api_key,
                model=self.model
            )
        return self._embeddings
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = await self.embeddings.aembed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} documents")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to embed documents: {e}")
            raise
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = await self.embeddings.aembed_query(text)
            logger.info("Generated embedding for query")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
    
    def get_vector_dimension(self) -> int:
        """Get the dimension of embedding vectors.
        
        Returns:
            Vector dimension
        """
        # Default dimensions for common OpenAI models
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        
        return model_dimensions.get(self.model, 1536)


