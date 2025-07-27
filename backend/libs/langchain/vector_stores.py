"""
LangChain vector store implementations using Qdrant.

Example Usage:
```python
# Direct wrapper usage
from qdrant_client import QdrantClient
from libs.langchain.embeddings import get_default_embeddings
from libs.langchain.vector_stores import QdrantVectorStoreWrapper

client = QdrantClient("http://localhost:6333")
embeddings = get_default_embeddings()

# Create wrapper instance
wrapper = QdrantVectorStoreWrapper(
    client=client,
    collection_name="documents", 
    embedding=embeddings
)

# Add documents with metadata
docs = [
    Document(page_content="Python programming", metadata={"type": "programming"}),
    Document(page_content="Web development", metadata={"type": "web"})
]
ids = wrapper.add_documents(docs)

# Search without filters
results = wrapper.similarity_search("query", k=3)

# Search with simple dict filter (automatically converted to Qdrant Filter)
filtered_results = wrapper.similarity_search(
    "programming", 
    k=3, 
    filter={"type": "programming"}
)

# Search with advanced Qdrant Filter (for complex conditions)
from qdrant_client.models import Filter, FieldCondition, MatchValue
advanced_filter = Filter(
    must=[
        FieldCondition(key="type", match=MatchValue(value="programming"))
    ]
)
advanced_results = wrapper.similarity_search("query", k=3, filter=advanced_filter)

# Get retriever for chain integration
retriever = wrapper.get_retriever(search_kwargs={"k": 5})
```
"""

from typing import List, Optional, Dict, Any, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue


class QdrantVectorStoreWrapper:
    """Wrapper for Qdrant vector store with convenience methods."""
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        embedding: Embeddings
    ):
        """Initialize Qdrant vector store wrapper.
        
        Args:
            client: Qdrant client instance
            collection_name: Name of the collection
            embedding: Embedding model instance
        """
        self.client = client
        self.collection_name = collection_name
        self.embedding = embedding
        
        # Initialize the LangChain Qdrant vector store
        # Note: Collection creation should be handled by VectorStoreClient
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embedding
        )
    
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
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            ids: Optional list of document IDs
            **kwargs: Additional arguments
            
        Returns:
            List of document IDs
        """
        return self.vector_store.add_documents(documents, ids=ids, **kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Filter]] = None,
        **kwargs
    ) -> List[Document]:
        """Search for similar documents.
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional metadata filter (dict or Qdrant Filter)
            **kwargs: Additional arguments
            
        Returns:
            List of similar documents
        """
        # Convert dict filter to Qdrant Filter if needed
        if isinstance(filter, dict):
            filter = self._convert_filter(filter)
            
        return self.vector_store.similarity_search(
            query, k=k, filter=filter, **kwargs
        )
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Union[Dict[str, Any], Filter]] = None,
        **kwargs
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with similarity scores.
        
        Args:
            query: Query string
            k: Number of documents to return
            filter: Optional metadata filter (dict or Qdrant Filter)
            **kwargs: Additional arguments
            
        Returns:
            List of (document, score) tuples
        """
        # Convert dict filter to Qdrant Filter if needed
        if isinstance(filter, dict):
            filter = self._convert_filter(filter)
            
        return self.vector_store.similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )
    
    def delete(self, ids: List[str]) -> bool:
        """Delete documents by IDs.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        return self.vector_store.delete(ids)
    
    def get_retriever(self, **kwargs):
        """Get a retriever instance for this vector store.
        
        Args:
            **kwargs: Arguments for retriever configuration
            
        Returns:
            VectorStoreRetriever instance
        """
        return self.vector_store.as_retriever(**kwargs)
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        client: QdrantClient,
        collection_name: str,
        **kwargs
    ) -> "QdrantVectorStoreWrapper":
        """Create vector store from documents.
        
        Args:
            documents: List of documents
            embedding: Embedding model
            client: Qdrant client
            collection_name: Collection name
            **kwargs: Additional arguments
            
        Returns:
            QdrantVectorStoreWrapper instance
        """
        instance = cls(
            client=client,
            collection_name=collection_name,
            embedding=embedding
        )
        instance.add_documents(documents)
        return instance
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        client: QdrantClient,
        collection_name: str,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> "QdrantVectorStoreWrapper":
        """Create vector store from texts.
        
        Args:
            texts: List of text strings
            embedding: Embedding model
            client: Qdrant client
            collection_name: Collection name
            metadatas: Optional list of metadata dicts
            **kwargs: Additional arguments
            
        Returns:
            QdrantVectorStoreWrapper instance
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
            embedding=embedding,
            client=client,
            collection_name=collection_name
        )