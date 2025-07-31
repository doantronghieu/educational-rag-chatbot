"""
LangChain retriever implementations and utilities.

Retriever class that handles retrieval functionality 
"""

from typing import Any, Dict, List, Optional, Union, Callable, Type, Literal
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.stores import BaseStore

from langchain_core.callbacks import CallbackManagerForRetrieverRun

# LangChain retriever imports
from langchain.retrievers.multi_query import MultiQueryRetriever as LangChainMultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever as LangChainContextualCompressionRetriever
from langchain.retrievers.ensemble import EnsembleRetriever as LangChainEnsembleRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever as LangChainSelfQueryRetriever
from langchain.retrievers.multi_vector import MultiVectorRetriever as LangChainMultiVectorRetriever
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever as LangChainParentDocumentRetriever
from langchain_community.document_transformers import LongContextReorder
from langchain.retrievers.document_compressors import LLMChainExtractor, EmbeddingsFilter, DocumentCompressorPipeline
from pydantic import BaseModel, Field


# Type definitions
RetrieverType = Literal[
    "vector", "similarity", "mmr", "score", "hybrid", "ensemble", 
    "multi_query", "reorder", "long_context", "compression", 
    "contextual_compression", "self_query", "multi_vector", "parent_document"
]

CompressorType = Literal["llm", "embeddings"]

class Retriever(BaseRetriever):
    """
    Single consolidated retriever that handles all retrieval functionality.
    """
    
    # Define fields for Pydantic compatibility
    retriever_type: RetrieverType = Field(..., description="Type of retrieval to perform")
    vectorstore: Optional[VectorStore] = Field(None, description="Vector store for vector-based retrieval")  
    search_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Search parameters for vector operations")
    base_retriever: Optional[BaseRetriever] = Field(None, description="Base retriever for composite retrievers")
    llm: Optional[BaseChatModel] = Field(None, description="Language model for LLM-based retrievers")
    embeddings: Optional[Embeddings] = Field(None, description="Embeddings model for embedding-based operations")
    docstore: Optional[BaseStore] = Field(None, description="Document store for multi-vector/parent retrievers")
    compressor_type: CompressorType = Field("llm", description="Type of compressor for compression retrievers")
    retrievers: Optional[List[BaseRetriever]] = Field(None, description="List of retrievers for ensemble")
    weights: Optional[List[float]] = Field(None, description="Weights for ensemble retrievers")
    c: int = Field(60, description="Reciprocal rank fusion parameter")
    document_content_description: Optional[str] = Field(None, description="Description of document content for self-query")
    metadata_field_info: Optional[List[Any]] = Field(None, description="Metadata field information for self-query")
    child_splitter: Optional[Any] = Field(None, description="Text splitter for child documents")
    id_key: str = Field("doc_id", description="Key for document IDs")
    similarity_threshold: float = Field(0.76, description="Similarity threshold for filtering")
    k: Optional[int] = Field(None, description="Number of documents to retrieve")
    get_input: Optional[Callable] = Field(None, description="Input function for LLM compressor")

    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents based on retriever type."""
        
        # Vector store based retrievers
        if self.retriever_type in ["vector", "similarity"]:
            return self._similarity_search(query)
        elif self.retriever_type == "mmr":
            return self._mmr_search(query)
        elif self.retriever_type == "score":
            return self._score_search(query)
        elif self.retriever_type == "hybrid":
            return self._hybrid_search(query)
            
        # Composite retrievers
        elif self.retriever_type == "ensemble":
            return self._ensemble_retrieve(query, run_manager)
        elif self.retriever_type == "multi_query":
            return self._multi_query_retrieve(query, run_manager)
            
        # Transform retrievers
        elif self.retriever_type in ["reorder", "long_context"]:
            return self._reorder_retrieve(query, run_manager)
            
        # Compression retrievers
        elif self.retriever_type in ["compression", "contextual_compression"]:
            return self._compression_retrieve(query, run_manager)
            
        # Self-query and multi-vector (delegated to LangChain)
        elif self.retriever_type == "self_query":
            return self._self_query_retrieve(query, run_manager)
        elif self.retriever_type == "multi_vector":
            return self._multi_vector_retrieve(query, run_manager)
        elif self.retriever_type == "parent_document":
            return self._parent_document_retrieve(query, run_manager)
        else:
            raise ValueError(f"Unsupported retriever type: {self.retriever_type}")
    
    def _validate_vectorstore(self, operation_name: str) -> None:
        """Validate that vectorstore is available for the operation."""
        if not self.vectorstore:
            raise ValueError(f"Vector store required for {operation_name}")
    
    def _validate_required_fields(self, fields: Dict[str, Any], operation: str) -> None:
        """Validate that required fields are present for the operation."""
        missing_fields = [name for name, value in fields.items() if value is None]
        if missing_fields:
            field_names = ", ".join(missing_fields)
            raise ValueError(f"{operation} requires {field_names}")
    
    def _create_langchain_retriever(self, retriever_class, **kwargs):
        """Generic factory for LangChain retrievers with consistent error handling."""
        try:
            return retriever_class(**kwargs)
        except Exception as e:
            raise ValueError(f"Failed to create {retriever_class.__name__}: {str(e)}")
    
    # Vector store methods
    def _similarity_search(self, query: str) -> List[Document]:
        self._validate_vectorstore("similarity search")
        return self.vectorstore.similarity_search(query, **self.search_kwargs)
    
    def _mmr_search(self, query: str) -> List[Document]:
        self._validate_vectorstore("MMR search")
        return self.vectorstore.max_marginal_relevance_search(query, **self.search_kwargs)
    
    def _score_search(self, query: str) -> List[Document]:
        self._validate_vectorstore("score search")
        docs_and_scores = self.vectorstore.similarity_search_with_score(query, **self.search_kwargs)
        score_key = self.search_kwargs.get("score_key", "similarity_score")
        
        scored_docs = []
        for doc, score in docs_and_scores:
            new_metadata = doc.metadata.copy()
            new_metadata[score_key] = score
            scored_docs.append(Document(page_content=doc.page_content, metadata=new_metadata))
        
        return scored_docs
    
    def _hybrid_search(self, query: str) -> List[Document]:
        self._validate_vectorstore("hybrid search")
        
        # Check if vectorstore supports true hybrid search
        if hasattr(self.vectorstore, '__class__') and 'Qdrant' in self.vectorstore.__class__.__name__:
            try:
                # Prepare search kwargs for Qdrant
                search_kwargs = self.search_kwargs.copy()
                hybrid_params = search_kwargs.pop('hybrid_params', {})
                
                # Check if vectorstore has sparse embedding for true hybrid
                if hasattr(self.vectorstore, 'sparse_embedding') and self.vectorstore.sparse_embedding is not None:
                    # QdrantVectorStore with sparse embeddings - true hybrid search
                    print("Using QdrantVectorStore with sparse embeddings for true hybrid search")
                    
                    # For true hybrid search, QdrantVectorStore should handle it automatically
                    return self.vectorstore.similarity_search(query, **search_kwargs)
                    
                elif hybrid_params and 'alpha' in hybrid_params:
                    # QdrantVectorStore without sparse embeddings, but with fusion parameters
                    # Try to create a basic fusion query if supported
                    try:
                        from qdrant_client.http import models
                        
                        # Create a simple fusion query for hybrid search
                        # This is a basic implementation - real hybrid would need sparse vectors
                        fusion_query = models.FusionQuery(
                            fusion=models.Fusion.RRF  # Reciprocal Rank Fusion
                        )
                        
                        print("Attempting QdrantVectorStore fusion search...")
                        return self.vectorstore.similarity_search(
                            query, 
                            hybrid_fusion=fusion_query,
                            **search_kwargs
                        )
                    except Exception as fusion_error:
                        print(f"Fusion search not available: {fusion_error}")
                        # Fall through to regular similarity search
                
                # Regular QdrantVectorStore similarity search
                print("Using QdrantVectorStore similarity search (hybrid fusion not configured)")
                return self.vectorstore.similarity_search(query, **search_kwargs)
                    
            except Exception as e:
                # If hybrid search fails, fall back to similarity search
                print(f"Hybrid search failed, falling back to similarity search: {e}")
                return self._similarity_search(query)
        else:
            # For non-Qdrant vectorstores, fall back to similarity search
            print(f"Vectorstore {self.vectorstore.__class__.__name__} does not support hybrid search, using similarity search")
            return self._similarity_search(query)
    
    # Composite methods
    def _ensemble_retrieve(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self._validate_required_fields({"retrievers": self.retrievers}, "Ensemble retriever")
        
        ensemble = self._create_langchain_retriever(
            LangChainEnsembleRetriever,
            retrievers=self.retrievers,
            weights=self.weights or [1.0 / len(self.retrievers)] * len(self.retrievers),
            c=self.c
        )
        
        return ensemble.invoke(query, config={"run_manager": run_manager})
    
    def _multi_query_retrieve(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self._validate_required_fields({
            "base_retriever": self.base_retriever,
            "llm": self.llm
        }, "Multi-query retriever")
        
        multi_query = LangChainMultiQueryRetriever.from_llm(
            retriever=self.base_retriever,
            llm=self.llm
        )
        
        return multi_query.invoke(query, config={"run_manager": run_manager})
    
    # Transform methods
    def _reorder_retrieve(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self._validate_required_fields({"base_retriever": self.base_retriever}, "Reorder retriever")
        
        docs = self.base_retriever.invoke(query, config={"run_manager": run_manager})
        
        if LongContextReorder is not None:
            reorderer = LongContextReorder()
            return reorderer.transform_documents(docs)
        else:
            if len(docs) <= 2:
                return docs
            
            reordered = [docs[0]]
            if len(docs) > 2:
                reordered.extend(docs[2:-1])
            if len(docs) > 1:
                reordered.append(docs[1])
            
            return reordered
    
    # Compression methods
    def _compression_retrieve(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self._validate_required_fields({"base_retriever": self.base_retriever}, "Compression retriever")
        
        if self.compressor_type == "llm":
            compressor = self._create_llm_compressor()
        elif self.compressor_type == "embeddings":
            compressor = self._create_embeddings_compressor()
        else:
            raise ValueError(f"Unsupported compressor type: {self.compressor_type}")
        
        compression_retriever = self._create_langchain_retriever(
            LangChainContextualCompressionRetriever,
            base_compressor=compressor,
            base_retriever=self.base_retriever
        )
        
        return compression_retriever.invoke(query, config={"run_manager": run_manager})
    
    def _create_llm_compressor(self):
        self._validate_required_fields({"llm": self.llm}, "LLM compressor")
        
        return LLMChainExtractor.from_llm(self.llm, get_input=self.get_input)
    
    def _create_embeddings_compressor(self):
        self._validate_required_fields({"embeddings": self.embeddings}, "Embeddings compressor")
        
        return EmbeddingsFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold,
            k=self.k
        )
    
    # Delegated methods for complex retrievers
    def _self_query_retrieve(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self._validate_required_fields({
            "vectorstore": self.vectorstore,
            "document_content_description": self.document_content_description,
            "metadata_field_info": self.metadata_field_info,
            "llm": self.llm
        }, "Self-query retriever")
        
        try:
            self_query = LangChainSelfQueryRetriever.from_llm(
                llm=self.llm,
                vectorstore=self.vectorstore,
                document_contents=self.document_content_description,
                metadata_field_info=self.metadata_field_info
            )
            
            return self_query.invoke(query, config={"run_manager": run_manager})
        except ValueError as e:
            if "not supported" in str(e):
                # Fallback to regular similarity search with warning
                supported_stores = ["Chroma", "Pinecone", "Weaviate", "Qdrant", "FAISS", "ElasticSearch"]
                raise ValueError(
                    f"Self-query retriever not supported for {self.vectorstore.__class__.__name__}. "
                    f"Supported vectorstores: {', '.join(supported_stores)}. "
                    f"Original error: {str(e)}"
                )
            else:
                raise e
    
    def _multi_vector_retrieve(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self._validate_required_fields({
            "vectorstore": self.vectorstore,
            "docstore": self.docstore
        }, "Multi-vector retriever")
        
        multi_vector = self._create_langchain_retriever(
            LangChainMultiVectorRetriever,
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key=self.id_key
        )
        
        return multi_vector.invoke(query, config={"run_manager": run_manager})
    
    def _parent_document_retrieve(self, query: str, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        self._validate_required_fields({
            "vectorstore": self.vectorstore,
            "docstore": self.docstore,
            "child_splitter": self.child_splitter
        }, "Parent document retriever")
        
        parent_doc = self._create_langchain_retriever(
            LangChainParentDocumentRetriever,
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter
        )
        
        return parent_doc.invoke(query, config={"run_manager": run_manager})
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the retriever (for retrievers that support it)."""
        if self.retriever_type == "parent_document":
            self._validate_required_fields({
                "vectorstore": self.vectorstore,
                "docstore": self.docstore,
                "child_splitter": self.child_splitter
            }, "Parent document retriever")
            
            parent_doc = self._create_langchain_retriever(
                LangChainParentDocumentRetriever,
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter
            )
            parent_doc.add_documents(documents)
        
        elif self.retriever_type == "multi_vector":
            self._validate_required_fields({
                "vectorstore": self.vectorstore,
                "docstore": self.docstore
            }, "Multi-vector retriever")
            
            multi_vector = self._create_langchain_retriever(
                LangChainMultiVectorRetriever,
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                id_key=self.id_key
            )
            multi_vector.add_documents(documents)
        
        else:
            raise ValueError(f"add_documents not supported for retriever type: {self.retriever_type}")
    
    # Factory methods
    
    @classmethod
    def create_vector_retriever(
        cls,
        vectorstore: VectorStore,
        search_type: RetrieverType = "similarity",
        **search_kwargs
    ) -> "Retriever":
        """Create a vector store retriever."""
        return cls(retriever_type=search_type, vectorstore=vectorstore, search_kwargs=search_kwargs)
    
    @classmethod
    def create_score_retriever(
        cls,
        vectorstore: VectorStore,
        score_key: str = "similarity_score",
        **search_kwargs
    ) -> "Retriever":
        """Create a score retriever that adds similarity scores to metadata."""
        search_kwargs["score_key"] = score_key
        return cls(retriever_type="score", vectorstore=vectorstore, search_kwargs=search_kwargs)
    
    @classmethod
    def create_hybrid_retriever(
        cls,
        vectorstore: VectorStore,
        hybrid_params: Optional[Dict[str, Any]] = None,
        **search_kwargs
    ) -> "Retriever":
        """Create a hybrid retriever combining vector and other search methods."""
        search_kwargs["hybrid_params"] = hybrid_params or {}
        return cls(retriever_type="hybrid", vectorstore=vectorstore, search_kwargs=search_kwargs)
    
    @classmethod
    def create_ensemble_retriever(
        cls,
        retrievers: List[BaseRetriever],
        weights: Optional[List[float]] = None,
        c: int = 60
    ) -> "Retriever":
        """Create an ensemble retriever using Reciprocal Rank Fusion."""
        return cls(retriever_type="ensemble", retrievers=retrievers, weights=weights, c=c)
    
    @classmethod
    def create_multi_query_retriever(
        cls,
        base_retriever: BaseRetriever,
        llm: BaseChatModel
    ) -> "Retriever":
        """Create a multi-query retriever using LLM-generated query perspectives."""
        return cls(retriever_type="multi_query", base_retriever=base_retriever, llm=llm)
    
    @classmethod
    def create_reorder_retriever(
        cls,
        base_retriever: BaseRetriever
    ) -> "Retriever":
        """Create a long context reorder retriever."""
        return cls(retriever_type="reorder", base_retriever=base_retriever)
    
    @classmethod
    def create_compression_retriever(
        cls,
        base_retriever: BaseRetriever,
        compressor_type: CompressorType = "llm",
        llm: Optional[BaseChatModel] = None,
        embeddings: Optional[Embeddings] = None,
        similarity_threshold: float = 0.76,
        k: Optional[int] = None,
        get_input: Optional[Callable] = None
    ) -> "Retriever":
        """Create a contextual compression retriever."""
        return cls(
            retriever_type="compression",
            base_retriever=base_retriever,
            compressor_type=compressor_type,
            llm=llm,
            embeddings=embeddings,
            similarity_threshold=similarity_threshold,
            k=k,
            get_input=get_input
        )
    
    @classmethod
    def create_self_query_retriever(
        cls,
        vectorstore: VectorStore,
        document_content_description: str,
        metadata_field_info: List[Any],
        llm: BaseChatModel
    ) -> "Retriever":
        """Create a self-query retriever."""
        return cls(
            retriever_type="self_query",
            vectorstore=vectorstore,
            document_content_description=document_content_description,
            metadata_field_info=metadata_field_info,
            llm=llm
        )
    
    @classmethod
    def create_multi_vector_retriever(
        cls,
        vectorstore: VectorStore,
        docstore: BaseStore,
        id_key: str = "doc_id"
    ) -> "Retriever":
        """Create a multi-vector retriever."""
        return cls(
            retriever_type="multi_vector",
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=id_key
        )
    
    @classmethod
    def create_parent_document_retriever(
        cls,
        vectorstore: VectorStore,
        docstore: BaseStore,
        child_splitter: Any
    ) -> "Retriever":
        """Create a parent document retriever."""
        return cls(
            retriever_type="parent_document",
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=child_splitter
        )


class RetrieverManager:
    """Manager for retrievers"""
    
    def __init__(self):
        self._retrievers: Dict[str, Retriever] = {}
    
    def register_retriever(self, name: str, retriever: Union[BaseRetriever, Retriever]) -> None:
        """Register a retriever with a name."""
        if isinstance(retriever, Retriever):
            self._retrievers[name] = retriever
        else:
            self._retrievers[name] = retriever
    
    def create_retriever(self, name: str, retriever_type: RetrieverType, **kwargs) -> Retriever:
        """Create and register a retriever based on type."""
        retriever = Retriever(retriever_type=retriever_type, **kwargs)
        self.register_retriever(name, retriever)
        return retriever
    
    async def retrieve(self, retriever_name: str, query: str, **kwargs) -> List[Document]:
        """Retrieve documents using named retriever (async by default)."""
        retriever = self._retrievers.get(retriever_name)
        if not retriever:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        
        docs = await retriever.ainvoke(query, config=kwargs)
        
        return docs
