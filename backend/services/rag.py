"""RAG (Retrieval Augmented Generation) service for PDF document processing."""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import re
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from libs.langchain.loaders import PDFLoader
from libs.langchain.text_splitters import TextSplitterManager
from utils.decorators import handle_errors
from core.dependencies import get_llm, get_vector_store_service


class StructuredQuery(BaseModel):
    """Structured query with search terms and metadata filters."""
    query: str = Field(description="The main search query text")
    page_filter: Optional[int] = Field(
        default=None, 
        description="Page number filter if mentioned in query"
    )
    topic: Optional[str] = Field(
        default=None,
        description="Main topic extracted/inferred from the query"
    )
    key_words: Optional[List[str]] = Field(
        default=None,
        description="Keywords extracted/inferred from the query"
    )


class ChunkAnalysis(BaseModel):
    """Analysis result for a document chunk."""
    topic: str = Field(description="Main topic of the chunk")
    key_words: List[str] = Field(description="Key words extracted from the chunk")


class RAGService:
    """RAG service for PDF document indexing and question answering.
    
    WORKFLOW: Index PDFs → Analyze queries → Retrieve with filters → Generate answers
    
    QUERY PROCESSING FLOW:
    User: "Find planning info from page 0" 
    → LLM extracts: StructuredQuery(query="planning", page_filter=0)
    → Database filters: {"page": 0} 
    → Vector search: similarity + metadata filtering
    → Answer: Context-aware response with source tracking
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize RAG service.
        
        Args:
            chunk_size: Maximum size of text chunks
            chunk_overlap: Overlap between text chunks
        """
        self.vector_store_service = get_vector_store_service()
        self.llm = get_llm()
        self.text_splitter_manager = TextSplitterManager(
            default_chunk_size=chunk_size,
            default_chunk_overlap=chunk_overlap
        )
        
        # Default RAG prompt template - improved to be less restrictive
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context. "
                      "Carefully read the context and extract relevant information to answer the question. "
                      "Focus on the main concepts and components mentioned in the context. "
                      "If you can find relevant information in the context, provide a clear and comprehensive answer. "
                      "Only say you don't have enough information if the context is completely irrelevant to the question."),
            ("human", "Context: {context}\n\nQuestion: {question}")
        ])
        
        # Query analysis prompt - extracts structured filters and topics from natural language
        self.query_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert at analyzing user queries for document search. "
                      "Given a user query, extract the main search terms and any specific filters. "
                      "Look for mentions of specific page numbers (e.g., 'page 0', 'first page'). "
                      "Also extract the main topic and key words from the query. "
                      "Only set filters if explicitly mentioned in the query."),
            ("human", "Analyze this query and extract search terms and any filters: {query}")
        ])
        
        # Chunk analysis prompt - extracts topic and keywords for metadata enrichment
        self.chunk_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze this document chunk and extract the main topic and key words. "
                      "Topic should be a short phrase describing the main subject. "
                      "Key words should be a list of important terms, concepts, and entities. "
                      "Be concise and focused - no comments or explanations."),
            ("human", "Document chunk: {content}")
        ])
        
    
    def _get_query_to_database_filter_mapping(self) -> Dict[str, str]:
        """Step 1: Configure how query filters map to database fields.
        
        Part of dynamic mapping system - defines translation rules.
        Returns: {'page_filter': 'page', 'author_filter': 'author', ...}
        """
        mappings = {}
        for field_name, field_info in StructuredQuery.model_fields.items():
            if field_name.endswith('_filter'):
                # Map page_filter -> page, author_filter -> author, etc.
                database_field = field_name.replace('_filter', '')
                mappings[field_name] = database_field
        return mappings
    
    def _get_analysis_to_metadata_field_mapping(self) -> Dict[str, str]:
        """Step 2: Configure how analysis results map to metadata keys.
        
        Part of dynamic mapping system - avoids conflicts between document/query metadata.
        Returns: {'query': 'analyzed_query', 'topic': 'query_topic', ...}
        """
        mappings = {}
        for field_name in StructuredQuery.model_fields.keys():
            if not field_name.endswith('_filter'):
                # Map query analysis fields to descriptive metadata keys
                if field_name == 'query':
                    mappings[field_name] = 'analyzed_query'  # The processed/rewritten query
                elif field_name == 'topic':
                    mappings[field_name] = 'query_topic'  # Topic extracted from user query
                elif field_name == 'key_words':
                    mappings[field_name] = 'query_keywords'  # Keywords extracted from user query
                else:
                    # Future fields get analysis_ prefix to avoid naming conflicts
                    mappings[field_name] = f'analysis_{field_name}'
        return mappings
    
    async def _ensure_collection_exists(self, collection_name: str) -> None:
        """Validate collection exists, raise ValueError if not."""
        if not await self.vector_store_service.client.collection_exists(collection_name):
            raise ValueError(f"Collection '{collection_name}' not found")
    
    def _extract_database_filters(self, structured_query: StructuredQuery) -> Dict[str, Any]:
        """Step 3: Extract actual filter values for vector database search.
        
        Uses mapping from step 1 to convert StructuredQuery filters to database format.
        Flow: StructuredQuery(page_filter=0) → {"page": 0} → Vector search filters
        
        Args:
            structured_query: Analyzed query from LLM
        Returns:
            Database filter dict, e.g., {'page': 0}
        """
        filters = {}
        # Apply step 1 mapping: query filter fields → database fields
        filter_mappings = self._get_query_to_database_filter_mapping()
        for query_field, database_field in filter_mappings.items():
            value = getattr(structured_query, query_field, None)
            if value is not None:
                filters[database_field] = value
        return filters
    
    def _create_query_analysis_record(self, analysis_result: StructuredQuery) -> Dict[str, Any]:
        """Step 4: Create comprehensive analysis record for user transparency.
        
        Combines extracted filters (step 3) + renamed analysis fields (step 2) 
        into complete metadata record for response tracking.
        
        Args:
            analysis_result: Structured query from LLM
        Returns:
            Complete analysis metadata: {filters_applied, analyzed_query, query_topic, ...}
        """
        # Combine step 3 (filters) + step 2 (field mappings) into complete record
        filters_applied = self._extract_database_filters(analysis_result)
        analysis_record = {"filters_applied": filters_applied if filters_applied else None}
        # Apply step 2 mapping: analysis fields → metadata keys
        metadata_mappings = self._get_analysis_to_metadata_field_mapping()
        for analysis_field, metadata_key in metadata_mappings.items():
            value = getattr(analysis_result, analysis_field, None)
            analysis_record[metadata_key] = value
            
        return analysis_record
    
    @handle_errors("Analyze query structure")
    async def analyze_query(self, query: str) -> StructuredQuery:
        """
        Core: Convert natural language query to structured search parameters.
        
        Uses LLM with structured output to extract filters, topics, and keywords.
        This feeds into the 4-step mapping system for database search and metadata.
        
        Args:
            query: Natural language query
        Returns:
            StructuredQuery with parsed filters, topics, keywords
        """
        # Create structured LLM for query analysis
        structured_llm = self.llm.model.with_structured_output(StructuredQuery)
        
        # Format the query analysis prompt
        messages = self.query_analysis_prompt.format_messages(query=query)
        
        # Get structured analysis
        structured_query = await structured_llm.ainvoke(messages)
        
        return structured_query
    
    def _get_collection_name(self, file_path: Path) -> str:
        """Generate collection name from file path.
        
        Collection names can only contain alphanumeric characters and underscores.
        """
        # Get filename without extension
        name = file_path.stem.lower()
        # Replace all non-alphanumeric characters with underscores
        name = re.sub(r'[^a-z0-9]', '_', name)
        # Remove multiple consecutive underscores
        name = re.sub(r'_{2,}', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure it starts with letter or underscore (not number)
        if name and name[0].isdigit():
            name = f"file_{name}"
        # Add pdf suffix
        return f"{name}_pdf"
    
    async def _add_metadata(self, documents: List[Document]) -> List[Document]:
        """Add metadata to documents including topic and keywords using batch LLM calls."""
        # Add chunk_index and clean up unnecessary PDF metadata
        for i, doc in enumerate(documents):
            doc.metadata.update({
                "chunk_index": i
            })
            
            # Remove unnecessary PDF metadata properties
            unwanted_keys = [
                'producer', 'creator', 'creationdate', 'file_path', 'format', 
                'title', 'subject', 'moddate', 'trapped', 'modDate', 
                'creationDate', 'file_name'
            ]
            for key in unwanted_keys:
                doc.metadata.pop(key, None)
        
        if not documents:
            return documents
        
        # Prepare batch inputs for topic and keywords extraction
        content_list = [doc.page_content for doc in documents]
        
        # Create structured LLM for chunk analysis
        structured_llm = self.llm.model.with_structured_output(ChunkAnalysis)
        
        # Prepare batch prompts for analysis
        batch_prompts = []
        for content in content_list:
            batch_prompts.append(self.chunk_analysis_prompt.format_messages(content=content))
        
        try:
            # Perform batch analysis
            analysis_results = await structured_llm.abatch(batch_prompts)
            
            # Add analysis results to document metadata
            for i, (doc, analysis) in enumerate(zip(documents, analysis_results)):
                analysis: ChunkAnalysis
                doc.metadata.update({
                    "topic": analysis.topic,
                    "key_words": analysis.key_words
                })
                
        except Exception as e:
            # If batch analysis fails, continue without topic/keywords
            print(f"Warning: Failed to extract topic/keywords: {e}")
            
        return documents
    
    @handle_errors("Index PDF document") 
    async def index_pdf(
        self, 
        file_path: str | Path, 
        collection_name: Optional[str] = None
    ) -> str:
        """
        Indexing: Load PDF → Split into chunks → Extract metadata → Store in vector DB.
        
        Enriches chunks with topic/keywords via batch LLM processing for better retrieval.
        
        Args:
            file_path: Path to PDF file
            collection_name: Collection name (auto-generated if None)
        Returns:
            Collection name used for indexing
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Generate collection name if not provided
        if collection_name is None:
            collection_name = self._get_collection_name(file_path)
        
        # Load PDF documents
        loader = PDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError(f"No content extracted from PDF: {file_path}")
        
        # Split documents into chunks using default settings
        splitter = self.text_splitter_manager.get_recursive_character_splitter()
        split_documents = splitter.split_documents(documents)
        
        # Add metadata including topic and keywords
        split_documents = await self._add_metadata(split_documents)
        
        # Store in vector database
        await self.vector_store_service.add_documents(
            documents=split_documents,
            collection_name=collection_name
        )
        
        return collection_name
    
    @handle_errors("Retrieve documents")
    async def retrieve(
        self, 
        query: str, 
        collection_name: str,
        k: int = 4,
        filter_conditions: Optional[Dict[str, Any]] = None,
        enable_analysis: bool = False
    ) -> List[Document]:
        """
        Core: Retrieve relevant documents with hybrid search (vector + metadata).
        
        Two modes: Simple (direct search) or Analysis (LLM extracts filters first).
        Analysis mode: query → analyze_query() → extract_database_filters() → search
        
        Args:
            query: Search query (natural language or processed)
            collection_name: Target collection
            k: Number of documents to retrieve
            filter_conditions: Manual filters (merged with auto-extracted)
            enable_analysis: Auto-extract filters via LLM analysis
        Returns:
            List of relevant documents
        """
        # Check if collection exists
        await self._ensure_collection_exists(collection_name)
        
        # Determine final search parameters
        if enable_analysis:
            # Analyze the query structure
            structured_query = await self.analyze_query(query)
            
            # Use analyzed parameters
            search_query = structured_query.query
            search_k = k  # Always use provided k value
            
            # Build and merge filter conditions (provided filters take precedence)
            analyzed_filters = self._extract_database_filters(structured_query)
            final_filters = analyzed_filters.copy()
            if filter_conditions:
                final_filters.update(filter_conditions)
            final_filters = final_filters if final_filters else None
        else:
            # Use provided parameters directly
            search_query = query
            final_filters = filter_conditions
            search_k = k
        
        # Perform similarity search
        documents = await self.vector_store_service.similarity_search(
            query=search_query,
            collection_name=collection_name,
            k=search_k,
            filter_conditions=final_filters
        )
        
        return documents
    
    @handle_errors("Generate answer")
    async def generate_answer(
        self, 
        query: str, 
        context_docs: List[Document],
        custom_prompt: Optional[ChatPromptTemplate] = None
    ) -> str:
        """
        Generate answer using LLM based on query and context documents.
        
        Args:
            query: User question
            context_docs: Retrieved context documents
            custom_prompt: Custom prompt template (uses default if None)
            
        Returns:
            Generated answer
        """
        # Combine context documents
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Use custom prompt if provided, otherwise use default
        prompt = custom_prompt or self.prompt_template
        
        # Format the prompt
        messages = prompt.format_messages(context=context, question=query)
        
        # Generate answer using .run() method
        response = await self.llm.run(messages)
        
        return response.content
    
    @handle_errors("Complete RAG query")
    async def query(
        self, 
        query: str, 
        collection_name: str,
        k: int = 4,
        custom_prompt: Optional[ChatPromptTemplate] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        enable_analysis: bool = False
    ) -> Dict[str, Any]:
        """
        Main: Complete RAG pipeline from user question to final answer.
        
        Full workflow: 
        1. analyze_query() → StructuredQuery (if enable_analysis=True)
        2. retrieve() → relevant documents (uses extracted filters)
        3. generate_answer() → LLM response with context
        4. create_query_analysis_record() → metadata for transparency
        
        Args:
            query: User question
            collection_name: Target collection
            k: Documents for context
            custom_prompt: Override default answer prompt
            filter_conditions: Manual filters (merged with auto-extracted)
            enable_analysis: Enable intelligent query processing
        Returns:
            {answer, source_documents, metadata: {analysis: {...}}}
        """
        analysis_result = None
        
        # Capture analysis result if enabled
        if enable_analysis:
            analysis_result = await self.analyze_query(query)
        
        # Retrieve relevant documents (with or without analysis)
        context_docs = await self.retrieve(
            query=query,
            collection_name=collection_name,
            k=k,
            filter_conditions=filter_conditions,
            enable_analysis=enable_analysis
        )
        
        # Generate answer
        answer = await self.generate_answer(query, context_docs, custom_prompt)
        
        result = {
            "answer": answer,
            "source_documents": context_docs,
            "metadata": {
                "collection_name": collection_name,
                "num_sources": len(context_docs),
                "query": query
            }
        }
        
        # Add analysis results if available
        if analysis_result:
            result["metadata"]["analysis"] = self._create_query_analysis_record(analysis_result)
        
        return result
    
    @handle_errors("List collections")
    async def list_collections(self) -> List[str]:
        """List all available collections."""
        collections_info = await self.vector_store_service.client.client.get_collections()
        return [collection.name for collection in collections_info.collections]
    
    @handle_errors("Delete collection")
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of collection to delete
            
        Returns:
            True if successfully deleted
        """
        await self._ensure_collection_exists(collection_name)
        
        return await self.vector_store_service.client.delete_collection(collection_name)