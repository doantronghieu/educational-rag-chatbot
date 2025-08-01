"""Integration test for RAG service with actual PDF processing."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import asyncio
import time
from typing import List, Dict, Any, Tuple, Set

from services.rag import RAGService
from langchain_core.documents import Document

# Common test variables
current_file: Path = Path(__file__).resolve()
backend_root: Path = current_file.parents[2]  # Go up from backend/tests/services/ to backend/
TEST_PDF: Path = backend_root / "tests" / "data" / "LLM-Powered-Autonomous-Agents_LilLog.pdf"


class RAGTestBase:
    """Base class for RAG service testing with shared infrastructure."""
    
    def __init__(self):
        """Initialize test base with RAG service and collection tracking."""
        self.rag_service = RAGService(chunk_size=1000, chunk_overlap=200)
        self.indexed_collections: Set[str] = set()
    
    async def index_pdf_with_tracking(self, pdf_path: Path) -> str:
        """Index PDF and track collection for cleanup."""
        collection_name = await self.rag_service.index_pdf(pdf_path)
        self.indexed_collections.add(collection_name)
        return collection_name
    
    @staticmethod
    def truncate_content(text: str, max_length: int = 200) -> str:
        """Truncate text with ellipsis if too long."""
        return text[:max_length] + "..." if len(text) > max_length else text
    
    @staticmethod
    def print_test_result(test_name: str, success: bool, details: str = "") -> None:
        """Standardized test result printing."""
        status = "✓" if success else "✗"
        print(f"{status} {test_name}{': ' + details if details else ''}")
    
    def print_document_preview(self, doc: Document, label: str = "Document") -> None:
        """Print formatted document preview."""
        content_preview = self.truncate_content(doc.page_content)
        print(f"{label} preview: {content_preview}")
        print(f"{label} metadata: {doc.metadata}")
    
    async def cleanup_collections(self) -> None:
        """Clean up all tracked collections."""
        if not self.indexed_collections:
            print("No collections to clean up")
            return
        
        print(f"Cleaning up {len(self.indexed_collections)} test collections...")
        
        for collection_name in self.indexed_collections.copy():
            try:
                deleted: bool = await self.rag_service.delete_collection(collection_name)
                self.print_test_result(f"Deleted collection: {collection_name}", deleted)
                if deleted:
                    self.indexed_collections.discard(collection_name)
            except Exception as e:
                print(f"✗ Failed to delete collection {collection_name}: {e}")


# Global test base instance
_test_base: RAGTestBase = RAGTestBase()


def create_rag_service() -> RAGService:
    """Initialize RAG service for testing (backward compatibility)."""
    return _test_base.rag_service


def test_collection_name_generation() -> None:
    """Test collection name generation with various file names."""
    print("Testing collection name generation...")
    
    # Test cases for different file name patterns
    test_cases: List[Tuple[str, str]] = [
        ("LLM-Powered-Autonomous-Agents_LilLog.pdf", "llm_powered_autonomous_agents_lillog_pdf"),
        ("My Document (2023).pdf", "my_document_2023_pdf"),
        ("file@#$%^&*()name.pdf", "file_name_pdf"),
        ("123numbers.pdf", "file_123numbers_pdf"),
        ("spaces   and___dashes--.pdf", "spaces_and_dashes_pdf"),
        ("normal_file.pdf", "normal_file_pdf"),
    ]
    
    for filename, expected in test_cases:
        test_path: Path = Path(filename)
        collection_name: str = _test_base.rag_service._get_collection_name(test_path)
        print(f"'{filename}' -> '{collection_name}' (expected: '{expected}')")
        assert collection_name == expected, f"Mismatch for {filename}"
    
    _test_base.print_test_result("All collection name generation tests", True)


def test_pdf_file_validation() -> None:
    """Test PDF file existence and validation."""
    print("Testing PDF file validation...")
    print(f"Test PDF path: {TEST_PDF}")
    print(f"PDF exists: {TEST_PDF.exists()}")
    if TEST_PDF.exists():
        size: int = TEST_PDF.stat().st_size
        print(f"PDF size: {size:,} bytes")
    else:
        print("PDF size: N/A")


async def test_pdf_indexing() -> str:
    """Test PDF document indexing process."""
    print("Testing PDF indexing...")
    
    try:
        start_time = time.time()
        collection_name: str = await _test_base.index_pdf_with_tracking(TEST_PDF)
        indexing_time = time.time() - start_time
        
        print(f"Successfully indexed PDF into collection: {collection_name}")
        print(f"Indexing time: {indexing_time:.2f} seconds")
        
        # Get collection info for debugging
        collections = await _test_base.rag_service.list_collections()
        print(f"Total collections after indexing: {len(collections)}")
        
        # Test retrieval to confirm indexing worked
        test_docs = await _test_base.rag_service.retrieve("test", collection_name, k=1)
        print(f"Indexed document count verification: {len(test_docs)} documents retrievable")
        
        if test_docs:
            doc = test_docs[0]
            print(f"Sample indexed document metadata keys: {list(doc.metadata.keys())}")
            print(f"Has topic metadata: {'topic' in doc.metadata}")
            print(f"Has key_words metadata: {'key_words' in doc.metadata}")
            
            # Show batch processing worked
            all_test_docs = await _test_base.rag_service.retrieve("", collection_name, k=5)
            batch_success = sum(1 for doc in all_test_docs if doc.metadata.get('topic') and doc.metadata.get('key_words'))
            print(f"Batch processing success rate: {batch_success}/{len(all_test_docs)} documents with topic+keywords")
        
        return collection_name
    except Exception as e:
        print(f"PDF indexing failed: {e}")
        raise


async def test_retrieval_and_filtering() -> None:
    """Test document retrieval with various filters."""
    print("Testing document retrieval and filtering...")
    
    collection_name: str = await _test_base.index_pdf_with_tracking(TEST_PDF)
    
    try:
        # Basic retrieval
        docs = await _test_base.rag_service.retrieve("What are autonomous agents?", collection_name, k=3)
        print(f"Basic retrieval: {len(docs)} documents")
        if docs:
            _test_base.print_document_preview(docs[0], "First document")
            print(f"First document full metadata: {docs[0].metadata}")
        
        # Get all docs to analyze available topics for debugging
        all_docs = await _test_base.rag_service.retrieve("", collection_name, k=10)
        topics = [doc.metadata.get('topic', 'N/A') for doc in all_docs[:5]]  # Sample first 5
        print(f"Sample of available topics: {topics}")
        
        # Page filtering
        page_docs = await _test_base.rag_service.retrieve(
            "planning", collection_name, k=2, filter_conditions={"page": 0}
        )
        print(f"Page 0 filter: {len(page_docs)} documents")
        if page_docs:
            print(f"Page 0 document topics: {[doc.metadata.get('topic', 'N/A') for doc in page_docs]}")
        
        # Topic filtering - use actual topic from sample docs 
        sample_topic = topics[0] if topics else "N/A"
        topic_docs = await _test_base.rag_service.retrieve(
            "memory", collection_name, k=2, filter_conditions={"topic": sample_topic}
        )
        print(f"Topic filter (using '{sample_topic}'): {len(topic_docs)} documents")
        
        # Try with a topic that should exist based on content
        memory_topics = [t for t in topics if 'memory' in t.lower() or 'Memory' in t]
        if memory_topics:
            memory_topic_docs = await _test_base.rag_service.retrieve(
                "memory", collection_name, k=2, filter_conditions={"topic": memory_topics[0]}
            )
            print(f"Topic filter (memory-related '{memory_topics[0]}'): {len(memory_topic_docs)} documents")
        else:
            print(f"No memory-related topics found in: {topics}")
        
        # With analysis
        analysis_docs = await _test_base.rag_service.retrieve(
            "Find information about planning from page 0", collection_name, k=2, enable_analysis=True
        )
        print(f"Analysis-enabled retrieval: {len(analysis_docs)} documents")
        if analysis_docs:
            print(f"Analysis result topics: {[doc.metadata.get('topic', 'N/A') for doc in analysis_docs]}")
        
        _test_base.print_test_result("Retrieval and filtering tests", True)
        
    except Exception as e:
        _test_base.print_test_result("Retrieval and filtering tests", False, str(e))
        raise


async def test_answer_generation() -> str:
    """Test answer generation with context documents."""
    print("Testing answer generation...")
    
    collection_name = await _test_base.index_pdf_with_tracking(TEST_PDF)
    documents = await _test_base.rag_service.retrieve("What are autonomous agents?", collection_name, k=3)
    
    if not documents:
        print("No documents to test answer generation")
        return ""
    
    query: str = "What are the key components of autonomous agents?"
    try:
        answer: str = await _test_base.rag_service.generate_answer(query, documents[:2])
        print(f"Generated answer for: '{query}'")
        print(f"Answer: {answer}")
        return answer
    except Exception as e:
        print(f"Answer generation failed: {e}")
        raise


async def test_query_pipeline() -> None:
    """Test complete RAG query pipeline with different modes."""
    print("Testing query pipeline...")
    
    collection_name = await _test_base.index_pdf_with_tracking(TEST_PDF)
    
    try:
        # Basic query
        result = await _test_base.rag_service.query(
            "How do autonomous agents work with language models?", collection_name, k=3
        )
        print(f"Basic query - Answer: {_test_base.truncate_content(result['answer'], 100)}")
        print(f"Sources: {result['metadata']['num_sources']}")
        
        # Query with analysis
        smart_result = await _test_base.rag_service.query(
            "What are the planning components from page 0?", collection_name, k=2, enable_analysis=True
        )
        print(f"Smart query - Answer: {_test_base.truncate_content(smart_result['answer'], 100)}")
        if "analysis" in smart_result["metadata"]:
            analysis = smart_result["metadata"]["analysis"]
            print(f"Analysis: {analysis.get('analyzed_query', 'N/A')}, Filters: {analysis.get('filters_applied', 'None')}")
        
        _test_base.print_test_result("Query pipeline tests", True)
        
    except Exception as e:
        _test_base.print_test_result("Query pipeline tests", False, str(e))
        raise


async def test_query_analysis() -> None:
    """Test structured query analysis."""
    print("Testing query analysis...")
    
    queries = [
        "What are autonomous agents?",
        "Find information about planning from page 0", 
        "Show me memory components",
        "Tell me about task decomposition from the first page",
        "What are the key concepts in LLM agents?"
    ]
    
    try:
        print("Query Analysis Results:")
        print("-" * 80)
        for i, query in enumerate(queries, 1):
            analysis = await _test_base.rag_service.analyze_query(query)
            print(f"{i}. Query: '{query}'")
            print(f"   -> Parsed Query: '{analysis.query}'")
            print(f"   -> Page Filter: {analysis.page_filter}")
            print(f"   -> Topic: {analysis.topic}")
            print(f"   -> Keywords: {analysis.key_words}")
            
            print()
        
        _test_base.print_test_result("Query analysis", True)
        
    except Exception as e:
        _test_base.print_test_result("Query analysis", False, str(e))


async def test_collection_management() -> None:
    """Test collection listing and deletion."""
    print("Testing collection management...")
    
    collection_name = await _test_base.index_pdf_with_tracking(TEST_PDF)
    
    try:
        collections = await _test_base.rag_service.list_collections()
        print(f"Available collections: {len(collections)}")
        print(f"Target collection exists: {collection_name in collections}")
        _test_base.print_test_result("Collection management", True)
    except Exception as e:
        _test_base.print_test_result("Collection management", False, str(e))
        raise


async def test_dynamic_mappings() -> None:
    """Test dynamic field mappings with refactored method names."""
    print("Testing dynamic mappings with new method names...")
    
    try:
        # Test filter mappings
        filter_mappings = _test_base.rag_service._get_query_to_database_filter_mapping()
        print(f"Filter mappings (StructuredQuery -> Document metadata):")
        for source, target in filter_mappings.items():
            print(f"  {source} -> {target}")
        
        # Test metadata mappings
        metadata_mappings = _test_base.rag_service._get_analysis_to_metadata_field_mapping()
        print(f"Metadata mappings (StructuredQuery -> Analysis metadata):")
        for source, target in metadata_mappings.items():
            print(f"  {source} -> {target}")
        
        # Test with actual analysis to show mappings in action
        collection_name = await _test_base.index_pdf_with_tracking(TEST_PDF)
        result = await _test_base.rag_service.query(
            "Find planning information from page 0", collection_name, k=1, enable_analysis=True
        )
        
        if "analysis" in result["metadata"]:
            print(f"Actual analysis metadata keys: {list(result['metadata']['analysis'].keys())}")
            print(f"Analysis metadata: {result['metadata']['analysis']}")
        
        _test_base.print_test_result("Dynamic mappings (refactored methods)", True)
        
    except Exception as e:
        _test_base.print_test_result("Dynamic mappings (refactored methods)", False, str(e))


async def test_metadata_extraction() -> None:
    """Test topic and keywords metadata extraction."""
    print("Testing metadata extraction...")
    
    collection_name = await _test_base.index_pdf_with_tracking(TEST_PDF)
    
    try:
        # Get some documents to check their metadata - use empty query to get diverse results
        docs = await _test_base.rag_service.retrieve("", collection_name, k=5)
        
        print(f"Retrieved {len(docs)} documents for metadata analysis")
        print("=" * 80)
        
        if docs:
            for i, doc in enumerate(docs, 1):
                print(f"Document {i}:")
                print(f"  Content preview: {_test_base.truncate_content(doc.page_content, 80)}")
                print(f"  Document topic (from chunk analysis): {doc.metadata.get('topic', 'N/A')}")
                print(f"  Document keywords (from chunk analysis): {doc.metadata.get('key_words', 'N/A')}")
                print(f"  Page: {doc.metadata.get('page', 'N/A')}")
                print(f"  Chunk Index: {doc.metadata.get('chunk_index', 'N/A')}")
                print(f"  Total metadata keys: {len(doc.metadata)}")
                print("-" * 40)
            
            # Batch processing confirmation
            topics_count = sum(1 for doc in docs if doc.metadata.get('topic'))
            keywords_count = sum(1 for doc in docs if doc.metadata.get('key_words'))
            print(f"Batch processing results:")
            print(f"  Documents with topics: {topics_count}/{len(docs)}")
            print(f"  Documents with keywords: {keywords_count}/{len(docs)}")
            
            _test_base.print_test_result("Metadata extraction", True)
        else:
            _test_base.print_test_result("Metadata extraction", False, "No documents found")
            
    except Exception as e:
        _test_base.print_test_result("Metadata extraction", False, str(e))


async def test_error_handling() -> None:
    """Test error handling scenarios."""
    print("Testing error handling...")
    
    # Test with non-existent file
    try:
        await _test_base.rag_service.index_pdf("nonexistent.pdf")
        print("ERROR: Should have failed with non-existent file")
    except FileNotFoundError as e:
        _test_base.print_test_result("Correctly handled non-existent file", True, type(e).__name__)
    except Exception as e:
        _test_base.print_test_result("Unexpected error for non-existent file", False, str(e))
    
    # Test retrieval from non-existent collection
    try:
        await _test_base.rag_service.retrieve("test query", "nonexistent_collection")
        print("ERROR: Should have failed with non-existent collection")
    except ValueError as e:
        _test_base.print_test_result("Correctly handled non-existent collection", True, type(e).__name__)
    except Exception as e:
        _test_base.print_test_result("Unexpected error for non-existent collection", False, str(e))


async def cleanup_collections() -> None:
    """Clean up collections created during testing."""
    await _test_base.cleanup_collections()


async def main() -> None:
    """Run all RAG service tests."""
    print("=== Running RAG Service Tests ===\n")
    
    try:
        test_collection_name_generation()
        print()
        
        test_pdf_file_validation()
        print()
        
        await test_pdf_indexing()
        print()
        
        await test_retrieval_and_filtering()
        print()
        
        await test_answer_generation()
        print()
        
        await test_query_pipeline()
        print()
        
        await test_query_analysis()
        print()
        
        await test_collection_management()
        print()
        
        await test_dynamic_mappings()
        print()
        
        await test_metadata_extraction()
        print()
        
        await test_error_handling()
        print()
        
        print("=== All RAG service tests completed ===\n")
        
    finally:
        # Always cleanup, even if tests fail
        await cleanup_collections()


if __name__ == "__main__":
    asyncio.run(main())