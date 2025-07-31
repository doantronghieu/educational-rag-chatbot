from pathlib import Path
import sys
import asyncio
sys.path.append(str(Path(__file__).resolve().parents[4]))

from backend.clients.vector_store import VectorStoreClient
from backend.libs.langchain.embeddings import EmbeddingService
from backend.libs.langchain.vector_stores import VectorStoreService
from langchain_core.documents import Document

# Common test variables
VECTOR_STORE_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "documents"
TEST_DOCUMENTS = [
    Document(page_content="Python programming", metadata={"type": "programming"}),
    Document(page_content="Web development", metadata={"type": "web"})
]
TEST_TEXTS = ["Machine learning concepts", "Data science workflows"]
TEST_METADATAS = [{"type": "ml"}, {"type": "data"}]


def create_vector_service():
    """Initialize VectorStoreService for testing."""
    vector_client = VectorStoreClient(VECTOR_STORE_URL)
    embedding_service = EmbeddingService()
    embeddings = embedding_service.embeddings
    
    return VectorStoreService(
        vector_store_client=vector_client,
        embedding_model=embeddings,
        default_collection=DEFAULT_COLLECTION
    )


async def test_add_documents():
    """Test adding documents to vector store."""
    print("Testing document addition...")
    service = create_vector_service()
    
    # Add documents (collection auto-created)
    docs = TEST_DOCUMENTS
    doc_ids = await service.add_documents(docs)
    print(f"Document IDs: {doc_ids}")


async def test_add_texts_with_metadata():
    """Test adding texts with metadata to vector store."""
    print("Testing text addition with metadata...")
    service = create_vector_service()
    
    # Add texts with metadata
    text_ids = await service.add_texts(
        texts=TEST_TEXTS,
        metadatas=TEST_METADATAS
    )
    print(f"Text IDs: {text_ids}")


async def test_similarity_search_operations():
    """Test unified search interface with different parameters."""
    print("Testing similarity search operations...")
    service = create_vector_service()
    
    # Unified search interface
    simple_results = await service.similarity_search("programming", k=3)
    scored_results = await service.similarity_search(
        "programming", 
        k=3, 
        filter_conditions={"type": "programming"},
        with_scores=True,
        score_threshold=0.7
    )
    print(f"Simple search results: {simple_results}")
    print(f"Scored search results: {scored_results}")


async def test_get_retriever():
    """Test getting retriever for chain integration."""
    print("Testing retriever creation...")
    service = create_vector_service()
    
    # Get retriever for chain integration
    retriever = await service.get_retriever(search_kwargs={"k": 5})
    print(f"Retriever: {retriever}")


async def test_multi_collection_usage():
    """Test multi-collection operations."""
    print("Testing multi-collection usage...")
    service = create_vector_service()
    docs = TEST_DOCUMENTS
    
    # Multi-collection usage
    await service.add_documents(docs, collection_name="other_collection")
    results = await service.similarity_search("query", collection_name="other_collection")
    print(f"Added documents to collection: other_collection")
    print(f"Multi-collection search results: {results}")


async def test_client_access_operations():
    """Test client access for advanced operations."""
    print("Testing client access operations...")
    service = create_vector_service()
    
    # Client access for advanced operations
    client_health = await service.client.health_check()
    collection_info = await service.client.get_collection_info(DEFAULT_COLLECTION)
    print(f"Client health: {client_health}")
    print(f"Collection info: {collection_info}")


async def main():
    """Run all vector store tests."""
    print("=== Running Vector Store Tests ===\n")
    
    await test_add_documents()
    print()
    
    await test_add_texts_with_metadata()
    print()
    
    await test_similarity_search_operations()
    print()
    
    await test_get_retriever()
    print()
    
    await test_multi_collection_usage()
    print()
    
    await test_client_access_operations()
    print()
    
    print("=== All tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())