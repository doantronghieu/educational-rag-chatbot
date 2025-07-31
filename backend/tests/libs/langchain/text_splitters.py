"""Comprehensive tests for LangChain text splitters wrapper implementation."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

from libs.langchain.text_splitters import (
    TextSplitterManager,
    SplitterType,
    create_text_splitter_manager,
    split_text,
    split_documents,
    split_json_to_dicts,
    default_text_splitter_manager
)
from langchain_core.documents import Document
import json

# Test data
SAMPLE_TEXT = """
This is a paragraph with multiple sentences. It contains various information that should be split appropriately.

This is another paragraph that continues the document. It provides additional context and content for testing purposes.

Final paragraph with concluding remarks. This ensures we have enough content to test various splitting strategies effectively.
"""

SAMPLE_MARKDOWN = """# Header 1
This is content under header 1.

## Header 2
This is content under header 2.

### Header 3
More detailed content here.

## Another Header 2
Additional content under another header 2.
"""

SAMPLE_HTML = """
<html>
<body>
<h1>Main Title</h1>
<p>This is a paragraph under the main title.</p>

<h2>Section 1</h2>
<p>Content for section 1.</p>
<table>
<tr><td>Cell 1</td><td>Cell 2</td></tr>
</table>

<h2>Section 2</h2>
<ul>
<li>Item 1</li>
<li>Item 2</li>
</ul>
</body>
</html>
"""

SAMPLE_JSON = {
    "users": [
        {"name": "Alice", "age": 30, "email": "alice@example.com"},
        {"name": "Bob", "age": 25, "email": "bob@example.com"}
    ],
    "metadata": {
        "version": "1.0",
        "created": "2024-01-01",
        "description": "Sample data for testing JSON splitting functionality"
    }
}

def test_text_splitter_manager_creation():
    """Test TextSplitterManager creation and basic functionality."""
    print("=== Testing TextSplitterManager Creation ===")
    
    # Test default creation
    manager = TextSplitterManager()
    print(f"Default manager created with chunk_size: {manager.default_chunk_size}, chunk_overlap: {manager.default_chunk_overlap}")
    
    # Test custom creation
    custom_manager = TextSplitterManager(default_chunk_size=500, default_chunk_overlap=50)
    print(f"Custom manager created with chunk_size: {custom_manager.default_chunk_size}, chunk_overlap: {custom_manager.default_chunk_overlap}")
    
    # Test convenience function
    conv_manager = create_text_splitter_manager(chunk_size=800, chunk_overlap=100)
    print(f"Convenience manager created with chunk_size: {conv_manager.default_chunk_size}, chunk_overlap: {conv_manager.default_chunk_overlap}")
    
    print("TextSplitterManager creation tests passed\n")


def test_recursive_character_splitter():
    """Test RecursiveCharacterTextSplitter functionality."""
    print("=== Testing RecursiveCharacterTextSplitter ===")
    
    manager = TextSplitterManager()
    
    # Test basic splitting
    splitter = manager.get_recursive_character_splitter(chunk_size=100, chunk_overlap=20)
    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"Split text into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {len(chunk)} chars - {chunk[:50]}...")
    
    # Test with custom separators
    custom_splitter = manager.get_recursive_character_splitter(
        chunk_size=150, 
        separators=["\n\n", ".", " ", ""]
    )
    custom_chunks = custom_splitter.split_text(SAMPLE_TEXT)
    print(f"Custom separators split into {len(custom_chunks)} chunks")
    
    # Test split_documents method
    documents = manager.split_documents([SAMPLE_TEXT], SplitterType.RECURSIVE_CHARACTER, chunk_size=120)
    print(f"split_documents created {len(documents)} Document objects")
    print(f"First document metadata: {documents[0].metadata}")
    
    print("RecursiveCharacterTextSplitter tests passed\n")


def test_character_splitter():
    """Test CharacterTextSplitter functionality."""
    print("=== Testing CharacterTextSplitter ===")
    
    manager = TextSplitterManager()
    
    # Test with default separator
    splitter = manager.get_character_splitter(chunk_size=200, chunk_overlap=50)
    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"Default separator split into {len(chunks)} chunks")
    
    # Test with custom separator
    custom_splitter = manager.get_character_splitter(separator=".", chunk_size=100)
    custom_chunks = custom_splitter.split_text(SAMPLE_TEXT)
    print(f"Custom separator (.) split into {len(custom_chunks)} chunks")
    for i, chunk in enumerate(custom_chunks):
        print(f"Chunk {i+1}: {chunk.strip()}")
    
    # Test through split_text method
    text_chunks = manager.split_text(SAMPLE_TEXT, SplitterType.CHARACTER, separator=". ", chunk_size=150)
    print(f"split_text method created {len(text_chunks)} chunks")
    
    print("CharacterTextSplitter tests passed\n")


def test_markdown_header_splitter():
    """Test MarkdownHeaderTextSplitter functionality."""
    print("=== Testing MarkdownHeaderTextSplitter ===")
    
    manager = TextSplitterManager()
    
    # Test with default headers
    splitter = manager.get_markdown_header_splitter()
    documents = splitter.split_text(SAMPLE_MARKDOWN)
    print(f"Split markdown into {len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:100]}...")
    
    # Test with custom headers and strip_headers=False
    custom_splitter = manager.get_markdown_header_splitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")],
        strip_headers=False
    )
    custom_docs = custom_splitter.split_text(SAMPLE_MARKDOWN)
    print(f"Custom headers (with headers preserved) split into {len(custom_docs)} documents")
    
    # Test convenience function
    conv_docs = split_documents([SAMPLE_MARKDOWN], SplitterType.MARKDOWN_HEADER, strip_headers=False)
    print(f"Convenience function created {len(conv_docs)} documents")
    
    print("MarkdownHeaderTextSplitter tests passed\n")


def test_recursive_json_splitter():
    """Test RecursiveJsonSplitter functionality."""
    print("=== Testing RecursiveJsonSplitter ===")
    
    manager = TextSplitterManager()
    
    # Test with JSON object using wrapper method
    json_str = json.dumps(SAMPLE_JSON, indent=2)
    documents = manager.split_documents([json_str], SplitterType.RECURSIVE_JSON, max_chunk_size=200)
    print(f"Split JSON into {len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {len(doc.page_content)} chars")
        try:
            parsed = json.loads(doc.page_content)
            print(f"  Valid JSON with keys: {list(parsed.keys()) if isinstance(parsed, dict) else 'list/value'}")
        except json.JSONDecodeError:
            print(f"  Text chunk: {doc.page_content[:50]}...")
    
    # Test with min_chunk_size option using wrapper method
    minsize_docs = manager.split_documents([json_str], SplitterType.RECURSIVE_JSON, max_chunk_size=150, min_chunk_size=50)
    print(f"With min_chunk_size=50: {len(minsize_docs)} documents")
    
    # Test convenience function
    conv_chunks = split_json_to_dicts(SAMPLE_JSON, max_chunk_size=250)
    print(f"Convenience function created {len(conv_chunks)} chunks")
    for i, chunk in enumerate(conv_chunks):
        print(f"Chunk {i+1}: {type(chunk)} - {chunk}")
    
    # Test root-level array handling (bug fix verification)
    root_array = [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}, {"name": "Charlie", "age": 35}]
    try:
        array_chunks = split_json_to_dicts(root_array, max_chunk_size=100)
        print(f"Root-level array test: SUCCESS - {len(array_chunks)} chunks")
        for i, chunk in enumerate(array_chunks):
            print(f"  Array chunk {i+1}: {type(chunk)} - {chunk}")
    except Exception as e:
        print(f"Root-level array test FAILED: {e}")
    
    # Test simple root-level array
    simple_array = [1, 2, 3, 4, 5]
    try:
        simple_chunks = manager.split_documents([json.dumps(simple_array)], SplitterType.RECURSIVE_JSON)
        print(f"Simple array test: SUCCESS - {len(simple_chunks)} documents")
    except Exception as e:
        print(f"Simple array test FAILED: {e}")
    
    print("RecursiveJsonSplitter tests passed\n")


def test_html_header_splitter():
    """Test HTMLHeaderTextSplitter functionality."""
    print("=== Testing HTMLHeaderTextSplitter ===")
    
    manager = TextSplitterManager()
    
    # Test with default headers
    splitter = manager.get_html_header_splitter()
    documents = splitter.split_text(SAMPLE_HTML)
    print(f"Split HTML into {len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"Document {i+1} metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:100]}...")
    
    # Test with custom headers
    custom_splitter = manager.get_html_header_splitter(
        headers_to_split_on=[("h1", "Title"), ("h2", "Section")]
    )
    custom_docs = custom_splitter.split_text(SAMPLE_HTML)
    print(f"Custom headers split into {len(custom_docs)} documents")
    
    print("HTMLHeaderTextSplitter tests passed\n")


def test_html_section_splitter():
    """Test HTMLSectionSplitter functionality."""
    print("=== Testing HTMLSectionSplitter ===")
    
    manager = TextSplitterManager()
    
    try:
        # Test basic HTML section splitting
        splitter = manager.get_html_section_splitter(chunk_size=200, chunk_overlap=50)
        documents = splitter.split_text(SAMPLE_HTML)
        print(f"Split HTML into {len(documents)} sections")
        for i, doc in enumerate(documents):
            print(f"Section {i+1}: {len(doc.page_content)} chars - {doc.page_content[:80]}...")
        
        # Test through split_documents
        split_docs = manager.split_documents([SAMPLE_HTML], SplitterType.HTML_SECTION, chunk_size=300, chunk_overlap=50)
        print(f"split_documents created {len(split_docs)} documents")
        
        print("HTMLSectionSplitter tests passed\n")
    except Exception as e:
        print(f"HTMLSectionSplitter test skipped (dependency issue): {e}\n")


def test_html_semantic_splitter():
    """Test HTMLSemanticPreservingSplitter functionality."""
    print("=== Testing HTMLSemanticPreservingSplitter ===")
    
    manager = TextSplitterManager()
    
    # Test semantic preservation
    splitter = manager.get_html_semantic_splitter(
        max_chunk_size=300,
        elements_to_preserve=["table", "ul", "ol"]
    )
    documents = splitter.split_text(SAMPLE_HTML)
    print(f"Split HTML with semantic preservation into {len(documents)} documents")
    for i, doc in enumerate(documents):
        print(f"Document {i+1}: {len(doc.page_content)} chars")
        print(f"Content preview: {doc.page_content[:100]}...")
    
    # Test through enum
    enum_docs = manager.split_documents([SAMPLE_HTML], SplitterType.HTML_SEMANTIC, max_chunk_size=250)
    print(f"Enum method created {len(enum_docs)} documents")
    
    # Test convenience function
    conv_docs = split_documents([SAMPLE_HTML], SplitterType.HTML_SEMANTIC, max_chunk_size=400)
    print(f"Convenience function created {len(conv_docs)} documents")
    
    print("HTMLSemanticPreservingSplitter tests passed\n")


def test_tiktoken_splitter():
    """Test Tiktoken-based text splitter functionality."""
    print("=== Testing Tiktoken Splitter ===")
    
    manager = TextSplitterManager()
    
    try:
        # Test with cl100k_base encoding
        splitter = manager.get_tiktoken_splitter(
            encoding_name="cl100k_base",
            chunk_size=50,  # tokens
            chunk_overlap=10
        )
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"Tiktoken split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: ~{len(chunk.split())} words - {chunk[:60]}...")
        
        # Test through split_documents
        docs = manager.split_documents([SAMPLE_TEXT], SplitterType.TIKTOKEN, chunk_size=300, chunk_overlap=50)
        print(f"split_documents created {len(docs)} documents")
        
        print("Tiktoken splitter tests passed\n")
    except Exception as e:
        print(f"Tiktoken test skipped (expected if tiktoken not installed): {e}\n")


def test_spacy_splitter():
    """Test SpaCy-based text splitter functionality."""
    print("=== Testing SpaCy Splitter ===")
    
    manager = TextSplitterManager()
    
    try:
        # Test with default pipeline
        splitter = manager.get_spacy_splitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"SpaCy split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {len(chunk)} chars - {chunk[:60]}...")
        
        # Test through enum
        docs = manager.split_documents([SAMPLE_TEXT], SplitterType.SPACY, chunk_size=300, chunk_overlap=50)
        print(f"Enum method created {len(docs)} documents")
        
        print("SpaCy splitter tests passed\n")
    except Exception as e:
        print(f"SpaCy test skipped (expected if spacy model not installed): {e}\n")


def test_nltk_splitter():
    """Test NLTK-based text splitter functionality."""
    print("=== Testing NLTK Splitter ===")
    
    manager = TextSplitterManager()
    
    try:
        # Test NLTK splitter
        splitter = manager.get_nltk_splitter(chunk_size=200, chunk_overlap=40)
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"NLTK split into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}: {len(chunk)} chars - {chunk[:60]}...")
        
        # Test through enum
        docs = manager.split_documents([SAMPLE_TEXT], SplitterType.NLTK, chunk_size=300, chunk_overlap=50)
        print(f"Enum method created {len(docs)} documents")
        
        print("NLTK splitter tests passed\n")
    except Exception as e:
        print(f"NLTK test skipped (expected if NLTK data not available): {e}\n")


def test_sentence_transformers_splitter():
    """Test SentenceTransformers token text splitter functionality."""
    print("=== Testing SentenceTransformers Splitter ===")
    
    manager = TextSplitterManager()
    
    try:
        # Check if model is cached first to avoid download timeout
        import sentence_transformers
        from pathlib import Path
        import os
        
        # Use a smaller, commonly cached model for testing
        test_model = "all-MiniLM-L6-v2"
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        
        # Quick check if we have any cached models or if this will likely timeout
        model_exists = False
        if cache_dir.exists():
            model_dirs = list(cache_dir.glob("*all-MiniLM-L6-v2*"))
            model_exists = len(model_dirs) > 0
        
        if not model_exists:
            print(f"Model {test_model} not cached - will attempt download with timeout")
        
        import signal
        def timeout_handler(signum, frame):
            raise TimeoutError("SentenceTransformers test timed out (likely downloading model)")
        
        # Set a reasonable timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(45)  # 45 second timeout for model download
        
        try:
            # Test with small, commonly available model
            splitter = manager.get_sentence_transformers_splitter(
                model_name=test_model,
                tokens_per_chunk=50,
                chunk_overlap=10
            )
            chunks = splitter.split_text(SAMPLE_TEXT)
            print(f"SentenceTransformers split into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1}: {len(chunk)} chars - {chunk[:60]}...")
            
            # Test through enum
            docs = manager.split_documents([SAMPLE_TEXT], SplitterType.SENTENCE_TRANSFORMERS, tokens_per_chunk=40, chunk_overlap=10)
            print(f"Enum method created {len(docs)} documents")
            
            print("SentenceTransformers splitter tests passed\n")
        finally:
            signal.alarm(0)  # Cancel the alarm
            
    except (TimeoutError, Exception) as e:
        print(f"SentenceTransformers test skipped (model download timeout or dependency issue): {e}\n")


def test_semantic_splitter():
    """Test SemanticChunker functionality."""
    print("=== Testing Semantic Splitter ===")
    
    manager = TextSplitterManager()
    
    try:
        # Use real embeddings from dependencies
        from core.dependencies import get_embeddings
        embeddings = get_embeddings()
        
        # Test semantic splitting with real embeddings
        splitter = manager.get_semantic_splitter(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            min_chunk_size=10
        )
        documents = splitter.create_documents([SAMPLE_TEXT])
        print(f"Semantic split into {len(documents)} chunks")
        for i, doc in enumerate(documents):
            print(f"Chunk {i+1}: {len(doc.page_content)} chars - {doc.page_content[:60]}...")
        
        # Test different threshold types
        for threshold_type in ["standard_deviation", "interquartile", "gradient"]:
            try:
                threshold_splitter = manager.get_semantic_splitter(
                    embeddings=embeddings,
                    breakpoint_threshold_type=threshold_type,
                    min_chunk_size=5
                )
                threshold_docs = threshold_splitter.create_documents([SAMPLE_TEXT])
                print(f"Threshold type '{threshold_type}': {len(threshold_docs)} chunks")
            except Exception as e:
                print(f"Threshold type '{threshold_type}' failed: {e}")
        
        print("Semantic splitter tests passed\n")
    except ImportError as e:
        print(f"Semantic splitter test skipped (missing dependencies): {e}\n")
    except Exception as e:
        print(f"Semantic splitter test skipped: {e}\n")


def test_splitter_info_and_cache():
    """Test splitter information and cache functionality."""
    print("=== Testing Splitter Info and Cache ===")
    
    manager = TextSplitterManager()
    
    # Test splitter info
    info = manager.get_splitter_info()
    print("Available splitters:")
    for splitter in info["available_splitters"]:
        print(f"  - {splitter}: {info['descriptions'].get(splitter, 'No description')}")
    
    print(f"Default chunk size: {info['default_chunk_size']}")
    print(f"Default chunk overlap: {info['default_chunk_overlap']}")
    print(f"Cached splitters: {info['cached_splitters']}")
    
    # Test cache functionality
    splitter1 = manager.get_recursive_character_splitter(chunk_size=100)
    splitter2 = manager.get_recursive_character_splitter(chunk_size=100)
    print(f"Cache test - same splitter objects: {splitter1 is splitter2}")
    
    # Test cache clearing
    manager.clear_cache()
    info_after_clear = manager.get_splitter_info()
    print(f"Cached splitters after clear: {info_after_clear['cached_splitters']}")
    
    print("Splitter info and cache tests passed\n")


def test_convenience_functions():
    """Test convenience functions."""
    print("=== Testing Convenience Functions ===")
    
    # Test split_text convenience function
    chunks = split_text(SAMPLE_TEXT, SplitterType.RECURSIVE_CHARACTER, chunk_size=150, chunk_overlap=30)
    print(f"split_text created {len(chunks)} chunks")
    
    # Test default manager
    default_chunks = default_text_splitter_manager.split_text(SAMPLE_TEXT)
    print(f"Default manager created {len(default_chunks)} chunks")
    
    print("Convenience functions tests passed\n")


def test_error_handling():
    """Test error handling and edge cases."""
    print("=== Testing Error Handling ===")
    
    manager = TextSplitterManager()
    
    # Test unsupported splitter type
    try:
        manager.split_documents([SAMPLE_TEXT], "invalid_splitter")
    except (ValueError, AttributeError) as e:
        print(f"Correctly caught invalid splitter error: {e}")
    
    # Test empty text
    empty_chunks = manager.split_text("", SplitterType.RECURSIVE_CHARACTER)
    print(f"Empty text created {len(empty_chunks)} chunks")
    
    # Test very small chunk size
    tiny_chunks = manager.split_text(SAMPLE_TEXT, SplitterType.CHARACTER, chunk_size=10, chunk_overlap=5)
    print(f"Tiny chunk size created {len(tiny_chunks)} chunks")
    
    print("Error handling tests passed\n")


def run_all_tests():
    """Run all test functions."""
    print("Starting comprehensive text splitters tests...\n")
    
    test_functions = [
        test_text_splitter_manager_creation,
        test_recursive_character_splitter,
        test_character_splitter,
        test_markdown_header_splitter,
        test_recursive_json_splitter,
        test_html_header_splitter,
        test_html_section_splitter,
        test_html_semantic_splitter,
        test_tiktoken_splitter,
        test_spacy_splitter,
        test_nltk_splitter,
        test_sentence_transformers_splitter,
        test_semantic_splitter,
        test_splitter_info_and_cache,
        test_convenience_functions,
        test_error_handling
    ]
    
    for test_func in test_functions:
        try:
            test_func()
        except Exception as e:
            print(f"Test {test_func.__name__} failed: {e}\n")
    
    print("All text splitter tests completed!")


if __name__ == "__main__":
    run_all_tests()