from pathlib import Path
import sys
import asyncio
sys.path.append(str(Path(__file__).resolve().parents[4]))

from backend.libs.langchain.loaders import PDFLoader

# Common test variables
current_file = Path(__file__).resolve()
backend_root = current_file.parents[3]  # Go up from backend/libs/langchain/ to backend/
TEST_PDF = backend_root / "tests" / "data" / "CV_Doan_Trong_Hieu.pdf"
MULTIPLE_PDF_FILES = [
    TEST_PDF,
    backend_root / "tests" / "data" / "CV_Doan_Trong_Hieu-1.pdf",  # Example additional file
]


def test_basic_pdf_loading():
    """Test basic PDF loading using class method."""
    print("Testing basic PDF loading...")
    docs = PDFLoader.load_pdf(TEST_PDF)
    print(f"Loaded {len(docs)} documents")
    return docs


def test_instance_loading():
    """Test PDF loading using instance methods."""
    print("Testing instance loading methods...")
    loader = PDFLoader(TEST_PDF)
    
    # Default loading
    docs = loader.load()
    print(f"Default load: {len(docs)} documents")
    
    # Explicit non-lazy loading
    docs = loader.load(lazy=False)
    print(f"Non-lazy load: {len(docs)} documents")
    
    # Lazy loading
    doc_iter = loader.load(lazy=True)
    doc_count = sum(1 for _ in doc_iter)
    print(f"Lazy load: {doc_count} documents")


def test_loading_modes():
    """Test different PDF loading modes."""
    print("Testing different loading modes...")
    
    # Page mode (default)
    loader = PDFLoader(TEST_PDF, mode="page")
    docs = loader.load()
    print(f"Page mode: {len(docs)} documents (one per page)")
    
    # Single mode
    loader = PDFLoader(TEST_PDF, mode="single")
    docs = loader.load()
    print(f"Single mode: {len(docs)} document (entire PDF)")
    
    # Single mode with custom delimiter
    loader = PDFLoader(
        TEST_PDF, 
        mode="single",
        pages_delimiter="\n\n--- NEW PAGE ---\n\n"
    )
    docs = loader.load()
    print(f"Single mode with delimiter: {len(docs)} document")


def test_multiple_pdf_files():
    """Test loading multiple PDF files."""
    print("Testing multiple PDF file loading...")
    docs = PDFLoader.load_pdf(MULTIPLE_PDF_FILES)
    print(f"Loaded {len(docs)} documents from {len(MULTIPLE_PDF_FILES)} files")


def test_lazy_loading():
    """Test lazy loading for memory efficiency."""
    print("Testing lazy loading...")
    loader = PDFLoader(TEST_PDF)
    doc_count = 0
    for doc in loader.load(lazy=True):
        doc_count += 1
        print(f"Processing document {doc_count}: {doc.metadata.get('file_name', 'Unknown')}")
        if doc_count >= 3:  # Limit output for demo
            break
    print(f"Processed {doc_count} documents lazily")


def test_image_extraction():
    """Test PDF loading with image extraction."""
    print("Testing image extraction...")
    loader = PDFLoader(
        TEST_PDF,
        mode="page",
        extract_images=True,
        images_inner_format="markdown-img",
    )
    docs_with_images = loader.load()
    print(f"Loaded {len(docs_with_images)} documents with image extraction")


def test_table_extraction():
    """Test PDF loading with table extraction."""
    print("Testing table extraction...")
    loader = PDFLoader(
        TEST_PDF,
        mode="page",
        extract_tables=True,
        table_format="markdown"
    )
    docs_with_tables = loader.load()
    print(f"Loaded {len(docs_with_tables)} documents with table extraction")


def test_combined_extraction():
    """Test PDF loading with both image and table extraction."""
    print("Testing combined image and table extraction...")
    loader = PDFLoader(
        TEST_PDF,
        mode="page",
        extract_images=True,
        extract_tables=True,
        table_format="html",
        images_inner_format="markdown-img"
    )
    docs_full_extraction = loader.load()
    print(f"Loaded {len(docs_full_extraction)} documents with full extraction")


def test_advanced_features():
    """Test advanced loading features following exact original patterns."""
    print("Testing advanced features...")
    
    # Safe loading by default - automatically handles errors
    docs = PDFLoader.load_pdf(TEST_PDF, mode="page")
    doc_iter = PDFLoader.load_pdf(TEST_PDF, mode="page", lazy=True)
    print(f"Safe loading: {len(docs)} documents")
    doc_count = sum(1 for _ in doc_iter)
    print(f"Safe lazy loading: {doc_count} documents")
    
    # Silent mode - suppresses error messages
    docs = PDFLoader.load_pdf("nonexistent.pdf", silent=True)  # Returns []
    docs = PDFLoader.load_pdf("document.txt", silent=True)     # Returns []
    print("Silent mode tested (returns empty list for errors)")
    
    # Advanced loading with image/table extraction
    docs = PDFLoader.load_pdf(
        TEST_PDF, 
        mode="page",
        extract_images=True,
        extract_tables=True,
        table_format="markdown"
    )
    print(f"Advanced loading with extraction: {len(docs)} documents")
    
    # Combination of all features - safe by default
    docs = PDFLoader.load_pdf(
        TEST_PDF,
        mode="page", 
        extract_images=True,
        extract_tables=True,
        lazy=True
    )
    doc_count = sum(1 for _ in docs)
    print(f"All features with lazy loading: {doc_count} documents")
    
    # Configuration integration examples
    # The loader automatically uses settings from backend/core/config.py:
    # - settings.openai_api_key for LLM image extraction
    # - settings.llm_model as default model (can be overridden)
    print("Configuration: Uses settings.openai_api_key and settings.llm_model")
    
    # Example with explicit model override
    loader = PDFLoader(
        TEST_PDF,
        extract_images=True,
        extract_tables=True
    )
    docs = loader.load()
    print(f"Model override example: {len(docs)} documents")


def main():
    """Run all PDF loader tests."""
    print("=== Running PDF Loader Tests ===\n")
    
    test_basic_pdf_loading()
    print()
    
    test_instance_loading()
    print()
    
    test_loading_modes()
    print()
    
    test_multiple_pdf_files()
    print()
    
    test_lazy_loading()
    print()
    
    test_image_extraction()
    print()
    
    test_table_extraction()
    print()
    
    test_combined_extraction()
    print()
    
    test_advanced_features()
    print()
    
    print("=== All PDF loader tests completed ===")


if __name__ == "__main__":
    main()