"""Document loaders for various file formats."""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Iterator, List, Optional, Union
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders.parsers import LLMImageBlobParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from core.config import settings


class PDFLoader:
    """PDF document loader using PyMuPDF with flexible configuration options."""
    
    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        mode: str = "page",
        pages_delimiter: str = "\n---PAGE BREAK---\n",
        extract_images: bool = False,
        extract_tables: bool = False,
        table_format: str = "markdown",
        images_inner_format: str = "markdown-img",
        llm_model: Optional[str] = None
    ):
        """
        Initialize PDF loader.
        
        Args:
            file_paths: Path to PDF file(s) - single path or list of paths
            mode: Loading mode - "page" (split by pages) or "single" (entire document)
            pages_delimiter: Delimiter used when mode is "single"
            extract_images: Whether to extract images from PDF using LLM
            extract_tables: Whether to extract tables from PDF
            table_format: Format for extracted tables ("markdown", "html", "csv")
            images_inner_format: Format for extracted images ("markdown-img")
            llm_model: LLM model for image extraction (defaults to config setting)
        """
        # Normalize to list of paths
        if isinstance(file_paths, (str, Path)):
            self.file_paths = [Path(file_paths)]
        else:
            self.file_paths = [Path(fp) for fp in file_paths]
            
        self.mode = mode
        self.pages_delimiter = pages_delimiter
        self.extract_images = extract_images
        self.extract_tables = extract_tables
        self.table_format = table_format
        self.images_inner_format = images_inner_format
        self.llm_model = llm_model or settings.llm_model
        
        # Validate all files
        for file_path in self.file_paths:
            self._validate_pdf_file(file_path)
    
    @staticmethod
    def _validate_pdf_file(file_path: Path) -> None:
        """Validate that file exists and is a PDF."""
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if not file_path.suffix.lower() == ".pdf":
            raise ValueError(f"File must be a PDF: {file_path}")
    
    def _create_loader(self, file_path: Path) -> PyMuPDFLoader:
        """Create PyMuPDFLoader instance for a specific file."""
        loader_kwargs = {
            "mode": self.mode
        }
        
        # Add mode-specific parameters
        if self.mode == "single":
            loader_kwargs["pages_delimiter"] = self.pages_delimiter
        elif self.mode != "page":
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'page' or 'single'")
        
        # Add image extraction parameters
        if self.extract_images:
            loader_kwargs["images_inner_format"] = self.images_inner_format
            loader_kwargs["images_parser"] = LLMImageBlobParser(
                model=ChatOpenAI(
                    model=self.llm_model,
                    max_tokens=1024,
                    api_key=settings.openai_api_key
                )
            )
        
        # Add table extraction parameters
        if self.extract_tables:
            loader_kwargs["extract_tables"] = self.table_format
        
        return PyMuPDFLoader(str(file_path), **loader_kwargs)
    
    def _add_metadata(self, doc: Document, file_path: Path) -> None:
        """Add custom metadata to document."""
        doc.metadata.update({
            "file_name": file_path.name,
        })
    
    def load(self, lazy: bool = False) -> Union[List[Document], Iterator[Document]]:
        """
        Load PDF documents from all files.
        
        Args:
            lazy: If True, returns iterator; if False, returns list (default)
            
        Returns:
            List of documents (lazy=False) or iterator yielding documents (lazy=True)
        """
        def document_generator():
            for file_path in self.file_paths:
                loader = self._create_loader(file_path)
                
                # Use appropriate loading method based on lazy flag
                document_source = loader.lazy_load() if lazy else loader.load()
                
                for doc in document_source:
                    self._add_metadata(doc, file_path)
                    yield doc
        
        if lazy:
            return document_generator()
        else:
            return list(document_generator())
    
    
    @classmethod
    def load_pdf(
        cls,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        mode: str = "page",
        pages_delimiter: str = "\n---PAGE BREAK---\n",
        extract_images: bool = False,
        extract_tables: bool = False,
        table_format: str = "markdown",
        lazy: bool = False,
        silent: bool = False
    ) -> Union[List[Document], Iterator[Document]]:
        """
        Safely load PDF file(s) with unified interface.
        
        Args:
            file_paths: Path to PDF file(s) - single path or list of paths
            mode: Loading mode - "page" or "single"
            pages_delimiter: Delimiter used when mode is "single"
            extract_images: Whether to extract images from PDF using LLM
            extract_tables: Whether to extract tables from PDF
            table_format: Format for extracted tables ("markdown", "html", "csv")
            lazy: If True, returns iterator; if False, returns list
            silent: If True, suppresses error messages
        
        Returns:
            List of documents (lazy=False) or iterator (lazy=True), empty if error
        """
        try:
            loader = cls(
                file_paths=file_paths,
                mode=mode,
                pages_delimiter=pages_delimiter,
                extract_images=extract_images,
                extract_tables=extract_tables,
                table_format=table_format
            )
            return loader.load(lazy=lazy)
        except (FileNotFoundError, ValueError) as e:
            if not silent:
                print(f"Error loading {file_paths}: {e}")
            return [] if not lazy else iter([])


