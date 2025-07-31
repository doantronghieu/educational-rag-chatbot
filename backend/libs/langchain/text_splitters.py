"""LangChain text splitters integration with comprehensive splitting strategies."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from typing import Any, Dict, List, Optional, Union, Literal, Callable
from functools import wraps
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveJsonSplitter,
    HTMLHeaderTextSplitter,
    HTMLSectionSplitter,
    HTMLSemanticPreservingSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters.base import TextSplitter

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import json
import os
from enum import Enum

# Suppress HuggingFace tokenizers parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Default configurations
BASE_HEADERS = [
    ("Header 1", "#", "h1"),
    ("Header 2", "##", "h2"),
    ("Header 3", "###", "h3"),
    ("Header 4", "####", "h4"),
]

def _generate_headers(format_type: str, max_levels: Optional[int] = None) -> List[tuple]:
    """Generate headers for different formats from BASE_HEADERS."""
    headers = BASE_HEADERS if max_levels is None else BASE_HEADERS[:max_levels]
    if format_type == "markdown":
        return [(md, name) for name, md, _ in headers]
    elif format_type == "html":
        return [(html, name) for name, _, html in headers]
    else:
        raise ValueError(f"Unknown format type: {format_type}")

DEFAULT_MARKDOWN_HEADERS = _generate_headers("markdown")
DEFAULT_HTML_HEADERS = _generate_headers("html")
DEFAULT_HTML_SECTION_HEADERS = _generate_headers("html", 3)
DEFAULT_HTML_SEMANTIC_HEADERS = _generate_headers("html", 2)
DEFAULT_HTML_PRESERVE_ELEMENTS = ["table", "ul", "ol", "pre", "code"]


class SplitterType(Enum):
    """Enumeration of available text splitter types."""

    RECURSIVE_CHARACTER = "recursive_character"
    CHARACTER = "character"
    MARKDOWN_HEADER = "markdown_header"
    RECURSIVE_JSON = "recursive_json"
    HTML_HEADER = "html_header"
    HTML_SECTION = "html_section"
    HTML_SEMANTIC = "html_semantic"
    SEMANTIC = "semantic"
    TIKTOKEN = "tiktoken"
    HUGGINGFACE = "huggingface"
    SPACY = "spacy"
    NLTK = "nltk"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class TextSplitterManager:
    """
    Unified text splitter manager providing access to all LangChain text splitting strategies.

    Supports length-based, text-structured, document-structured, and semantic splitting approaches
    with configurable parameters and consistent API across all splitter types.
    """

    def __init__(
        self, default_chunk_size: int = 1000, default_chunk_overlap: int = 200
    ):
        """
        Initialize TextSplitterManager with default configuration.

        Args:
            default_chunk_size: Default maximum chunk size for text splitting
            default_chunk_overlap: Default overlap between chunks
        """
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self._splitters_cache = {}

    def _validate_chunk_params(
        self, chunk_size: Optional[int], chunk_overlap: Optional[int]
    ) -> tuple[int, int]:
        """Validate and normalize chunk size and overlap parameters."""
        final_chunk_size = chunk_size or self.default_chunk_size
        final_chunk_overlap = chunk_overlap or self.default_chunk_overlap

        # Ensure chunk_overlap is not larger than chunk_size
        if final_chunk_overlap >= final_chunk_size:
            final_chunk_overlap = max(0, final_chunk_size // 4)

        return final_chunk_size, final_chunk_overlap

    def _generate_cache_key(self, base_name: str, **params) -> str:
        """Generate consistent cache key from base name and parameters."""
        key_parts = [base_name]
        for key, value in params.items():
            if isinstance(value, (list, tuple)):
                key_parts.append(f"{len(value)}")
            else:
                key_parts.append(str(value))
        return "_".join(key_parts)

    def _get_cached_splitter(self, base_name_or_cache_key: str, splitter_class_or_factory=None, 
                            chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None, 
                            use_chunks: bool = True, **kwargs) -> Any:
        """Get splitter from cache or create new one. Supports both factory function and class-based creation."""
        
        # Handle factory function case (backward compatibility for tiktoken)
        # Only treat as factory function if it's actually a function and not a class
        if (callable(splitter_class_or_factory) and 
            not isinstance(splitter_class_or_factory, type) and 
            not use_chunks):
            cache_key = base_name_or_cache_key
            if cache_key not in self._splitters_cache:
                self._splitters_cache[cache_key] = splitter_class_or_factory()
            return self._splitters_cache[cache_key]
        
        # Handle class-based creation with parameter validation
        cache_params = kwargs.copy()
        if use_chunks:
            final_chunk_size, final_chunk_overlap = self._validate_chunk_params(chunk_size, chunk_overlap)
            cache_params.update({"chunk_size": final_chunk_size, "chunk_overlap": final_chunk_overlap})
        
        cache_key = self._generate_cache_key(base_name_or_cache_key, **cache_params)
        
        if cache_key not in self._splitters_cache:
            if use_chunks:
                self._splitters_cache[cache_key] = splitter_class_or_factory(
                    chunk_size=final_chunk_size, chunk_overlap=final_chunk_overlap, **kwargs)
            else:
                self._splitters_cache[cache_key] = splitter_class_or_factory(**kwargs)
        
        return self._splitters_cache[cache_key]


    def _get_document_processing_strategy(self, splitter_type: SplitterType) -> Callable[[TextSplitter, Document], List[Document]]:
        """Get processing strategy for different splitter types."""
        
        def markdown_strategy(splitter: TextSplitter, doc: Document) -> List[Document]:
            return splitter.split_text(doc.page_content)
        
        def html_strategy(splitter: TextSplitter, doc: Document) -> List[Document]:
            return splitter.split_text(doc.page_content)
        
        def html_semantic_strategy(splitter: TextSplitter, doc: Document) -> List[Document]:
            return self._chunks_to_documents(splitter.split_text(doc.page_content), doc.metadata)
        
        def json_strategy(splitter: RecursiveJsonSplitter, doc: Document) -> List[Document]:
            return self._handle_json_splitting(doc, splitter)
        
        def default_strategy(splitter: TextSplitter, doc: Document) -> List[Document]:
            text_chunks = splitter.split_text(doc.page_content)
            return self._chunks_to_documents(text_chunks, doc.metadata)
        
        strategies = {
            SplitterType.MARKDOWN_HEADER: markdown_strategy,
            SplitterType.HTML_HEADER: html_strategy,
            SplitterType.HTML_SEMANTIC: html_semantic_strategy,
            SplitterType.RECURSIVE_JSON: json_strategy,
        }
        
        return strategies.get(splitter_type, default_strategy)

    def _create_document_from_chunk(self, chunk: Any, metadata: Dict[str, Any] = None) -> Document:
        """Create Document from chunk with metadata copying."""
        if isinstance(chunk, Document):
            return chunk
        return Document(
            page_content=str(chunk),
            metadata=(metadata or {}).copy()
        )

    def _process_json_chunk(self, chunk: Any) -> str:
        """Process JSON chunk into string format."""
        return (
            json.dumps(chunk)
            if isinstance(chunk, (dict, list))
            else str(chunk)
        )

    def _convert_to_documents(self, documents: List[Union[Document, str]]) -> List[Document]:
        """Convert mixed list of strings and Documents to list of Documents."""
        return [
            self._create_document_from_chunk(doc) if isinstance(doc, str) else doc
            for doc in documents
        ]

    def _chunks_to_documents(self, chunks: List[Any], metadata: Dict[str, Any] = None) -> List[Document]:
        """Convert any type of chunks to Documents with metadata."""
        return [
            self._create_document_from_chunk(chunk, metadata)
            for chunk in chunks
        ]

    def _handle_json_splitting(self, doc: Document, splitter: RecursiveJsonSplitter) -> List[Document]:
        """Handle JSON splitting with array wrapping workaround."""
        results = []
        if hasattr(splitter, "split_json"):
            try:
                json_data = json.loads(doc.page_content)

                # Workaround for LangChain bug: wrap root-level arrays in an object
                if isinstance(json_data, list):
                    wrapped_data = {"items": json_data}
                    json_chunks = splitter.split_json(json_data=wrapped_data)
                    # Unwrap the results if they contain the wrapper
                    for chunk in json_chunks:
                        if (
                            isinstance(chunk, dict)
                            and "items" in chunk
                            and len(chunk) == 1
                        ):
                            # If chunk only contains our wrapper, extract the items
                            unwrapped_chunk = chunk["items"]
                            chunk_str = self._process_json_chunk(unwrapped_chunk)
                        else:
                            chunk_str = self._process_json_chunk(chunk)
                        results.append(
                            self._create_document_from_chunk(chunk_str, doc.metadata)
                        )
                else:
                    json_chunks = splitter.split_json(json_data=json_data)
                    chunk_strings = [self._process_json_chunk(chunk) for chunk in json_chunks]
                    results.extend(self._chunks_to_documents(chunk_strings, doc.metadata))
            except (json.JSONDecodeError, Exception):
                # Fallback to text splitting
                text_chunks = splitter.split_text(doc.page_content)
                results.extend(self._chunks_to_documents(text_chunks, doc.metadata))
        else:
            text_chunks = splitter.split_text(doc.page_content)
            results.extend(self._chunks_to_documents(text_chunks, doc.metadata))
        return results

    def get_recursive_character_splitter(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
        length_function: Callable[[str], int] = len,
        is_separator_regex: bool = False,
    ) -> RecursiveCharacterTextSplitter:
        """
        Get RecursiveCharacterTextSplitter - recommended for generic text.

        Splits text recursively using list of characters, trying to keep paragraphs,
        sentences, and words together for better semantic coherence.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of characters to split on (default: ["\n\n", "\n", " ", ""])
            length_function: Function to determine chunk size
            is_separator_regex: Whether separators are regex patterns
        """
        return self._get_cached_splitter(
            "recursive_char",
            RecursiveCharacterTextSplitter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""],
            length_function=length_function,
            is_separator_regex=is_separator_regex,
        )

    def get_character_splitter(
        self,
        separator: str = "\n\n",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        length_function: Callable[[str], int] = len,
        is_separator_regex: bool = False,
    ) -> CharacterTextSplitter:
        """
        Get CharacterTextSplitter for simple character-based splitting.

        Args:
            separator: Character sequence to split on
            chunk_size: Maximum length of each chunk
            chunk_overlap: Overlap between chunks
            length_function: Function to calculate text length
            is_separator_regex: Whether separator is regex
        """
        return self._get_cached_splitter(
            "char",
            CharacterTextSplitter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            length_function=length_function,
            is_separator_regex=is_separator_regex,
        )

    def get_markdown_header_splitter(
        self,
        headers_to_split_on: List[tuple] = DEFAULT_MARKDOWN_HEADERS,
        strip_headers: bool = True,
        return_each_line: bool = False,
    ) -> MarkdownHeaderTextSplitter:
        """
        Get MarkdownHeaderTextSplitter for markdown document structure splitting.

        Splits markdown by headers while preserving document hierarchy metadata.

        Args:
            headers_to_split_on: List of (header_level, header_name) tuples
            strip_headers: Whether to remove headers from content
            return_each_line: Whether to return each line as separate document
        """
        return self._get_cached_splitter(
            "md_header",
            MarkdownHeaderTextSplitter,
            use_chunks=False,
            headers_to_split_on=headers_to_split_on,
            strip_headers=strip_headers,
            return_each_line=return_each_line,
        )

    def get_recursive_json_splitter(
        self, max_chunk_size: Optional[int] = None, min_chunk_size: Optional[int] = None
    ) -> RecursiveJsonSplitter:
        """
        Get RecursiveJsonSplitter for JSON data splitting.

        Splits JSON while controlling chunk sizes and preserving object structure.

        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
        """
        kwargs = {"max_chunk_size": max_chunk_size or self.default_chunk_size}
        if min_chunk_size is not None:
            kwargs["min_chunk_size"] = min_chunk_size
            
        return self._get_cached_splitter(
            "json",
            RecursiveJsonSplitter,
            use_chunks=False,
            **kwargs
        )

    def get_html_header_splitter(
        self, headers_to_split_on: List[tuple] = DEFAULT_HTML_HEADERS
    ) -> HTMLHeaderTextSplitter:
        """
        Get HTMLHeaderTextSplitter for HTML document structure splitting.

        Splits HTML based on header tags while preserving hierarchical structure.

        Args:
            headers_to_split_on: List of (header_tag, header_name) tuples
        """
        return self._get_cached_splitter(
            "html_header",
            HTMLHeaderTextSplitter,
            use_chunks=False,
            headers_to_split_on=headers_to_split_on
        )

    def get_html_section_splitter(
        self,
        headers_to_split_on: List[tuple] = DEFAULT_HTML_SECTION_HEADERS,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> HTMLSectionSplitter:
        """
        Get HTMLSectionSplitter for HTML section-based splitting.

        Uses XSLT transformations to detect sections and splits large ones.

        Args:
            headers_to_split_on: List of (header_tag, header_name) tuples
            chunk_size: Maximum chunk size for large sections
            chunk_overlap: Overlap for large section splitting
        """
        return self._get_cached_splitter(
            "html_section",
            HTMLSectionSplitter,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            headers_to_split_on=headers_to_split_on,
        )

    def get_html_semantic_splitter(
        self,
        headers_to_split_on: List[tuple] = DEFAULT_HTML_SEMANTIC_HEADERS,
        max_chunk_size: Optional[int] = None,
        elements_to_preserve: List[str] = DEFAULT_HTML_PRESERVE_ELEMENTS,
    ) -> HTMLSemanticPreservingSplitter:
        """
        Get HTMLSemanticPreservingSplitter for semantic HTML element preservation.

        Preserves structured elements like tables and lists while splitting.

        Args:
            headers_to_split_on: Headers for document structure
            max_chunk_size: Maximum target chunk size
            elements_to_preserve: HTML elements to keep intact
        """
        return self._get_cached_splitter(
            "html_semantic",
            HTMLSemanticPreservingSplitter,
            use_chunks=False,
            headers_to_split_on=headers_to_split_on,
            max_chunk_size=max_chunk_size or self.default_chunk_size,
            elements_to_preserve=elements_to_preserve,
        )

    def get_semantic_splitter(
        self,
        embeddings: Optional[Embeddings] = None,
        breakpoint_threshold_type: Literal[
            "percentile", "standard_deviation", "interquartile", "gradient"
        ] = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        min_chunk_size: int = 10,
        _threshold_defaults: dict = {
            "percentile": 95.0,
            "standard_deviation": 3.0,
            "interquartile": 1.5,
            "gradient": 0.5,
        },
    ):
        """
        Get SemanticChunker for embedding-based semantic splitting.

        Uses embedding models to determine chunk boundaries based on semantic similarity.
        Note: Requires langchain-experimental package for SemanticChunker.

        Args:
            embeddings: Embedding model for semantic analysis (required if not provided)
            breakpoint_threshold_type: Method for determining split points
            breakpoint_threshold_amount: Threshold value for splitting (uses type-specific default if None)
            min_chunk_size: Minimum chunk size
        """
        if embeddings is None:
            raise ValueError(
                "embeddings parameter is required for semantic splitting. Use get_embeddings() from core.dependencies."
            )

        # Use default threshold amount if not provided
        if breakpoint_threshold_amount is None:
            breakpoint_threshold_amount = _threshold_defaults[breakpoint_threshold_type]

        return self._get_cached_splitter(
            "semantic",
            SemanticChunker,
            use_chunks=False,
            embeddings=embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
            min_chunk_size=min_chunk_size,
        )

    def get_tiktoken_splitter(
        self,
        encoding_name: str = "cl100k_base",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separator: str = "\n\n",
        is_separator_regex: bool = False,
    ) -> CharacterTextSplitter:
        """
        Get Tiktoken-based text splitter for OpenAI models.

        Uses OpenAI's tiktoken tokenizer for accurate token counting.

        Args:
            encoding_name: Tiktoken encoding name
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
            separator: Text separator
            is_separator_regex: Whether separator is regex
        """
        final_chunk_size, final_chunk_overlap = self._validate_chunk_params(chunk_size, chunk_overlap)
        cache_key = self._generate_cache_key(
            "tiktoken", chunk_size=final_chunk_size, chunk_overlap=final_chunk_overlap,
            encoding_name=encoding_name, separator=separator, is_separator_regex=is_separator_regex
        )

        return self._get_cached_splitter(
            cache_key,
            lambda: CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=encoding_name,
                chunk_size=final_chunk_size,
                chunk_overlap=final_chunk_overlap,
                separator=separator,
                is_separator_regex=is_separator_regex,
            ),
            use_chunks=False  # Tell the method this is a factory function pattern
        )

    def get_huggingface_splitter(
        self,
        tokenizer,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separator: str = "\n\n",
        is_separator_regex: bool = False,
    ) -> CharacterTextSplitter:
        """
        Get HuggingFace tokenizer-based text splitter.

        Args:
            tokenizer: HuggingFace tokenizer instance
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
            separator: Text separator
            is_separator_regex: Whether separator is regex
        """
        final_chunk_size, final_chunk_overlap = self._validate_chunk_params(chunk_size, chunk_overlap)
        return CharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=final_chunk_size,
            chunk_overlap=final_chunk_overlap,
            separator=separator,
            is_separator_regex=is_separator_regex,
        )

    def get_spacy_splitter(
        self,
        pipeline: str = "en_core_web_sm",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> SpacyTextSplitter:
        """
        Get SpaCy-based text splitter for natural language processing.

        Args:
            pipeline: SpaCy pipeline name
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        return self._get_cached_splitter(
            "spacy", SpacyTextSplitter, chunk_size=chunk_size, chunk_overlap=chunk_overlap, pipeline=pipeline
        )

    def get_nltk_splitter(
        self,
        separator: str = "\n\n",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> NLTKTextSplitter:
        """
        Get NLTK-based text splitter for natural language processing.

        Args:
            separator: Text separator
            chunk_size: Maximum characters per chunk
            chunk_overlap: Character overlap between chunks
        """
        return self._get_cached_splitter(
            "nltk", NLTKTextSplitter, chunk_size=chunk_size, chunk_overlap=chunk_overlap, separator=separator
        )

    def get_sentence_transformers_splitter(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        tokens_per_chunk: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> SentenceTransformersTokenTextSplitter:
        """
        Get SentenceTransformers token-based text splitter.

        Args:
            model_name: SentenceTransformers model name
            tokens_per_chunk: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
        """
        final_tokens_per_chunk = tokens_per_chunk or (self.default_chunk_size // 4)
        final_chunk_overlap = chunk_overlap or (self.default_chunk_overlap // 4)
        
        return self._get_cached_splitter(
            "sentence_transformers",
            SentenceTransformersTokenTextSplitter,
            use_chunks=False,
            model_name=model_name,
            tokens_per_chunk=final_tokens_per_chunk,
            chunk_overlap=final_chunk_overlap,
        )

    def split_documents(
        self,
        documents: List[Union[Document, str]],
        splitter_type: SplitterType = SplitterType.RECURSIVE_CHARACTER,
        **kwargs,
    ) -> List[Document]:
        """
        Split documents using specified splitter type.

        Args:
            documents: List of documents or strings to split
            splitter_type: Type of splitter to use
            **kwargs: Additional arguments for specific splitter
        """
        # Get appropriate splitter
        splitter_map = {
            SplitterType.RECURSIVE_CHARACTER: self.get_recursive_character_splitter,
            SplitterType.CHARACTER: self.get_character_splitter,
            SplitterType.MARKDOWN_HEADER: self.get_markdown_header_splitter,
            SplitterType.RECURSIVE_JSON: self.get_recursive_json_splitter,
            SplitterType.HTML_HEADER: self.get_html_header_splitter,
            SplitterType.HTML_SECTION: self.get_html_section_splitter,
            SplitterType.HTML_SEMANTIC: self.get_html_semantic_splitter,
            SplitterType.SEMANTIC: self.get_semantic_splitter,
            SplitterType.TIKTOKEN: self.get_tiktoken_splitter,
            SplitterType.SPACY: self.get_spacy_splitter,
            SplitterType.NLTK: self.get_nltk_splitter,
            SplitterType.SENTENCE_TRANSFORMERS: self.get_sentence_transformers_splitter,
        }

        if splitter_type not in splitter_map:
            raise ValueError(f"Unsupported splitter type: {splitter_type}")

        splitter: TextSplitter = splitter_map[splitter_type](**kwargs)

        # Convert strings to Documents if needed
        doc_list = self._convert_to_documents(documents)

        # Use split_documents if available, otherwise use strategy pattern
        if hasattr(splitter, "split_documents"):
            return splitter.split_documents(doc_list)
        else:
            strategy = self._get_document_processing_strategy(splitter_type)
            results = []
            for doc in doc_list:
                results.extend(strategy(splitter, doc))
            return results

    def split_text(
        self,
        text: str,
        splitter_type: SplitterType = SplitterType.RECURSIVE_CHARACTER,
        **kwargs,
    ) -> List[str]:
        """
        Split text using specified splitter type.

        Args:
            text: Text to split
            splitter_type: Type of splitter to use
            **kwargs: Additional arguments for specific splitter
        """
        documents = self.split_documents([text], splitter_type, **kwargs)
        return [doc.page_content for doc in documents]

    def get_splitter_info(self) -> Dict[str, Any]:
        """Get information about available splitters and their configurations."""
        return {
            "available_splitters": [splitter.value for splitter in SplitterType],
            "default_chunk_size": self.default_chunk_size,
            "default_chunk_overlap": self.default_chunk_overlap,
            "cached_splitters": len(self._splitters_cache),
            "descriptions": {
                "recursive_character": "Recommended for generic text, splits recursively preserving semantics",
                "character": "Simple character-based splitting with configurable separator",
                "markdown_header": "Splits markdown by headers preserving document structure",
                "recursive_json": "JSON-aware splitting maintaining object integrity",
                "html_header": "HTML splitting based on header tags",
                "html_section": "HTML section-based splitting with XSLT detection",
                "html_semantic": "HTML splitting preserving semantic elements like tables and lists",
                "semantic": "Embedding-based splitting using semantic similarity",
                "tiktoken": "OpenAI tokenizer-based splitting for accurate token counting",
                "huggingface": "HuggingFace tokenizer integration",
                "spacy": "SpaCy NLP pipeline-based splitting",
                "nltk": "NLTK natural language processing splitting",
                "sentence_transformers": "SentenceTransformers model token splitting",
            },
        }

    def clear_cache(self):
        """Clear the splitter cache to free memory."""
        self._splitters_cache.clear()


# Convenience functions for common use cases
def create_text_splitter_manager(
    chunk_size: int = 1000, chunk_overlap: int = 200
) -> TextSplitterManager:
    """Create a new TextSplitterManager with specified defaults."""
    return TextSplitterManager(chunk_size, chunk_overlap)


def _create_manager_and_execute(operation: str, data, splitter_type=SplitterType.RECURSIVE_CHARACTER, chunk_size=1000, chunk_overlap=200, **kwargs):
    """Common function to create manager and execute text/document splitting operations."""
    return getattr(TextSplitterManager(chunk_size, chunk_overlap), operation)(data, splitter_type, **kwargs)

def split_text(text: str, splitter_type=SplitterType.RECURSIVE_CHARACTER, chunk_size=1000, chunk_overlap=200, **kwargs) -> List[str]:
    """Quick function to split text using any splitter type."""
    return _create_manager_and_execute("split_text", text, splitter_type, chunk_size, chunk_overlap, **kwargs)

def split_documents(documents: List[Union[Document, str]], splitter_type=SplitterType.RECURSIVE_CHARACTER, chunk_size=1000, chunk_overlap=200, **kwargs) -> List[Document]:
    """Quick function to split documents using any splitter type."""
    return _create_manager_and_execute("split_documents", documents, splitter_type, chunk_size, chunk_overlap, **kwargs)


def split_json_to_dicts(
    json_data: Union[str, dict, list], max_chunk_size: int = 1000
) -> List[dict]:
    """Quick function to split JSON data and return as dict objects."""
    json_str = json.dumps(json_data, ensure_ascii=False) if isinstance(json_data, (dict, list)) else json_data

    try:
        documents: list[Document] = _create_manager_and_execute(
            "split_documents", [json_str], SplitterType.RECURSIVE_JSON, 
            chunk_size=1000, chunk_overlap=200, max_chunk_size=max_chunk_size
        )
        
        return [
            json.loads(doc.page_content) if doc.page_content.strip().startswith(('{', '['))
            else {"text": doc.page_content}
            for doc in documents
        ]
    except Exception:
        return [json_data if isinstance(json_data, dict) else {"data": json_data}]


# Default manager instance for easy access
default_text_splitter_manager = TextSplitterManager()
