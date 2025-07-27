"""Vector database client."""


class VectorStoreClient:
    """Vector store client for vector operations."""

    def __init__(self, url: str):
        self.url = url
        # TODO: Initialize qdrant-client when implementing RAG features
