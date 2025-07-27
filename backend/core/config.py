"""Application configuration settings."""

from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database URL for Prisma
    database_url: str

    # LLM Configuration
    llm_api_key: str
    llm_model: str

    # Vector Store
    vector_store_url: str

    # Storage
    storage_endpoint: str
    storage_access_key: str
    storage_secret_key: str
    storage_bucket: str
    storage_secure: bool

    # Application
    debug: bool
    log_level: str

    # CORS
    cors_origins: List[str]

    # File upload
    max_file_size: int
    allowed_file_types: List[str]

    # API
    api_host: str
    api_port: int

    class Config:
        env_file = Path(__file__).parent.parent / ".env"  # backend/.env
        env_file_encoding = "utf-8"
        extra = "ignore"  # Allow extra environment variables


# Global settings instance
settings = Settings()
