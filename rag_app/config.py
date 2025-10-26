"""
Configuration management using pydantic BaseSettings.
Handles environment variables and default values for the RAG application.
"""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL")
    
    # Chunking Configuration
    chunk_size: int = Field(default=500, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    
    # Retrieval Configuration
    faiss_k: int = Field(default=20, env="FAISS_K")
    fts_k: int = Field(default=20, env="FTS_K")
    rerank_top_n: int = Field(default=5, env="RERANK_TOP_N")
    rerank_candidates: int = Field(default=30, env="RERANK_CANDIDATES")
    confidence_threshold: float = Field(default=0.35, env="CONFIDENCE_THRESHOLD")
    
    # File System Configuration
    data_dir: str = Field(default="data", env="DATA_DIR")
    docs_dir: str = Field(default="data/docs", env="DOCS_DIR")
    indices_dir: str = Field(default="data/indices", env="INDICES_DIR")
    sqlite_dir: str = Field(default="data/sqlite", env="SQLITE_DIR")
    chunks_dir: str = Field(default="data/chunks", env="CHUNKS_DIR")
    
    # Security Configuration
    max_upload_size: int = Field(default=250 * 1024 * 1024, env="MAX_UPLOAD_SIZE")  # 250MB
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @validator("chunk_overlap")
    def validate_chunk_overlap(cls, v, values):
        """Ensure chunk overlap is less than chunk size."""
        chunk_size = values.get("chunk_size", 500)
        if v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v
    
    @validator("confidence_threshold")
    def validate_confidence_threshold(cls, v):
        """Ensure confidence threshold is between 0 and 1."""
        if not 0 <= v <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        return v
    
    def setup_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.docs_dir,
            self.indices_dir,
            self.sqlite_dir,
            self.chunks_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def docs_path(self) -> Path:
        """Get the docs directory path."""
        return Path(self.docs_dir)
    
    @property
    def indices_path(self) -> Path:
        """Get the indices directory path."""
        return Path(self.indices_dir)
    
    @property
    def sqlite_path(self) -> Path:
        """Get the sqlite directory path."""
        return Path(self.sqlite_dir)
    
    @property
    def chunks_path(self) -> Path:
        """Get the chunks directory path."""
        return Path(self.chunks_dir)


# Global settings instance
settings = Settings()

# Setup directories on import
settings.setup_directories()
