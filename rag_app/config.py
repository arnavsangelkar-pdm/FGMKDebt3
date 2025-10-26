from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    CHUNK_SIZE: int = 800  # Increased for better context
    CHUNK_OVERLAP: int = 100  # Increased overlap for better continuity
    FAISS_K: int = 30  # More candidates from vector search
    FTS_K: int = 30  # More candidates from keyword search
    RERANK_CANDIDATES: int = 50  # More candidates for reranking
    RERANK_TOP_N: int = 8  # More final results
    CONFIDENCE_THRESHOLD: float = 0.2  # Lower threshold for more results

    DATA_DIR: str = "data"

    @property
    def paths(self):
        root = Path(self.DATA_DIR)
        return {
            "data": root,
            "docs": root / "docs",
            "indices": root / "indices",
            "sqlite": root / "sqlite",
            "chunks": root / "chunks",
        }

settings = Settings()  # loads from env
for p in settings.paths.values():
    p.mkdir(parents=True, exist_ok=True)
