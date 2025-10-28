from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    FAISS_K: int = 20
    FTS_K: int = 20
    RERANK_CANDIDATES: int = 30
    RERANK_TOP_N: int = 5
    CONFIDENCE_THRESHOLD: float = 0.35

    DATA_DIR: str = "data"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB

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
    
    @property
    def docs_path(self):
        return Path(self.DATA_DIR) / "docs"

settings = Settings()  # loads from env
for p in settings.paths.values():
    p.mkdir(parents=True, exist_ok=True)
