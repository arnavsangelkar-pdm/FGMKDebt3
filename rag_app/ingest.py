"""
Document ingestion pipeline.
Handles PDF parsing, chunking, embedding, and indexing.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from openai import OpenAI

from config import settings
from models import IngestResponse
from utils.parsing import PDFParser
from utils.chunking import TokenAwareChunker
from store.faiss_store import FAISSStore
from store.sqlite_store import SQLiteStore
from utils.logging import log_timing


logger = logging.getLogger(__name__)


class DocumentIngester:
    """Document ingestion pipeline."""
    
    def __init__(self, openai_client: OpenAI):
        """
        Initialize the document ingester.
        
        Args:
            openai_client: OpenAI client for embeddings
        """
        self.openai_client = openai_client
        self.logger = logger
        
        # Initialize components
        self.parser = PDFParser()
        self.chunker = TokenAwareChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.faiss_store = FAISSStore(openai_client)
        self.sqlite_store = SQLiteStore()
    
    def ingest_document(self, pdf_path: Path, doc_id: str) -> IngestResponse:
        """
        Ingest a PDF document into the RAG system.
        
        Args:
            pdf_path: Path to the PDF file
            doc_id: Document identifier
            
        Returns:
            IngestResponse with processing results
        """
        start_time = time.time()
        self.logger.info(f"Starting document ingestion for {doc_id}, pdf_path={str(pdf_path)}")
        
        try:
            # Step 1: Parse PDF
            parse_start = time.time()
            pages = self.parser.parse_pdf(pdf_path, doc_id)
            parse_time = time.time() - parse_start
            log_timing(self.logger, "pdf_parsing", parse_time, doc_id=doc_id, pages_count=len(pages))
            
            if not pages:
                raise ValueError("No text extracted from PDF")
            
            # Step 2: Chunk text
            chunk_start = time.time()
            chunks = self.chunker.chunk_pages(pages, doc_id)
            chunk_time = time.time() - chunk_start
            log_timing(self.logger, "text_chunking", chunk_time, doc_id=doc_id, chunks_count=len(chunks))
            
            if not chunks:
                raise ValueError("No chunks created from pages")
            
            # Validate chunks
            if not self.chunker.validate_chunks(chunks):
                raise ValueError("Chunk validation failed")
            
            # Step 3: Save chunks snapshot
            self._save_chunks_snapshot(doc_id, chunks)
            
            # Step 4: Embed and index in FAISS
            faiss_start = time.time()
            self.faiss_store.upsert_chunks(doc_id, chunks)
            faiss_time = time.time() - faiss_start
            log_timing(self.logger, "faiss_indexing", faiss_time, doc_id=doc_id, vectors_count=len(chunks))
            
            # Step 5: Index in SQLite FTS5
            sqlite_start = time.time()
            self.sqlite_store.upsert_chunks(doc_id, chunks)
            sqlite_time = time.time() - sqlite_start
            log_timing(self.logger, "sqlite_indexing", sqlite_time, doc_id=doc_id, chunks_count=len(chunks))
            
            # Calculate total processing time
            total_time = time.time() - start_time
            
            self.logger.info(f"Document ingestion completed for {doc_id}, pages_count={len(pages)}, chunks_count={len(chunks)}, total_time={total_time}")
            
            return IngestResponse(
                doc_id=doc_id,
                pages_count=len(pages),
                chunks_count=len(chunks),
                processing_time=total_time,
                message=f"Successfully ingested {len(pages)} pages into {len(chunks)} chunks"
            )
            
        except Exception as e:
            self.logger.error(f"Document ingestion failed for {doc_id}: {str(e)}", exc_info=True)
            raise
    
    def _save_chunks_snapshot(self, doc_id: str, chunks: List) -> None:
        """
        Save chunks to a Parquet file for debugging.
        
        Args:
            doc_id: Document identifier
            chunks: List of chunks to save
        """
        try:
            # Convert chunks to DataFrame
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "token_count": chunk.token_count
                })
            
            df = pd.DataFrame(chunk_data)
            
            # Save to Parquet
            chunks_file = settings.paths["chunks"] / f"{doc_id}.parquet"
            df.to_parquet(chunks_file, index=False)
            
            self.logger.info(f"Saved chunks snapshot for {doc_id}, chunks_file={str(chunks_file)}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save chunks snapshot for {doc_id}: {str(e)}")
            # Don't raise - this is just for debugging
    
    def get_ingestion_stats(self, doc_id: str) -> Dict[str, Any]:
        """
        Get ingestion statistics for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            # Get FAISS stats
            faiss_stats = self.faiss_store.get_stats(doc_id)
            
            # Get SQLite stats
            sqlite_stats = self.sqlite_store.get_stats(doc_id)
            
            # Get chunks file info
            chunks_file = settings.paths["chunks"] / f"{doc_id}.parquet"
            chunks_size = chunks_file.stat().st_size / (1024 * 1024) if chunks_file.exists() else 0
            
            # Get PDF file info
            pdf_file = settings.paths["docs"] / f"{doc_id}.pdf"
            pdf_size = pdf_file.stat().st_size / (1024 * 1024) if pdf_file.exists() else 0
            
            return {
                "doc_id": doc_id,
                "faiss": faiss_stats,
                "sqlite": sqlite_stats,
                "chunks_file_size_mb": chunks_size,
                "pdf_file_size_mb": pdf_size,
                "chunk_size": settings.CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get ingestion stats for {doc_id}: {str(e)}", exc_info=True)
            return {"doc_id": doc_id, "error": str(e)}
