"""
SQLite FTS5 store implementation for keyword search.
Handles creation, loading, and searching of full-text search indices.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import sqlite3
from sqlite3 import Connection

from ..config import settings
from ..utils.chunking import Chunk


logger = logging.getLogger(__name__)


class SQLiteStore:
    """SQLite FTS5 store for keyword search."""
    
    def __init__(self):
        """Initialize the SQLite store."""
        self.logger = logger
        
        # Ensure sqlite directory exists
        settings.paths["sqlite"].mkdir(parents=True, exist_ok=True)
    
    def create_database(self, doc_id: str) -> Connection:
        """
        Create a new SQLite database with FTS5 for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            SQLite connection
        """
        db_path = self._get_db_path(doc_id)
        
        # Remove existing database if it exists
        if db_path.exists():
            db_path.unlink()
        
        conn = sqlite3.connect(str(db_path))
        
        try:
            # Create chunks table
            conn.execute("""
                CREATE TABLE chunks (
                    id INTEGER PRIMARY KEY,
                    page INTEGER NOT NULL,
                    section TEXT,
                    text TEXT NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    chunk_id TEXT NOT NULL UNIQUE,
                    token_count INTEGER NOT NULL
                )
            """)
            
            # Create FTS5 virtual table
            conn.execute("""
                CREATE VIRTUAL TABLE chunks_fts USING fts5(
                    text,
                    content='chunks',
                    content_rowid='id'
                )
            """)
            
            # Create triggers to keep FTS5 in sync with chunks table
            conn.execute("""
                CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                END
            """)
            
            conn.execute("""
                CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
                END
            """)
            
            conn.commit()
            self.logger.info(f"Created SQLite database for {doc_id}, db_path={str(db_path)}")
            
        except Exception as e:
            conn.close()
            self.logger.error(f"Failed to create SQLite database for {doc_id}: {str(e)}", exc_info=True)
            raise
        
        return conn
    
    def load_database(self, doc_id: str) -> Optional[Connection]:
        """
        Load an existing SQLite database for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            SQLite connection if found, None otherwise
        """
        db_path = self._get_db_path(doc_id)
        
        if not db_path.exists():
            return None
        
        try:
            conn = sqlite3.connect(str(db_path))
            
            # Verify the database has the required tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name IN ('chunks', 'chunks_fts')")
            tables = [row[0] for row in cursor.fetchall()]
            
            if len(tables) != 2:
                self.logger.warning(f"SQLite database for {doc_id} is missing required tables")
                conn.close()
                return None
            
            # Get chunk count
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            chunk_count = cursor.fetchone()[0]
            
            self.logger.info(f"Loaded SQLite database for {doc_id}, chunks_count={chunk_count}")
            return conn
            
        except Exception as e:
            self.logger.error(f"Failed to load SQLite database for {doc_id}: {str(e)}", exc_info=True)
            return None
    
    def upsert_chunks(self, doc_id: str, chunks: List[Chunk]) -> None:
        """
        Upsert chunks into the SQLite database (replaces existing data).
        
        Args:
            doc_id: Document identifier
            chunks: List of chunks to store
        """
        if not chunks:
            self.logger.warning(f"No chunks provided for {doc_id}")
            return
        
        self.logger.info(f"Starting SQLite upsert for {doc_id}, chunks_count={len(chunks)}")
        
        # Create new database (this replaces any existing one)
        conn = self.create_database(doc_id)
        
        try:
            # Insert chunks
            cursor = conn.cursor()
            
            for chunk in chunks:
                cursor.execute("""
                    INSERT INTO chunks (page, section, text, char_start, char_end, chunk_id, token_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    chunk.page,
                    chunk.section,
                    chunk.text,
                    chunk.char_start,
                    chunk.char_end,
                    chunk.chunk_id,
                    chunk.token_count
                ))
            
            conn.commit()
            
            # Verify FTS5 is populated
            cursor = conn.execute("SELECT COUNT(*) FROM chunks_fts")
            fts_count = cursor.fetchone()[0]
            
            self.logger.info(
                f"SQLite upsert completed for {doc_id}, chunks_count={len(chunks)}, fts_count={fts_count}"
            )
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to upsert chunks for {doc_id}: {str(e)}", exc_info=True)
            raise
        finally:
            conn.close()
    
    def bm25_search(self, doc_id: str, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for chunks using BM25 scoring.
        
        Args:
            doc_id: Document identifier
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        conn = self.load_database(doc_id)
        if conn is None:
            self.logger.warning(f"No SQLite database found for {doc_id}")
            return []
        
        try:
            # Use FTS5 bm25() function for scoring
            cursor = conn.execute("""
                SELECT 
                    c.id,
                    c.page,
                    c.section,
                    c.text,
                    c.char_start,
                    c.char_end,
                    c.chunk_id,
                    c.token_count,
                    bm25(chunks_fts) as bm25_score
                FROM chunks_fts
                JOIN chunks c ON chunks_fts.rowid = c.id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """, (query, k))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    "id": row[0],
                    "page": row[1],
                    "section": row[2],
                    "text": row[3],
                    "char_start": row[4],
                    "char_end": row[5],
                    "chunk_id": row[6],
                    "token_count": row[7],
                    "bm25_score": float(row[8])
                }
                results.append(result)
            
            self.logger.info(
                f"BM25 search completed for {doc_id}",
                doc_id=doc_id,
                query=query,
                query_k=k,
                results_count=len(results)
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search SQLite database for {doc_id}: {str(e)}", exc_info=True)
            return []
        finally:
            conn.close()
    
    def get_stats(self, doc_id: str) -> Dict[str, Any]:
        """
        Get statistics about the SQLite database for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary with database statistics
        """
        db_path = self._get_db_path(doc_id)
        
        if not db_path.exists():
            return {"exists": False}
        
        conn = self.load_database(doc_id)
        if conn is None:
            return {"exists": False, "error": "Failed to load database"}
        
        try:
            # Get chunk count
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            chunks_count = cursor.fetchone()[0]
            
            # Get FTS5 count
            cursor = conn.execute("SELECT COUNT(*) FROM chunks_fts")
            fts_count = cursor.fetchone()[0]
            
            # Get page count
            cursor = conn.execute("SELECT COUNT(DISTINCT page) FROM chunks")
            pages_count = cursor.fetchone()[0]
            
            # Get total characters
            cursor = conn.execute("SELECT SUM(LENGTH(text)) FROM chunks")
            total_chars = cursor.fetchone()[0] or 0
            
            return {
                "exists": True,
                "chunks_count": chunks_count,
                "fts_count": fts_count,
                "pages_count": pages_count,
                "total_characters": total_chars,
                "db_size_mb": db_path.stat().st_size / (1024 * 1024)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get SQLite stats for {doc_id}: {str(e)}", exc_info=True)
            return {"exists": False, "error": str(e)}
        finally:
            conn.close()
    
    def _get_db_path(self, doc_id: str) -> Path:
        """Get the path to the SQLite database file."""
        return settings.paths["sqlite"] / f"{doc_id}.db"
