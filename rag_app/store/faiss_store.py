"""
FAISS vector store implementation for document embeddings.
Handles creation, loading, saving, and searching of vector indices.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
from openai import OpenAI

from config import settings
from utils.chunking import Chunk


logger = logging.getLogger(__name__)


class FAISSStore:
    """FAISS vector store for document embeddings."""
    
    def __init__(self, openai_client: OpenAI):
        """
        Initialize the FAISS store.
        
        Args:
            openai_client: OpenAI client for embeddings
        """
        self.openai_client = openai_client
        self.logger = logger
        self.embedding_model = settings.OPENAI_EMBEDDING_MODEL
        self.embedding_dim = 1536  # text-embedding-3-small dimension
        
        # Ensure indices directory exists
        settings.paths["indices"].mkdir(parents=True, exist_ok=True)
    
    def create_index(self, doc_id: str) -> faiss.IndexFlatIP:
        """
        Create a new FAISS index for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            New FAISS index
        """
        # Use Inner Product (cosine similarity) index
        index = faiss.IndexFlatIP(self.embedding_dim)
        self.logger.info(f"Created new FAISS index for {doc_id}")
        return index
    
    def load_index(self, doc_id: str) -> Optional[faiss.IndexFlatIP]:
        """
        Load an existing FAISS index for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            FAISS index if found, None otherwise
        """
        index_path = self._get_index_path(doc_id)
        meta_path = self._get_meta_path(doc_id)
        
        if not index_path.exists() or not meta_path.exists():
            return None
        
        try:
            index = faiss.read_index(str(index_path))
            self.logger.info(f"Loaded FAISS index for {doc_id}, vectors_count={index.ntotal}")
            return index
        except Exception as e:
            self.logger.error(f"Failed to load FAISS index for {doc_id}: {str(e)}", exc_info=True)
            return None
    
    def save_index(self, doc_id: str, index: faiss.IndexFlatIP, metadata: Dict[str, Any]) -> None:
        """
        Save a FAISS index and its metadata.
        
        Args:
            doc_id: Document identifier
            index: FAISS index to save
            metadata: Metadata mapping vector IDs to chunk information
        """
        index_path = self._get_index_path(doc_id)
        meta_path = self._get_meta_path(doc_id)
        
        try:
            # Save FAISS index
            faiss.write_index(index, str(index_path))
            
            # Save metadata
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            index_size_mb = index_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Saved FAISS index for {doc_id}, vectors_count={index.ntotal}, index_size_mb={index_size_mb}")
            
        except Exception as e:
            self.logger.error(f"Failed to save FAISS index for {doc_id}: {str(e)}", exc_info=True)
            raise
    
    def upsert_chunks(self, doc_id: str, chunks: List[Chunk]) -> None:
        """
        Upsert chunks into the FAISS index (replaces existing index).
        
        Args:
            doc_id: Document identifier
            chunks: List of chunks to embed and store
        """
        if not chunks:
            self.logger.warning(f"No chunks provided for {doc_id}")
            return
        
        self.logger.info(f"Starting FAISS upsert for {doc_id}, chunks_count={len(chunks)}")
        
        try:
            # Generate embeddings for all chunks
            embeddings = self._generate_embeddings(chunks)
            
            # Create new index
            index = self.create_index(doc_id)
            
            # Normalize embeddings for cosine similarity
            embeddings = embeddings.astype('float32')
            faiss.normalize_L2(embeddings)
            
            # Add vectors to index
            index.add(embeddings)
            
            # Create metadata mapping
            metadata = {}
            for i, chunk in enumerate(chunks):
                metadata[str(i)] = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "page": chunk.page,
                    "section": chunk.section,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "text": chunk.text,
                    "token_count": chunk.token_count
                }
            
            # Save index and metadata
            self.save_index(doc_id, index, metadata)
            
            self.logger.info(f"FAISS upsert completed for {doc_id}, vectors_count={len(chunks)}")
            
        except Exception as e:
            self.logger.error(f"Failed to upsert chunks for {doc_id}: {str(e)}", exc_info=True)
            raise
    
    def search(self, doc_id: str, query_embedding: np.ndarray, k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for similar chunks in the FAISS index.
        
        Args:
            doc_id: Document identifier
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        # Load index and metadata
        index = self.load_index(doc_id)
        if index is None:
            self.logger.warning(f"No FAISS index found for {doc_id}")
            return []
        
        metadata = self._load_metadata(doc_id)
        if not metadata:
            self.logger.warning(f"No metadata found for {doc_id}")
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search
            scores, indices = index.search(query_embedding, min(k, index.ntotal))
            
            # Build results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for empty slots
                    continue
                
                vector_id = str(idx)
                if vector_id in metadata:
                    result = metadata[vector_id].copy()
                    result["faiss_score"] = float(score)
                    result["vector_id"] = vector_id
                    results.append(result)
            
            self.logger.info(f"FAISS search completed for {doc_id}, query_k={k}, results_count={len(results)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search FAISS index for {doc_id}: {str(e)}", exc_info=True)
            return []
    
    def get_stats(self, doc_id: str) -> Dict[str, Any]:
        """
        Get statistics about the FAISS index for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary with index statistics
        """
        index_path = self._get_index_path(doc_id)
        meta_path = self._get_meta_path(doc_id)
        
        if not index_path.exists():
            return {"exists": False}
        
        try:
            index = self.load_index(doc_id)
            metadata = self._load_metadata(doc_id)
            
            return {
                "exists": True,
                "vectors_count": index.ntotal if index else 0,
                "index_size_mb": index_path.stat().st_size / (1024 * 1024),
                "metadata_size_mb": meta_path.stat().st_size / (1024 * 1024) if meta_path.exists() else 0,
                "chunks_count": len(metadata) if metadata else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get FAISS stats for {doc_id}: {str(e)}", exc_info=True)
            return {"exists": False, "error": str(e)}
    
    def _generate_embeddings(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Generate embeddings for a list of chunks.
        
        Args:
            chunks: List of chunks to embed
            
        Returns:
            Numpy array of embeddings
        """
        texts = [chunk.text for chunk in chunks]
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            
            embeddings = np.array([data.embedding for data in response.data], dtype='float32')
            self.logger.info(f"Generated {len(embeddings)} embeddings")
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {str(e)}", exc_info=True)
            raise
    
    def _load_metadata(self, doc_id: str) -> Dict[str, Any]:
        """
        Load metadata for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Metadata dictionary
        """
        meta_path = self._get_meta_path(doc_id)
        
        if not meta_path.exists():
            return {}
        
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata for {doc_id}: {str(e)}", exc_info=True)
            return {}
    
    def _get_index_path(self, doc_id: str) -> Path:
        """Get the path to the FAISS index file."""
        return settings.paths["indices"] / f"{doc_id}.faiss"
    
    def _get_meta_path(self, doc_id: str) -> Path:
        """Get the path to the metadata file."""
        return settings.paths["indices"] / f"{doc_id}.faiss.meta.json"
