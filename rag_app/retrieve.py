"""
Hybrid retrieval implementation with Reciprocal Rank Fusion (RRF) and reranking.
Combines vector search (FAISS) and keyword search (SQLite FTS5) for better results.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import numpy as np
from sentence_transformers import CrossEncoder
from openai import OpenAI

from config import settings
from store.faiss_store import FAISSStore
from store.sqlite_store import SQLiteStore


logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining vector and keyword search with reranking."""
    
    def __init__(self, openai_client: OpenAI):
        """
        Initialize the hybrid retriever.
        
        Args:
            openai_client: OpenAI client for embeddings
        """
        self.openai_client = openai_client
        self.logger = logger
        
        # Initialize stores
        self.faiss_store = FAISSStore(openai_client)
        self.sqlite_store = SQLiteStore()
        
        # Initialize reranker
        self.reranker = CrossEncoder('BAAI/bge-reranker-base')
        
        # Configuration
        self.faiss_k = settings.faiss_k
        self.fts_k = settings.fts_k
        self.rerank_candidates = settings.rerank_candidates
        self.rerank_top_n = settings.rerank_top_n
        self.confidence_threshold = settings.confidence_threshold
    
    def retrieve(self, doc_id: str, question: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Perform hybrid retrieval with reranking.
        
        Args:
            doc_id: Document identifier
            question: Query question
            k: Number of final results to return (overrides config)
            
        Returns:
            List of retrieved and reranked results
        """
        if k is None:
            k = self.rerank_top_n
        
        self.logger.info(f"Starting hybrid retrieval for {doc_id}", doc_id=doc_id, question=question)
        
        try:
            # Step 1: Generate query embedding
            query_embedding = self._generate_query_embedding(question)
            
            # Step 2: Vector search (FAISS)
            faiss_results = self.faiss_store.search(doc_id, query_embedding, self.faiss_k)
            
            # Step 3: Keyword search (SQLite FTS5)
            fts_results = self.sqlite_store.bm25_search(doc_id, question, self.fts_k)
            
            # Step 4: Reciprocal Rank Fusion
            rrf_results = self._reciprocal_rank_fusion(faiss_results, fts_results)
            
            # Step 5: Rerank top candidates
            reranked_results = self._rerank_candidates(question, rrf_results[:self.rerank_candidates])
            
            # Step 6: Apply confidence threshold
            final_results = self._apply_confidence_threshold(reranked_results[:k])
            
            self.logger.info(
                f"Hybrid retrieval completed for {doc_id}",
                doc_id=doc_id,
                faiss_results=len(faiss_results),
                fts_results=len(fts_results),
                rrf_results=len(rrf_results),
                final_results=len(final_results)
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Failed hybrid retrieval for {doc_id}: {str(e)}", exc_info=True)
            return []
    
    def _generate_query_embedding(self, question: str) -> np.ndarray:
        """
        Generate embedding for the query.
        
        Args:
            question: Query question
            
        Returns:
            Query embedding vector
        """
        try:
            response = self.openai_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=question
            )
            
            embedding = np.array(response.data[0].embedding)
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to generate query embedding: {str(e)}", exc_info=True)
            raise
    
    def _reciprocal_rank_fusion(self, faiss_results: List[Dict], fts_results: List[Dict], k: int = 60) -> List[Dict[str, Any]]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        Args:
            faiss_results: Results from FAISS vector search
            fts_results: Results from SQLite FTS5 search
            k: RRF parameter (higher = more dampening)
            
        Returns:
            Combined and ranked results
        """
        # Create a mapping of chunk_id to combined score
        combined_scores = defaultdict(float)
        result_map = {}
        
        # Process FAISS results
        for rank, result in enumerate(faiss_results):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            combined_scores[chunk_id] += rrf_score
            result_map[chunk_id] = result.copy()
            result_map[chunk_id]["faiss_rank"] = rank
            result_map[chunk_id]["faiss_rrf_score"] = rrf_score
        
        # Process FTS results
        for rank, result in enumerate(fts_results):
            chunk_id = result["chunk_id"]
            rrf_score = 1.0 / (k + rank + 1)
            combined_scores[chunk_id] += rrf_score
            if chunk_id in result_map:
                result_map[chunk_id]["fts_rank"] = rank
                result_map[chunk_id]["fts_rrf_score"] = rrf_score
                result_map[chunk_id]["bm25_score"] = result["bm25_score"]
            else:
                result_map[chunk_id] = result.copy()
                result_map[chunk_id]["fts_rank"] = rank
                result_map[chunk_id]["fts_rrf_score"] = rrf_score
                result_map[chunk_id]["faiss_rank"] = None
                result_map[chunk_id]["faiss_rrf_score"] = 0.0
        
        # Sort by combined RRF score
        sorted_results = []
        for chunk_id, score in sorted(combined_scores.items(), key=lambda x: x[1], reverse=True):
            result = result_map[chunk_id]
            result["rrf_score"] = score
            sorted_results.append(result)
        
        self.logger.info(
            f"RRF completed",
            faiss_results=len(faiss_results),
            fts_results=len(fts_results),
            combined_results=len(sorted_results)
        )
        
        return sorted_results
    
    def _rerank_candidates(self, question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank candidates using BGE reranker.
        
        Args:
            question: Query question
            candidates: List of candidate results
            
        Returns:
            Reranked results with confidence scores
        """
        if not candidates:
            return []
        
        try:
            # Prepare query-document pairs for reranking
            pairs = []
            for candidate in candidates:
                pairs.append([question, candidate["text"]])
            
            # Get reranking scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Normalize scores to [0, 1] range
            min_score = min(rerank_scores)
            max_score = max(rerank_scores)
            if max_score > min_score:
                normalized_scores = [(score - min_score) / (max_score - min_score) for score in rerank_scores]
            else:
                normalized_scores = [1.0] * len(rerank_scores)
            
            # Add scores to candidates and sort
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = float(rerank_scores[i])
                candidate["confidence"] = float(normalized_scores[i])
            
            # Sort by rerank score (descending)
            reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            
            self.logger.info(
                f"Reranking completed",
                candidates_count=len(candidates),
                top_score=reranked[0]["rerank_score"] if reranked else 0,
                avg_confidence=sum(c["confidence"] for c in reranked) / len(reranked) if reranked else 0
            )
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"Failed to rerank candidates: {str(e)}", exc_info=True)
            # Return original candidates without reranking
            for candidate in candidates:
                candidate["rerank_score"] = 0.0
                candidate["confidence"] = 0.0
            return candidates
    
    def _apply_confidence_threshold(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply confidence threshold to filter results.
        
        Args:
            results: List of reranked results
            
        Returns:
            Filtered results above confidence threshold
        """
        if not results:
            return []
        
        # Check if the best result meets confidence threshold
        best_confidence = results[0].get("confidence", 0.0)
        
        if best_confidence < self.confidence_threshold:
            self.logger.warning(
                f"Best result confidence {best_confidence:.3f} below threshold {self.confidence_threshold}",
                best_confidence=best_confidence,
                threshold=self.confidence_threshold
            )
            return []  # Return empty results if confidence is too low
        
        # Return all results (they're already sorted by confidence)
        return results
    
    def get_retrieval_stats(self, doc_id: str) -> Dict[str, Any]:
        """
        Get retrieval statistics for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary with retrieval statistics
        """
        faiss_stats = self.faiss_store.get_stats(doc_id)
        sqlite_stats = self.sqlite_store.get_stats(doc_id)
        
        return {
            "doc_id": doc_id,
            "faiss": faiss_stats,
            "sqlite": sqlite_stats,
            "reranker_model": "BAAI/bge-reranker-base",
            "confidence_threshold": self.confidence_threshold
        }
