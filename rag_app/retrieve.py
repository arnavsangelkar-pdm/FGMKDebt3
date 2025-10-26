"""
Hybrid retrieval implementation with Reciprocal Rank Fusion (RRF) and reranking.
Combines vector search (FAISS) and keyword search (SQLite FTS5) for better results.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder
from openai import OpenAI

from .config import settings
from .store.faiss_store import FAISSStore
from .store.sqlite_store import SQLiteStore


logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_reranker():
    log = logging.getLogger(__name__)
    log.info("Loading bge-reranker-base...")
    model = CrossEncoder("BAAI/bge-reranker-base")
    log.info("Loaded reranker")
    return model


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
        
        # Configuration
        self.faiss_k = settings.FAISS_K
        self.fts_k = settings.FTS_K
        self.rerank_candidates = settings.RERANK_CANDIDATES
        self.rerank_top_n = settings.RERANK_TOP_N
        self.confidence_threshold = settings.CONFIDENCE_THRESHOLD
    
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
        
        self.logger.info(f"Starting hybrid retrieval for {doc_id}, question={question}")
        
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
            
            self.logger.info(f"Hybrid retrieval completed for {doc_id}, faiss_results={len(faiss_results)}, fts_results={len(fts_results)}, rrf_results={len(rrf_results)}, final_results={len(final_results)}")
            
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
                model=settings.OPENAI_EMBEDDING_MODEL,
                input=question
            )
            
            embedding = np.array(response.data[0].embedding, dtype='float32')
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
        
        self.logger.info(f"RRF completed, faiss_results={len(faiss_results)}, fts_results={len(fts_results)}, combined_results={len(sorted_results)}")
        
        return sorted_results
    
    def _rerank_candidates(self, question: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank candidates using BGE reranker with improved confidence scoring.
        
        Args:
            question: Query question
            candidates: List of candidate results
            
        Returns:
            Reranked results with confidence scores
        """
        if not candidates:
            return []
        
        try:
            # Get reranker (lazy loaded)
            reranker = get_reranker()
            
            # Prepare query-document pairs for reranking
            pairs = []
            for candidate in candidates:
                pairs.append([question, candidate["text"]])
            
            # Get reranking scores
            rerank_scores = reranker.predict(pairs)
            
            # Improved confidence scoring using sigmoid normalization
            # This preserves the relative differences better than min-max normalization
            import math
            
            # Apply sigmoid normalization to get better confidence distribution
            def sigmoid_normalize(score, scale=2.0):
                return 1 / (1 + math.exp(-scale * score))
            
            normalized_scores = [sigmoid_normalize(score) for score in rerank_scores]
            
            # Add scores to candidates and sort
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = float(rerank_scores[i])
                candidate["confidence"] = float(normalized_scores[i])
                
                # Add combined score that considers both RRF and rerank scores
                rrf_score = candidate.get("rrf_score", 0.0)
                combined_score = 0.7 * normalized_scores[i] + 0.3 * rrf_score
                candidate["combined_score"] = combined_score
            
            # Sort by combined score (descending) for better ranking
            reranked = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
            
            top_score = reranked[0]["rerank_score"] if reranked else 0
            top_confidence = reranked[0]["confidence"] if reranked else 0
            avg_conf = sum(c["confidence"] for c in reranked) / len(reranked) if reranked else 0
            self.logger.info(f"Reranking completed, candidates_count={len(candidates)}, top_score={top_score:.3f}, top_confidence={top_confidence:.3f}, avg_confidence={avg_conf:.3f}")
            
            return reranked
            
        except Exception as e:
            self.logger.error(f"Failed to rerank candidates: {str(e)}", exc_info=True)
            # Return original candidates without reranking but with basic confidence
            for i, candidate in enumerate(candidates):
                candidate["rerank_score"] = 0.0
                # Use RRF score as fallback confidence
                candidate["confidence"] = candidate.get("rrf_score", 0.0)
                candidate["combined_score"] = candidate["confidence"]
            return candidates
    
    def _apply_confidence_threshold(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply confidence threshold to filter results with improved logic.
        
        Args:
            results: List of reranked results
            
        Returns:
            Filtered results above confidence threshold
        """
        if not results:
            return []
        
        # Get the best result's confidence
        best_confidence = results[0].get("confidence", 0.0)
        best_combined_score = results[0].get("combined_score", 0.0)
        
        # More flexible thresholding: use either confidence or combined score
        # Also consider if we have at least one decent result
        meets_threshold = (best_confidence >= self.confidence_threshold or 
                          best_combined_score >= self.confidence_threshold)
        
        if not meets_threshold:
            # If no result meets threshold, but we have some results, 
            # return the top 2-3 results anyway (they might still be useful)
            if len(results) >= 2:
                self.logger.warning(f"Best result confidence {best_confidence:.3f} below threshold {self.confidence_threshold}, returning top 2 results")
                return results[:2]
            elif len(results) >= 1:
                self.logger.warning(f"Best result confidence {best_confidence:.3f} below threshold {self.confidence_threshold}, returning top result")
                return results[:1]
            else:
                self.logger.warning(f"Best result confidence {best_confidence:.3f} below threshold {self.confidence_threshold}, no results returned")
                return []
        
        # Return all results (they're already sorted by combined score)
        self.logger.info(f"Confidence threshold passed, returning {len(results)} results, best_confidence={best_confidence:.3f}")
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
