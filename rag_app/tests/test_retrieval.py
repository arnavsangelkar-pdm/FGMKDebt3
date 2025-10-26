"""
Tests for retrieval functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from retrieve import HybridRetriever
from utils.chunking import Chunk


class TestHybridRetriever:
    """Test cases for HybridRetriever."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_openai_client = Mock()
        self.retriever = HybridRetriever(self.mock_openai_client)
    
    def test_reciprocal_rank_fusion(self):
        """Test Reciprocal Rank Fusion (RRF) functionality."""
        # Mock FAISS results
        faiss_results = [
            {"chunk_id": "chunk1", "text": "First result", "faiss_score": 0.9},
            {"chunk_id": "chunk2", "text": "Second result", "faiss_score": 0.8},
        ]
        
        # Mock FTS results
        fts_results = [
            {"chunk_id": "chunk2", "text": "Second result", "bm25_score": 0.7},
            {"chunk_id": "chunk3", "text": "Third result", "bm25_score": 0.6},
        ]
        
        # Test RRF
        rrf_results = self.retriever._reciprocal_rank_fusion(faiss_results, fts_results)
        
        assert len(rrf_results) == 3  # Should have 3 unique chunks
        
        # Check that chunk2 (appears in both) has higher score than chunk3 (only in FTS)
        chunk2_score = next(r["rrf_score"] for r in rrf_results if r["chunk_id"] == "chunk2")
        chunk3_score = next(r["rrf_score"] for r in rrf_results if r["chunk_id"] == "chunk3")
        
        assert chunk2_score > chunk3_score
    
    def test_rerank_candidates(self):
        """Test candidate reranking functionality."""
        # Mock candidates
        candidates = [
            {"chunk_id": "chunk1", "text": "This is about machine learning algorithms"},
            {"chunk_id": "chunk2", "text": "This is about cooking recipes"},
            {"chunk_id": "chunk3", "text": "This is about neural networks and deep learning"},
        ]
        
        question = "What are neural networks?"
        
        # Mock the reranker to return predictable scores
        with patch.object(self.retriever.reranker, 'predict') as mock_predict:
            # Mock scores: chunk3 should score highest (most relevant to neural networks)
            mock_predict.return_value = [0.3, 0.1, 0.9]  # chunk1, chunk2, chunk3
            
            reranked = self.retriever._rerank_candidates(question, candidates)
            
            assert len(reranked) == 3
            assert reranked[0]["chunk_id"] == "chunk3"  # Should be ranked first
            assert reranked[0]["rerank_score"] == 0.9
            assert reranked[0]["confidence"] == 1.0  # Highest score gets confidence 1.0
    
    def test_apply_confidence_threshold(self):
        """Test confidence threshold filtering."""
        # Mock results with different confidence scores
        results = [
            {"chunk_id": "chunk1", "confidence": 0.8, "text": "High confidence result"},
            {"chunk_id": "chunk2", "confidence": 0.4, "text": "Medium confidence result"},
            {"chunk_id": "chunk3", "confidence": 0.2, "text": "Low confidence result"},
        ]
        
        # Test with threshold 0.5
        self.retriever.confidence_threshold = 0.5
        filtered = self.retriever._apply_confidence_threshold(results)
        
        assert len(filtered) == 1
        assert filtered[0]["chunk_id"] == "chunk1"
        
        # Test with threshold 0.1 (should return all)
        self.retriever.confidence_threshold = 0.1
        filtered = self.retriever._apply_confidence_threshold(results)
        
        assert len(filtered) == 3
    
    def test_apply_confidence_threshold_empty(self):
        """Test confidence threshold with empty results."""
        results = []
        filtered = self.retriever._apply_confidence_threshold(results)
        assert len(filtered) == 0
    
    def test_generate_query_embedding(self):
        """Test query embedding generation."""
        question = "What is machine learning?"
        mock_embedding = np.random.rand(1536).astype(np.float32)
        
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = mock_embedding.tolist()
        
        self.mock_openai_client.embeddings.create.return_value = mock_response
        
        embedding = self.retriever._generate_query_embedding(question)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1536,)
        np.testing.assert_array_equal(embedding, mock_embedding)
    
    def test_retrieve_no_results(self):
        """Test retrieval when no results are found."""
        doc_id = "test_doc"
        question = "What is the meaning of life?"
        
        # Mock empty results from both stores
        with patch.object(self.retriever.faiss_store, 'search') as mock_faiss, \
             patch.object(self.retriever.sqlite_store, 'bm25_search') as mock_fts, \
             patch.object(self.retriever, '_generate_query_embedding') as mock_embedding:
            
            mock_embedding.return_value = np.random.rand(1536)
            mock_faiss.return_value = []
            mock_fts.return_value = []
            
            results = self.retriever.retrieve(doc_id, question)
            
            assert len(results) == 0
    
    def test_retrieve_with_results(self):
        """Test retrieval with actual results."""
        doc_id = "test_doc"
        question = "What is machine learning?"
        
        # Mock results
        faiss_results = [
            {"chunk_id": "chunk1", "text": "Machine learning is a subset of AI", "page": 1}
        ]
        fts_results = [
            {"chunk_id": "chunk2", "text": "Deep learning uses neural networks", "page": 2}
        ]
        
        with patch.object(self.retriever.faiss_store, 'search') as mock_faiss, \
             patch.object(self.retriever.sqlite_store, 'bm25_search') as mock_fts, \
             patch.object(self.retriever, '_generate_query_embedding') as mock_embedding, \
             patch.object(self.retriever, '_rerank_candidates') as mock_rerank:
            
            mock_embedding.return_value = np.random.rand(1536)
            mock_faiss.return_value = faiss_results
            mock_fts.return_value = fts_results
            
            # Mock reranking to return results with confidence above threshold
            reranked_results = [
                {"chunk_id": "chunk1", "text": "Machine learning is a subset of AI", "confidence": 0.8}
            ]
            mock_rerank.return_value = reranked_results
            
            results = self.retriever.retrieve(doc_id, question)
            
            assert len(results) == 1
            assert results[0]["chunk_id"] == "chunk1"
            assert results[0]["confidence"] == 0.8
    
    def test_get_retrieval_stats(self):
        """Test retrieval statistics gathering."""
        doc_id = "test_doc"
        
        # Mock stats from stores
        mock_faiss_stats = {"exists": True, "vectors_count": 100}
        mock_sqlite_stats = {"exists": True, "chunks_count": 100}
        
        with patch.object(self.retriever.faiss_store, 'get_stats') as mock_faiss, \
             patch.object(self.retriever.sqlite_store, 'get_stats') as mock_sqlite:
            
            mock_faiss.return_value = mock_faiss_stats
            mock_sqlite.return_value = mock_sqlite_stats
            
            stats = self.retriever.get_retrieval_stats(doc_id)
            
            assert stats["doc_id"] == doc_id
            assert stats["faiss"] == mock_faiss_stats
            assert stats["sqlite"] == mock_sqlite_stats
            assert stats["reranker_model"] == "BAAI/bge-reranker-base"
            assert stats["confidence_threshold"] == self.retriever.confidence_threshold
