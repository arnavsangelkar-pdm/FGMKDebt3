"""
Debug tool for inspecting query results and retrieval quality.
Helps diagnose why answers might be incorrect or irrelevant.
"""

import logging
from typing import List, Dict, Any
from openai import OpenAI

from .config import settings
from .retrieve import HybridRetriever
from .answer import AnswerGenerator
from .utils.logging import setup_logging

logger = logging.getLogger(__name__)


class QueryDebugger:
    """Debug tool for analyzing query performance."""
    
    def __init__(self, openai_client: OpenAI):
        """Initialize the query debugger."""
        self.openai_client = openai_client
        self.retriever = HybridRetriever(openai_client)
        self.answer_generator = AnswerGenerator(openai_client)
        self.logger = logger
    
    def debug_query(self, doc_id: str, question: str, k: int = None) -> Dict[str, Any]:
        """
        Debug a query to understand retrieval and answer quality.
        
        Args:
            doc_id: Document identifier
            question: Query question
            k: Number of results to retrieve
            
        Returns:
            Dictionary with detailed debug information
        """
        self.logger.info(f"Starting debug query for doc_id={doc_id}, question={question}")
        
        debug_info = {
            "question": question,
            "doc_id": doc_id,
            "retrieval_analysis": {},
            "answer_analysis": {},
            "recommendations": []
        }
        
        try:
            # Step 1: Analyze retrieval
            retrieval_debug = self._analyze_retrieval(doc_id, question, k)
            debug_info["retrieval_analysis"] = retrieval_debug
            
            # Step 2: Analyze answer generation
            if retrieval_debug["final_results"]:
                answer_debug = self._analyze_answer_generation(question, retrieval_debug["final_results"], doc_id)
                debug_info["answer_analysis"] = answer_debug
            
            # Step 3: Generate recommendations
            recommendations = self._generate_recommendations(debug_info)
            debug_info["recommendations"] = recommendations
            
            return debug_info
            
        except Exception as e:
            self.logger.error(f"Debug query failed: {str(e)}", exc_info=True)
            debug_info["error"] = str(e)
            return debug_info
    
    def _analyze_retrieval(self, doc_id: str, question: str, k: int = None) -> Dict[str, Any]:
        """Analyze the retrieval process step by step."""
        analysis = {
            "query_embedding_generated": False,
            "faiss_results": [],
            "fts_results": [],
            "rrf_results": [],
            "reranked_results": [],
            "final_results": [],
            "confidence_scores": [],
            "retrieval_stats": {}
        }
        
        try:
            # Generate query embedding
            query_embedding = self.retriever._generate_query_embedding(question)
            analysis["query_embedding_generated"] = True
            
            # FAISS search
            faiss_results = self.retriever.faiss_store.search(doc_id, query_embedding, self.retriever.faiss_k)
            analysis["faiss_results"] = [
                {
                    "chunk_id": r["chunk_id"],
                    "page": r["page"],
                    "text_preview": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"],
                    "faiss_score": r.get("faiss_score", 0.0),
                    "text_length": len(r["text"])
                }
                for r in faiss_results
            ]
            
            # FTS search
            fts_results = self.retriever.sqlite_store.bm25_search(doc_id, question, self.retriever.fts_k)
            analysis["fts_results"] = [
                {
                    "chunk_id": r["chunk_id"],
                    "page": r["page"],
                    "text_preview": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"],
                    "bm25_score": r.get("bm25_score", 0.0),
                    "text_length": len(r["text"])
                }
                for r in fts_results
            ]
            
            # RRF combination
            rrf_results = self.retriever._reciprocal_rank_fusion(faiss_results, fts_results)
            analysis["rrf_results"] = [
                {
                    "chunk_id": r["chunk_id"],
                    "page": r["page"],
                    "text_preview": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"],
                    "rrf_score": r.get("rrf_score", 0.0),
                    "faiss_rank": r.get("faiss_rank"),
                    "fts_rank": r.get("fts_rank"),
                    "text_length": len(r["text"])
                }
                for r in rrf_results[:10]  # Top 10 for analysis
            ]
            
            # Reranking
            rerank_candidates = rrf_results[:self.retriever.rerank_candidates]
            reranked_results = self.retriever._rerank_candidates(question, rerank_candidates)
            analysis["reranked_results"] = [
                {
                    "chunk_id": r["chunk_id"],
                    "page": r["page"],
                    "text_preview": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"],
                    "rerank_score": r.get("rerank_score", 0.0),
                    "confidence": r.get("confidence", 0.0),
                    "combined_score": r.get("combined_score", 0.0),
                    "text_length": len(r["text"])
                }
                for r in reranked_results
            ]
            
            # Final results after confidence threshold
            final_results = self.retriever._apply_confidence_threshold(reranked_results[:k or self.retriever.rerank_top_n])
            analysis["final_results"] = [
                {
                    "chunk_id": r["chunk_id"],
                    "page": r["page"],
                    "text_preview": r["text"][:100] + "..." if len(r["text"]) > 100 else r["text"],
                    "rerank_score": r.get("rerank_score", 0.0),
                    "confidence": r.get("confidence", 0.0),
                    "combined_score": r.get("combined_score", 0.0),
                    "text_length": len(r["text"])
                }
                for r in final_results
            ]
            
            # Confidence score analysis
            if reranked_results:
                confidences = [r.get("confidence", 0.0) for r in reranked_results]
                analysis["confidence_scores"] = {
                    "min": min(confidences),
                    "max": max(confidences),
                    "avg": sum(confidences) / len(confidences),
                    "threshold": self.retriever.confidence_threshold,
                    "above_threshold": sum(1 for c in confidences if c >= self.retriever.confidence_threshold)
                }
            
            # Retrieval statistics
            analysis["retrieval_stats"] = {
                "faiss_count": len(faiss_results),
                "fts_count": len(fts_results),
                "rrf_count": len(rrf_results),
                "reranked_count": len(reranked_results),
                "final_count": len(final_results),
                "confidence_threshold": self.retriever.confidence_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Retrieval analysis failed: {str(e)}", exc_info=True)
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_answer_generation(self, question: str, retrieved_results: List[Dict], doc_id: str) -> Dict[str, Any]:
        """Analyze the answer generation process."""
        analysis = {
            "answer_generated": False,
            "answer_text": "",
            "citations": [],
            "snippets_used": [],
            "confidence": 0.0,
            "answer_quality": {}
        }
        
        try:
            # Generate answer
            answer_result = self.answer_generator.generate_answer(question, retrieved_results, doc_id)
            
            analysis["answer_generated"] = True
            analysis["answer_text"] = answer_result.answer
            analysis["citations"] = [
                {
                    "page": c.page,
                    "chunk_id": c.chunk_id,
                    "char_start": c.char_start,
                    "char_end": c.char_end
                }
                for c in answer_result.citations
            ]
            analysis["snippets_used"] = [
                {
                    "page": s.page,
                    "text_preview": s.text[:100] + "..." if len(s.text) > 100 else s.text,
                    "text_length": len(s.text)
                }
                for s in answer_result.snippets
            ]
            analysis["confidence"] = answer_result.confidence or 0.0
            
            # Answer quality analysis
            analysis["answer_quality"] = {
                "length": len(answer_result.answer),
                "has_citations": len(answer_result.citations) > 0,
                "citation_count": len(answer_result.citations),
                "is_refusal": answer_result.answer.strip() == "Not found in document.",
                "found": answer_result.found,
                "snippet_count": len(answer_result.snippets)
            }
            
        except Exception as e:
            self.logger.error(f"Answer analysis failed: {str(e)}", exc_info=True)
            analysis["error"] = str(e)
        
        return analysis
    
    def _generate_recommendations(self, debug_info: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on debug analysis."""
        recommendations = []
        
        retrieval = debug_info.get("retrieval_analysis", {})
        answer = debug_info.get("answer_analysis", {})
        
        # Check if no results were retrieved
        if not retrieval.get("final_results"):
            if retrieval.get("faiss_results") or retrieval.get("fts_results"):
                recommendations.append("Results were retrieved but filtered out by confidence threshold. Consider lowering CONFIDENCE_THRESHOLD.")
            else:
                recommendations.append("No results retrieved from either FAISS or FTS. Check if document is properly indexed.")
        
        # Check confidence scores
        conf_scores = retrieval.get("confidence_scores", {})
        if conf_scores.get("max", 0) < 0.5:
            recommendations.append("Low confidence scores detected. Consider improving chunk quality or adjusting reranking parameters.")
        
        # Check answer quality
        answer_quality = answer.get("answer_quality", {})
        if answer_quality.get("is_refusal", False):
            recommendations.append("Answer generation returned 'Not found in document.' Consider improving retrieval or chunking strategy.")
        
        if not answer_quality.get("has_citations", False) and answer_quality.get("length", 0) > 50:
            recommendations.append("Answer lacks citations despite being substantial. Check citation extraction logic.")
        
        # Check retrieval diversity
        faiss_count = retrieval.get("retrieval_stats", {}).get("faiss_count", 0)
        fts_count = retrieval.get("retrieval_stats", {}).get("fts_count", 0)
        if faiss_count == 0 or fts_count == 0:
            recommendations.append("One retrieval method returned no results. Check both FAISS and FTS indices.")
        
        return recommendations


def debug_query_cli(doc_id: str, question: str, k: int = None):
    """CLI function to debug a query."""
    setup_logging()
    
    try:
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        debugger = QueryDebugger(openai_client)
        
        result = debugger.debug_query(doc_id, question, k)
        
        print(f"\n=== QUERY DEBUG REPORT ===")
        print(f"Question: {result['question']}")
        print(f"Document: {result['doc_id']}")
        
        # Retrieval analysis
        retrieval = result.get("retrieval_analysis", {})
        print(f"\n--- RETRIEVAL ANALYSIS ---")
        print(f"FAISS results: {len(retrieval.get('faiss_results', []))}")
        print(f"FTS results: {len(retrieval.get('fts_results', []))}")
        print(f"Final results: {len(retrieval.get('final_results', []))}")
        
        if retrieval.get("confidence_scores"):
            conf = retrieval["confidence_scores"]
            print(f"Confidence: min={conf['min']:.3f}, max={conf['max']:.3f}, avg={conf['avg']:.3f}")
            print(f"Above threshold: {conf['above_threshold']}/{len(retrieval.get('reranked_results', []))}")
        
        # Answer analysis
        answer = result.get("answer_analysis", {})
        if answer.get("answer_generated"):
            print(f"\n--- ANSWER ANALYSIS ---")
            print(f"Answer: {answer['answer_text']}")
            print(f"Citations: {len(answer.get('citations', []))}")
            print(f"Confidence: {answer.get('confidence', 0):.3f}")
        
        # Recommendations
        recommendations = result.get("recommendations", [])
        if recommendations:
            print(f"\n--- RECOMMENDATIONS ---")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        return result
        
    except Exception as e:
        print(f"Debug failed: {str(e)}")
        return None


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python debug_query.py <doc_id> <question> [k]")
        sys.exit(1)
    
    doc_id = sys.argv[1]
    question = sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    debug_query_cli(doc_id, question, k)
