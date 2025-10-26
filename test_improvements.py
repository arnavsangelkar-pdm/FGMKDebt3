#!/usr/bin/env python3
"""
Test script to verify the RAG improvements.
Run this to test query quality after the improvements.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the rag_app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "rag_app"))

from openai import OpenAI
from config import settings
from debug_query import QueryDebugger


async def test_query_quality(doc_id: str, question: str):
    """Test query quality with debug information."""
    print(f"\n{'='*60}")
    print(f"TESTING QUERY QUALITY")
    print(f"{'='*60}")
    print(f"Document ID: {doc_id}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    try:
        # Initialize OpenAI client
        openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Initialize debugger
        debugger = QueryDebugger(openai_client)
        
        # Run debug analysis
        debug_result = debugger.debug_query(doc_id, question)
        
        # Print results
        print(f"\n--- RETRIEVAL ANALYSIS ---")
        retrieval = debug_result.get("retrieval_analysis", {})
        print(f"FAISS results: {len(retrieval.get('faiss_results', []))}")
        print(f"FTS results: {len(retrieval.get('fts_results', []))}")
        print(f"Final results: {len(retrieval.get('final_results', []))}")
        
        if retrieval.get("confidence_scores"):
            conf = retrieval["confidence_scores"]
            print(f"Confidence scores:")
            print(f"  Min: {conf['min']:.3f}")
            print(f"  Max: {conf['max']:.3f}")
            print(f"  Avg: {conf['avg']:.3f}")
            print(f"  Above threshold: {conf['above_threshold']}")
            print(f"  Threshold: {conf['threshold']}")
        
        # Show top retrieved chunks
        final_results = retrieval.get("final_results", [])
        if final_results:
            print(f"\n--- TOP RETRIEVED CHUNKS ---")
            for i, chunk in enumerate(final_results[:3], 1):
                print(f"{i}. Page {chunk['page']} (Confidence: {chunk['confidence']:.3f})")
                print(f"   Text: {chunk['text_preview']}")
                print()
        
        # Show answer analysis
        answer = debug_result.get("answer_analysis", {})
        if answer.get("answer_generated"):
            print(f"--- ANSWER ANALYSIS ---")
            print(f"Answer: {answer['answer_text']}")
            print(f"Citations: {len(answer.get('citations', []))}")
            print(f"Confidence: {answer.get('confidence', 0):.3f}")
            
            quality = answer.get("answer_quality", {})
            print(f"Answer length: {quality.get('length', 0)} characters")
            print(f"Has citations: {quality.get('has_citations', False)}")
            print(f"Is refusal: {quality.get('is_refusal', False)}")
        
        # Show recommendations
        recommendations = debug_result.get("recommendations", [])
        if recommendations:
            print(f"\n--- RECOMMENDATIONS ---")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
        
        return debug_result
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return None


def main():
    """Main test function."""
    print("RAG System Quality Test")
    print("This script tests the improved RAG system with debug information.")
    
    # Check if we have a document to test with
    docs_dir = Path("data/docs")
    if not docs_dir.exists():
        print(f"Error: {docs_dir} directory not found. Please ensure you have ingested a document first.")
        return
    
    # List available documents
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Error: No PDF files found in {docs_dir}. Please ingest a document first.")
        return
    
    print(f"\nAvailable documents:")
    for i, pdf_file in enumerate(pdf_files, 1):
        doc_id = pdf_file.stem
        print(f"{i}. {doc_id}")
    
    # Get user input
    try:
        choice = input(f"\nSelect document (1-{len(pdf_files)}): ").strip()
        doc_idx = int(choice) - 1
        if doc_idx < 0 or doc_idx >= len(pdf_files):
            print("Invalid selection.")
            return
        
        doc_id = pdf_files[doc_idx].stem
        question = input("Enter your question: ").strip()
        
        if not question:
            print("No question provided.")
            return
        
        # Run the test
        asyncio.run(test_query_quality(doc_id, question))
        
    except KeyboardInterrupt:
        print("\nTest cancelled.")
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
