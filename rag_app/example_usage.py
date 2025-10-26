#!/usr/bin/env python3
"""
Example usage of the RAG application.
This script demonstrates how to use the RAG service programmatically.
"""

import asyncio
import aiohttp
import json
from pathlib import Path

async def test_rag_service():
    """Test the RAG service with example operations."""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        # Test health check
        print("1. Testing health check...")
        async with session.get(f"{base_url}/health") as response:
            if response.status == 200:
                data = await response.json()
                print(f"✓ Health check passed: {data}")
            else:
                print(f"✗ Health check failed: {response.status}")
                return
        
        # Test document ingestion (you'll need to provide a PDF file)
        print("\n2. Testing document ingestion...")
        pdf_path = Path("example.pdf")  # Replace with your PDF file
        
        if not pdf_path.exists():
            print(f"✗ PDF file not found: {pdf_path}")
            print("  Please place a PDF file named 'example.pdf' in the current directory")
            return
        
        # Upload PDF
        with open(pdf_path, 'rb') as f:
            data = aiohttp.FormData()
            data.add_field('doc_id', 'example_doc')
            data.add_field('file', f, filename='example.pdf', content_type='application/pdf')
            
            async with session.post(f"{base_url}/ingest", data=data) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ Document ingested successfully:")
                    print(f"  - Pages: {result['pages_count']}")
                    print(f"  - Chunks: {result['chunks_count']}")
                    print(f"  - Processing time: {result['processing_time']:.2f}s")
                else:
                    error = await response.json()
                    print(f"✗ Ingestion failed: {error}")
                    return
        
        # Test document stats
        print("\n3. Testing document stats...")
        async with session.get(f"{base_url}/docs/example_doc/stats") as response:
            if response.status == 200:
                stats = await response.json()
                print(f"✓ Document stats:")
                print(f"  - Pages: {stats['pages_count']}")
                print(f"  - Chunks: {stats['chunks_count']}")
                print(f"  - FAISS vectors: {stats['faiss_vectors_count']}")
                print(f"  - File size: {stats['file_size_mb']:.2f} MB")
            else:
                error = await response.json()
                print(f"✗ Stats failed: {error}")
        
        # Test document querying
        print("\n4. Testing document querying...")
        questions = [
            "What is this document about?",
            "What are the main topics discussed?",
            "Can you summarize the key points?"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n4.{i} Query: {question}")
            
            query_data = {
                "doc_id": "example_doc",
                "question": question,
                "k": 3
            }
            
            async with session.post(
                f"{base_url}/query",
                json=query_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ Answer: {result['answer']}")
                    print(f"  - Found: {result['found']}")
                    print(f"  - Confidence: {result.get('confidence', 'N/A')}")
                    print(f"  - Citations: {len(result['citations'])}")
                    print(f"  - Processing time: {result['processing_time']:.2f}s")
                else:
                    error = await response.json()
                    print(f"✗ Query failed: {error}")

def main():
    """Run the example."""
    print("RAG Service Example Usage")
    print("=" * 40)
    print("Make sure the RAG service is running on http://localhost:8000")
    print("You can start it with: python run.py")
    print()
    
    try:
        asyncio.run(test_rag_service())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"\nExample failed: {e}")

if __name__ == "__main__":
    main()
