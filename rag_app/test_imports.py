#!/usr/bin/env python3
"""
Test script to check imports.
"""

def test_imports():
    try:
        print("Testing basic imports...")
        import fastapi
        print("✓ FastAPI imported")
        
        import uvicorn
        print("✓ Uvicorn imported")
        
        import pydantic
        print("✓ Pydantic imported")
        
        import pydantic_settings
        print("✓ Pydantic Settings imported")
        
        import fitz
        print("✓ PyMuPDF imported")
        
        import tiktoken
        print("✓ Tiktoken imported")
        
        import faiss
        print("✓ FAISS imported")
        
        import pandas
        print("✓ Pandas imported")
        
        import openai
        print("✓ OpenAI imported")
        
        print("\nTesting sentence-transformers...")
        from sentence_transformers import CrossEncoder
        print("✓ CrossEncoder imported")
        
        print("\nTesting app modules...")
        from config import settings
        print("✓ Config imported")
        
        from models import IngestRequest
        print("✓ Models imported")
        
        from utils.parsing import PDFParser
        print("✓ Parsing imported")
        
        from utils.chunking import TokenAwareChunker
        print("✓ Chunking imported")
        
        from store.faiss_store import FAISSStore
        print("✓ FAISS Store imported")
        
        from store.sqlite_store import SQLiteStore
        print("✓ SQLite Store imported")
        
        from retrieve import HybridRetriever
        print("✓ Retrieval imported")
        
        from answer import AnswerGenerator
        print("✓ Answer imported")
        
        from ingest import DocumentIngester
        print("✓ Ingest imported")
        
        print("\nAll imports successful!")
        return True
        
    except Exception as e:
        print(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_imports()
