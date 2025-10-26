#!/usr/bin/env python3
import sys
sys.path.append('.')

from rag_app.config import settings
from openai import OpenAI
import numpy as np
import faiss

def test_faiss():
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    try:
        # Generate embeddings
        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=["This is a test document"]
        )
        
        embeddings = np.array([data.embedding for data in response.data], dtype='float32')
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Embeddings dtype: {embeddings.dtype}")
        
        # Create FAISS index
        index = faiss.IndexFlatIP(1536)
        print("Created FAISS index")
        
        # Normalize embeddings
        faiss.normalize_L2(embeddings)
        print("Normalized embeddings")
        
        # Add to index
        index.add(embeddings)
        print("Added embeddings to index")
        
        print("Success!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_faiss()
