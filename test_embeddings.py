#!/usr/bin/env python3
import sys
sys.path.append('.')

from rag_app.config import settings
from openai import OpenAI
import numpy as np

def test_embeddings():
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    try:
        response = client.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=["This is a test"]
        )
        
        embeddings = np.array([data.embedding for data in response.data])
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        print(f"First few values: {embeddings[0][:5]}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_embeddings()
