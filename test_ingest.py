#!/usr/bin/env python3
import sys
sys.path.append('.')

from rag_app.ingest import DocumentIngester
from openai import OpenAI

def test_ingestion():
    # Create a mock OpenAI client
    openai_client = OpenAI(api_key="test_key")
    ingester = DocumentIngester(openai_client)
    
    pdf_path = "/Users/pdm/Desktop/FGMK/Wingspire Credit Agreement.pdf"
    doc_id = "test_doc"
    
    try:
        print(f"Testing document ingestion for: {pdf_path}")
        response = ingester.ingest_document(pdf_path, doc_id)
        print(f"Successfully ingested document: {response}")
    except Exception as e:
        print(f"Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ingestion()
