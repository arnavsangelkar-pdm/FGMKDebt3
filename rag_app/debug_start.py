#!/usr/bin/env python3
"""
Debug startup script for the RAG application.
"""

import sys
import traceback

def main():
    try:
        print("Starting RAG application...")
        
        # Test imports
        print("Testing imports...")
        import app
        print("✓ App imported successfully")
        
        # Test configuration
        print("Testing configuration...")
        from config import settings
        print(f"✓ Configuration loaded: {settings.host}:{settings.port}")
        
        # Test OpenAI client
        print("Testing OpenAI client...")
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        print("✓ OpenAI client created")
        
        # Start server
        print("Starting server...")
        import uvicorn
        uvicorn.run(
            "app:app",
            host=settings.host,
            port=settings.port,
            reload=False,
            log_level="info"
        )
        
    except Exception as e:
        print(f"Error starting application: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
