#!/usr/bin/env python3
"""
Test script to verify the RAG application installation.
"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported."""
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "pydantic_settings",
        "fitz",  # PyMuPDF
        "tiktoken",
        "faiss",
        "pandas",
        "pyarrow",
        "openai",
        "sqlite3"
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from config import settings
        print(f"✓ Configuration loaded")
        print(f"  - Chunk size: {settings.chunk_size}")
        print(f"  - Chunk overlap: {settings.chunk_overlap}")
        print(f"  - Data directory: {settings.data_dir}")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_openai_key():
    """Test OpenAI API key."""
    print("\nTesting OpenAI API key...")
    try:
        from config import settings
        if settings.openai_api_key and settings.openai_api_key != "your_openai_api_key_here":
            print("✓ OpenAI API key is set")
            return True
        else:
            print("✗ OpenAI API key not set or using default value")
            print("  Please set OPENAI_API_KEY in your .env file")
            return False
    except Exception as e:
        print(f"✗ OpenAI API key error: {e}")
        return False

def main():
    """Run all tests."""
    print("RAG Application Installation Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_openai_key
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 40)
    if all(results):
        print("✓ All tests passed! The application is ready to run.")
        print("\nTo start the application:")
        print("  python run.py")
        print("  or")
        print("  uvicorn app:app --reload")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
