#!/usr/bin/env python3
"""
Simple server startup script.
"""

import uvicorn
from app import app

if __name__ == "__main__":
    print("Starting RAG server on http://localhost:3001")
    uvicorn.run(app, host="0.0.0.0", port=3001, log_level="info")
