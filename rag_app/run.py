#!/usr/bin/env python3
"""
Simple script to run the RAG application.
"""

import uvicorn
from config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )
