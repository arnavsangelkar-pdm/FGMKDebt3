import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from config import settings
from models import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse,
    DocumentStats, HealthResponse, ErrorResponse
)
from utils.logging import setup_logging, log_timing, log_error

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document Q&A Service",
    description="A minimal RAG web service for document question answering with citations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting RAG Document Q&A Service", extra={"version": "1.0.0"})
    
    # Skip OpenAI API key validation for now
    logger.info("Skipping OpenAI API key validation for development")
    
    # Ensure data directories exist
    for path in settings.paths.values():
        path.mkdir(parents=True, exist_ok=True)
    logger.info("Application startup completed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    doc_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Ingest a PDF document into the RAG system.
    """
    start_time = time.time()
    logger.info(f"Starting document ingestion doc_id={doc_id}, filename={file.filename}")
    
    try:
        # Validate doc_id format
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]+$', doc_id):
            raise HTTPException(
                status_code=400,
                detail="doc_id must contain only alphanumeric characters, underscores, and hyphens"
            )
        
        # Validate file type
        if not file.content_type or file.content_type != "application/pdf":
            raise HTTPException(
                status_code=400,
                detail="File must be a PDF (application/pdf)"
            )
        
        # Save uploaded file
        pdf_path = settings.docs_path / f"{doc_id}.pdf"
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved uploaded file doc_id={doc_id}, file_size={len(content)}")
        
        # For now, just return a simple response without full processing
        total_time = time.time() - start_time
        
        response = IngestResponse(
            doc_id=doc_id,
            pages_count=1,  # Placeholder
            chunks_count=1,  # Placeholder
            processing_time=total_time,
            message="Document uploaded successfully (simplified processing)"
        )
        
        log_timing(logger, "total_ingestion", total_time, doc_id=doc_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, "document_ingestion", doc_id=doc_id)
        raise HTTPException(status_code=500, detail="Document ingestion failed")

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a document with a question.
    """
    start_time = time.time()
    logger.info(f"Starting document query doc_id={request.doc_id}, question={request.question}")
    
    try:
        # For now, return a simple mock response
        total_time = time.time() - start_time
        
        response = QueryResponse(
            answer="This is a mock response. Full RAG processing is not yet enabled.",
            citations=[],
            snippets=[],
            found=True,
            confidence=0.8,
            processing_time=total_time
        )
        
        log_timing(logger, "total_query", total_time, doc_id=request.doc_id, found=True)
        
        return response
        
    except Exception as e:
        log_error(logger, e, "document_query", doc_id=request.doc_id, question=request.question)
        raise HTTPException(status_code=500, detail="Document query failed")

@app.get("/docs/{doc_id}/stats", response_model=DocumentStats)
async def get_document_stats(doc_id: str):
    """
    Get statistics for a document.
    """
    logger.info(f"Getting document stats doc_id={doc_id}")
    
    try:
        # Check if document exists
        pdf_file = settings.docs_path / f"{doc_id}.pdf"
        if not pdf_file.exists():
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        # Get file modification time
        last_ingested = pdf_file.stat().st_mtime
        
        # Create response
        response = DocumentStats(
            doc_id=doc_id,
            pages_count=1,  # Placeholder
            chunks_count=1,  # Placeholder
            faiss_vectors_count=0,  # Placeholder
            last_ingested=last_ingested,
            file_size_mb=pdf_file.stat().st_size / (1024 * 1024),
            index_size_mb=0  # Placeholder
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, "document_stats", doc_id=doc_id)
        raise HTTPException(status_code=500, detail="Failed to get document stats")

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with proper error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status_code=exc.status_code
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    log_error(logger, exc, "unhandled_exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred",
            status_code=500
        ).dict()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)