"""
Working FastAPI app with lazy loading of heavy dependencies.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from config import settings
from models import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse,
    HealthResponse, ErrorResponse
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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading
openai_client = None
ingester = None
retriever = None
answer_generator = None

def get_openai_client():
    """Get or create OpenAI client."""
    global openai_client
    if openai_client is None:
        openai_client = OpenAI(api_key=settings.openai_api_key)
    return openai_client

def get_ingester():
    """Get or create document ingester."""
    global ingester
    if ingester is None:
        from ingest import DocumentIngester
        ingester = DocumentIngester(get_openai_client())
    return ingester

def get_retriever():
    """Get or create hybrid retriever."""
    global retriever
    if retriever is None:
        from retrieve import HybridRetriever
        retriever = HybridRetriever(get_openai_client())
    return retriever

def get_answer_generator():
    """Get or create answer generator."""
    global answer_generator
    if answer_generator is None:
        from answer import AnswerGenerator
        answer_generator = AnswerGenerator(get_openai_client())
    return answer_generator

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting RAG Document Q&A Service", extra={"version": "1.0.0"})
    
    # Validate OpenAI API key
    try:
        client = get_openai_client()
        client.models.list()
        logger.info("OpenAI API key validated successfully")
    except Exception as e:
        logger.error(f"OpenAI API key validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="OpenAI API key validation failed")
    
    # Ensure data directories exist
    settings.setup_directories()
    logger.info("Application startup completed")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(status="ok", version="1.0.0")

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    doc_id: str,
    file: UploadFile = File(...)
):
    """
    Ingest a PDF document into the RAG system.
    """
    start_time = time.time()
    logger.info(f"Starting document ingestion", doc_id=doc_id, filename=file.filename)
    
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
        
        # Check file size
        if file.size and file.size > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds maximum allowed size of {settings.max_upload_size / (1024*1024):.1f}MB"
            )
        
        # Save uploaded file
        pdf_path = settings.docs_path / f"{doc_id}.pdf"
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved uploaded file", doc_id=doc_id, file_size=len(content))
        
        # Ingest document
        ingester = get_ingester()
        response = ingester.ingest_document(pdf_path, doc_id)
        
        # Log total processing time
        total_time = time.time() - start_time
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
    logger.info(f"Starting document query", doc_id=request.doc_id, question=request.question)
    
    try:
        # Retrieve relevant chunks
        retrieve_start = time.time()
        retriever = get_retriever()
        retrieved_results = retriever.retrieve(
            doc_id=request.doc_id,
            question=request.question,
            k=request.k
        )
        retrieve_time = time.time() - retrieve_start
        log_timing(logger, "retrieval", retrieve_time, doc_id=request.doc_id, results_count=len(retrieved_results))
        
        # Generate answer
        answer_start = time.time()
        answer_generator = get_answer_generator()
        answer_result = answer_generator.generate_answer(
            question=request.question,
            retrieved_results=retrieved_results,
            doc_id=request.doc_id
        )
        answer_time = time.time() - answer_start
        log_timing(logger, "answer_generation", answer_time, doc_id=request.doc_id, found=answer_result.found)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Create response
        response = QueryResponse(
            answer=answer_result.answer,
            citations=answer_result.citations,
            snippets=answer_result.snippets,
            found=answer_result.found,
            confidence=answer_result.confidence,
            processing_time=total_time
        )
        
        log_timing(logger, "total_query", total_time, doc_id=request.doc_id, found=answer_result.found)
        
        return response
        
    except Exception as e:
        log_error(logger, e, "document_query", doc_id=request.doc_id, question=request.question)
        raise HTTPException(status_code=500, detail="Document query failed")


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
    uvicorn.run(app, host="0.0.0.0", port=3001, log_level="info")
