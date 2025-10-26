"""
FastAPI application for the RAG web service.
Provides endpoints for document ingestion, querying, and health checks.
"""

import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from .config import settings
from .models import (
    IngestRequest, IngestResponse, QueryRequest, QueryResponse,
    HealthResponse, ErrorResponse
)
from .ingest import DocumentIngester
from .retrieve import HybridRetriever
from .answer import AnswerGenerator
from .debug_query import QueryDebugger
from .utils.logging import setup_logging, log_timing, log_error


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
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3002",
        "https://*.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Initialize OpenAI client
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# Initialize services
ingester = DocumentIngester(openai_client)
retriever = HybridRetriever(openai_client)
answer_generator = AnswerGenerator(openai_client)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting RAG Document Q&A Service version=1.0.0")
    
    # Validate OpenAI API key (skip for test keys)
    if settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "test_key":
        try:
            # Test the API key with a simple request
            openai_client.models.list()
            logger.info("OpenAI API key validated successfully")
        except Exception as e:
            logger.error(f"OpenAI API key validation failed: {str(e)}")
            raise HTTPException(status_code=500, detail="OpenAI API key validation failed")
    else:
        logger.info("Skipping OpenAI API key validation for test environment")
    
    # Data directories are created on import in config.py
    logger.info("Application startup completed")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
async def serve_frontend():
    """Serve the frontend application."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(static_path))
    else:
        return {"message": "Frontend not built. Please run 'npm run build' in the rag-frontend directory."}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    doc_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Ingest a PDF document into the RAG system.
    
    Args:
        doc_id: Unique document identifier
        file: PDF file to ingest
        
    Returns:
        IngestResponse with processing results
    """
    start_time = time.time()
    logger.info(f"Starting document ingestion for doc_id={doc_id}, filename={file.filename}")
    
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
        pdf_path = settings.paths["docs"] / f"{doc_id}.pdf"
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved uploaded file doc_id={doc_id}, file_size={len(content)}")
        
        # Ingest document
        response = ingester.ingest_document(pdf_path, doc_id)
        
        # Log total processing time
        total_time = time.time() - start_time
        log_timing(logger, "total_ingestion", total_time, doc_id=doc_id)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        log_error(logger, e, "document_ingestion")
        raise HTTPException(status_code=500, detail="Document ingestion failed")


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a document with a question.
    
    Args:
        request: QueryRequest with doc_id, question, and optional parameters
        
    Returns:
        QueryResponse with answer, citations, and snippets
    """
    start_time = time.time()
    logger.info(f"Starting document query doc_id={request.doc_id}, question={request.question}")
    
    try:
        # Retrieve relevant chunks
        retrieve_start = time.time()
        retrieved_results = retriever.retrieve(
            doc_id=request.doc_id,
            question=request.question,
            k=request.k
        )
        retrieve_time = time.time() - retrieve_start
        log_timing(logger, "retrieval", retrieve_time, doc_id=request.doc_id, results_count=len(retrieved_results))
        
        # Generate answer
        answer_start = time.time()
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
        log_error(logger, e, "document_query")
        raise HTTPException(status_code=500, detail="Document query failed")


@app.post("/debug/query")
async def debug_query(request: QueryRequest):
    """
    Debug a query to understand retrieval and answer quality.
    
    Args:
        request: QueryRequest with doc_id, question, and optional parameters
        
    Returns:
        Detailed debug information about the query process
    """
    logger.info(f"Starting debug query doc_id={request.doc_id}, question={request.question}")
    
    try:
        # Initialize debugger
        debugger = QueryDebugger(openai_client)
        
        # Run debug analysis
        debug_result = debugger.debug_query(
            doc_id=request.doc_id,
            question=request.question,
            k=request.k
        )
        
        return JSONResponse(content=debug_result)
        
    except Exception as e:
        log_error(logger, e, "debug_query", doc_id=request.doc_id, question=request.question)
        raise HTTPException(status_code=500, detail="Debug query failed")


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
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
