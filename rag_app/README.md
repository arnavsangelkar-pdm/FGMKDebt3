# RAG Document Q&A Service

A minimal, production-ready Retrieval-Augmented Generation (RAG) web service that ingests large PDFs, supports Q&A grounded strictly in documents, and returns answers with page-level citations.

## Features

- **PDF Processing**: Parse large PDFs (200+ pages) using PyMuPDF
- **Token-Aware Chunking**: Intelligent text splitting using tiktoken
- **Hybrid Retrieval**: Combines vector search (FAISS) and keyword search (SQLite FTS5)
- **Reciprocal Rank Fusion**: Advanced ranking algorithm for better results
- **Reranking**: Uses BAAI/bge-reranker-base for improved relevance
- **Citation Support**: Returns answers with precise page-level citations
- **Local Persistence**: All indices stored locally for fast access
- **Production Ready**: Docker support and Render deployment configuration

## Tech Stack

- **Parser**: PyMuPDF (fitz)
- **Chunking**: tiktoken (token-aware splitting)
- **Vector Store**: FAISS (local, persisted to disk)
- **Keyword Search**: SQLite FTS5 (local file DB, persisted)
- **Reranker**: BAAI/bge-reranker-base via sentence-transformers
- **LLM & Embeddings**: OpenAI (gpt-4o/gpt-4o-mini for answers; text-embedding-3-small for embeddings)
- **API**: FastAPI + Uvicorn
- **Deployment**: Docker + Render

## Quick Start

### Local Development

1. **Clone and setup**:
   ```bash
   cd rag_app
   cp .env.sample .env
   # Edit .env and set OPENAI_API_KEY
   pip install -r requirements.txt
   ```

2. **Run the service**:
   ```bash
   uvicorn rag_app.app:app --host 0.0.0.0 --port 8000
   ```

3. **Test the service**:
   ```bash
   curl http://localhost:8000/health
   ```

### Ingest a Document

```bash
curl -F doc_id=mydoc -F file=@/path/to/your/document.pdf http://localhost:8000/ingest
```

### Query a Document

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"mydoc","question":"What are the main findings?"}'
```

### Get Document Stats

```bash
curl http://localhost:8000/docs/mydoc/stats
```

## API Endpoints

### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### `POST /ingest`
Ingest a PDF document into the RAG system.

**Parameters**:
- `doc_id` (string): Unique document identifier
- `file` (file): PDF file to ingest

**Response**:
```json
{
  "doc_id": "mydoc",
  "pages_count": 150,
  "chunks_count": 450,
  "processing_time": 45.2,
  "message": "Successfully ingested 150 pages into 450 chunks"
}
```

### `POST /query`
Query a document with a question.

**Request Body**:
```json
{
  "doc_id": "mydoc",
  "question": "What are the main findings?",
  "k": 5
}
```

**Response**:
```json
{
  "answer": "The main findings show significant improvements in performance [Doc: p. 45]. The results indicate a 25% increase in efficiency [Doc: p. 67].",
  "citations": [
    {
      "doc_id": "mydoc",
      "page": 45,
      "chunk_id": "chunk_123",
      "char_start": 100,
      "char_end": 200
    }
  ],
  "snippets": [
    {
      "page": 45,
      "text": "The main findings show significant improvements..."
    }
  ],
  "found": true,
  "confidence": 0.85,
  "processing_time": 2.3
}
```

### `GET /docs/{doc_id}/stats`
Get statistics for a document.

**Response**:
```json
{
  "doc_id": "mydoc",
  "pages_count": 150,
  "chunks_count": 450,
  "faiss_vectors_count": 450,
  "last_ingested": "2024-01-15T10:30:00Z",
  "file_size_mb": 12.5,
  "index_size_mb": 8.2
}
```

## Configuration

The service can be configured using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model for answer generation |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI model for embeddings |
| `CHUNK_SIZE` | `500` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `FAISS_K` | `20` | Number of FAISS results to retrieve |
| `FTS_K` | `20` | Number of FTS5 results to retrieve |
| `RERANK_TOP_N` | `5` | Final number of results after reranking |
| `RERANK_CANDIDATES` | `30` | Number of candidates for reranking |
| `CONFIDENCE_THRESHOLD` | `0.35` | Minimum confidence for answers |
| `MAX_UPLOAD_SIZE` | `104857600` | Maximum upload size in bytes (100MB) |

## Architecture

### Document Processing Pipeline

1. **PDF Parsing**: Extract text and metadata using PyMuPDF
2. **Chunking**: Split text into overlapping chunks using tiktoken
3. **Embedding**: Generate embeddings using OpenAI text-embedding-3-small
4. **Indexing**: Store in both FAISS (vector) and SQLite FTS5 (keyword) indices
5. **Persistence**: Save chunks metadata to Parquet for debugging

### Query Processing Pipeline

1. **Query Embedding**: Generate embedding for the user question
2. **Hybrid Retrieval**: Search both FAISS and SQLite FTS5 indices
3. **Reciprocal Rank Fusion**: Combine results using RRF algorithm
4. **Reranking**: Use BGE reranker to improve relevance
5. **Answer Generation**: Generate answer with citations using OpenAI
6. **Citation Extraction**: Parse citations from the generated answer

### Data Storage

```
data/
├── docs/           # Uploaded PDF files
├── indices/        # FAISS vector indices (*.faiss, *.faiss.meta.json)
├── sqlite/         # SQLite FTS5 databases (*.db)
└── chunks/         # Chunk metadata snapshots (*.parquet)
```

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

The tests cover:
- Text chunking functionality
- Hybrid retrieval and RRF
- Answer generation and citation extraction

## Docker Deployment

### Build and Run Locally

```bash
# Build the image
docker build -t rag-app .

# Run the container
docker run -p 8000:10000 \
  -e OPENAI_API_KEY=your_key_here \
  rag-app
```

### Deploy to Render

1. **Connect your repository** to Render
2. **Create a new Web Service** from the repository
3. **Set environment variables**:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - Other variables will use defaults from `render.yaml`
4. **Deploy**: Render will automatically build and deploy using the Dockerfile

The service will be available at your Render URL with the `/health` endpoint for monitoring.

## Performance Considerations

- **Embedding Generation**: The most time-consuming step during ingestion
- **Reranking**: Adds ~1-2 seconds to query time but significantly improves quality
- **Memory Usage**: FAISS indices are loaded into memory for fast search
- **Storage**: Each document requires ~50-100MB of storage for indices

## Troubleshooting

### Common Issues

1. **OpenAI API Key**: Ensure your API key is valid and has sufficient credits
2. **PDF Processing**: Some PDFs may have poor text extraction quality
3. **Memory**: Large documents may require more memory for processing
4. **Storage**: Ensure sufficient disk space for indices

### Logs

The service uses structured JSON logging. Check logs for:
- Processing times for each step
- Error details with context
- Performance metrics

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue on GitHub with relevant details
