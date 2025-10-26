import React, { useState } from 'react';
import './App.css';
import DocumentUpload from './components/DocumentUpload';
import DocumentQuery from './components/DocumentQuery';

interface Document {
  doc_id: string;
  pages_count: number;
  chunks_count: number;
  processing_time: number;
  message: string;
}

interface QueryResponse {
  answer: string;
  citations: Array<{
    doc_id: string;
    page: number;
    chunk_id: string;
    char_start: number;
    char_end: number;
  }>;
  snippets: Array<{
    page: number;
    text: string;
  }>;
  found: boolean;
  confidence?: number;
  processing_time: number;
}

function App() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocId, setSelectedDocId] = useState<string>('');
  const [queryResponse, setQueryResponse] = useState<QueryResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'upload' | 'query'>('upload');

  const handleDocumentUploaded = (doc: Document) => {
    setDocuments(prev => [...prev, doc]);
    setSelectedDocId(doc.doc_id);
    setActiveTab('query');
  };

  const handleQueryResponse = (response: QueryResponse) => {
    setQueryResponse(response);
  };

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-text">
              <h1>Debt Agreement Analysis</h1>
              <p>Intelligent Document Analysis</p>
            </div>
          </div>
          <nav className="header-nav">
            <button 
              className={`nav-button ${activeTab === 'upload' ? 'active' : ''}`}
              onClick={() => setActiveTab('upload')}
            >
              Upload Documents
            </button>
            <button 
              className={`nav-button ${activeTab === 'query' ? 'active' : ''}`}
              onClick={() => setActiveTab('query')}
              disabled={documents.length === 0}
            >
              Ask Questions
            </button>
          </nav>
        </div>
      </header>

      <main className="app-main">
        <div className="main-container">
          {activeTab === 'upload' && (
            <div className="content-section">
              <div className="section-header">
                <h2>Upload Documents</h2>
                <p>Upload PDF documents to create an intelligent knowledge base</p>
              </div>
              <DocumentUpload onDocumentUploaded={handleDocumentUploaded} />
              
              {documents.length > 0 && (
                <div className="documents-list">
                  <h3>Uploaded Documents</h3>
                  <div className="documents-grid">
                    {documents.map((doc) => (
                      <div key={doc.doc_id} className="document-card">
                        <div className="document-info">
                          <h4>{doc.doc_id}</h4>
                          <div className="document-stats">
                            <span className="stat">{doc.pages_count} pages</span>
                            <span className="stat">{doc.chunks_count} chunks</span>
                          </div>
                          <div className="processing-time">
                            Processed in {doc.processing_time.toFixed(2)}s
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'query' && documents.length > 0 && (
            <div className="content-section">
              <div className="section-header">
                <h2>Ask Questions</h2>
                <p>Query your documents with natural language questions</p>
              </div>
              
              <div className="document-selector-section">
                <label htmlFor="document-select" className="selector-label">
                  Select Document to Query
                </label>
                <select 
                  id="document-select"
                  value={selectedDocId} 
                  onChange={(e) => setSelectedDocId(e.target.value)}
                  className="document-select"
                >
                  <option value="">Choose a document...</option>
                  {documents.map((doc) => (
                    <option key={doc.doc_id} value={doc.doc_id}>
                      {doc.doc_id} ({doc.pages_count} pages, {doc.chunks_count} chunks)
                    </option>
                  ))}
                </select>
              </div>

              {selectedDocId && (
                <DocumentQuery 
                  docId={selectedDocId} 
                  onQueryResponse={handleQueryResponse}
                />
              )}

              {queryResponse && (
                <div className="query-results">
                  <div className="results-header">
                    <h3>Query Results</h3>
                    <div className="result-meta">
                      <span className={`status-badge ${queryResponse.found ? 'found' : 'not-found'}`}>
                        {queryResponse.found ? 'Answer Found' : 'No Answer Found'}
                      </span>
                      {queryResponse.confidence && (
                        <span className="confidence-badge">
                          {Math.round(queryResponse.confidence * 100)}% Confidence
                        </span>
                      )}
                      <span className="time-badge">
                        {queryResponse.processing_time.toFixed(2)}s
                      </span>
                    </div>
                  </div>
                  
                  <div className="answer-section">
                    <h4>Answer</h4>
                    <div className="answer-content">
                      {queryResponse.answer}
                    </div>
                  </div>
                  
                  {queryResponse.citations.length > 0 && (
                    <div className="citations-section">
                      <h4>Sources</h4>
                      <div className="citations-list">
                        {queryResponse.citations.map((citation, index) => (
                          <div key={index} className="citation-item">
                            <div className="citation-header">
                              <span className="citation-doc">{citation.doc_id}</span>
                              <span className="citation-page">Page {citation.page}</span>
                            </div>
                            <div className="citation-chunk">Chunk: {citation.chunk_id}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {activeTab === 'query' && documents.length === 0 && (
            <div className="empty-state">
              <h3>No Documents Available</h3>
              <p>Upload some documents first to start asking questions</p>
              <button 
                className="cta-button"
                onClick={() => setActiveTab('upload')}
              >
                Upload Documents
              </button>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;