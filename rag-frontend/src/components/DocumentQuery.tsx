import React, { useState } from 'react';
import axios from 'axios';

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

interface DocumentQueryProps {
  docId: string;
  onQueryResponse: (response: QueryResponse) => void;
}

const DocumentQuery: React.FC<DocumentQueryProps> = ({ docId, onQueryResponse }) => {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const response = await axios.post(`${process.env.REACT_APP_API_URL || ''}/query`, {
        doc_id: docId,
        question: question.trim(),
        k: 5
      });

      onQueryResponse(response.data);
      setQuestion(''); // Clear form after successful query
    } catch (err: any) {
      if (err.response) {
        setError(`Query failed: ${err.response.data.detail || err.response.data.error || 'Unknown error'}`);
      } else if (err.request) {
        setError('Query failed: Could not connect to server');
      } else {
        setError(`Query failed: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-container">
      <form onSubmit={handleSubmit} className="query-form">
        <div className="form-group">
          <label htmlFor="question">Ask a question about the document</label>
          <textarea
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g., What are the main findings? What is the conclusion? What methods were used?"
            className="query-input"
            rows={4}
            disabled={loading}
          />
          <div className="form-help">
            Ask specific questions to get detailed answers from your document
          </div>
        </div>
        
        <div className="query-actions">
          <button
            type="submit"
            className="query-button"
            disabled={loading || !question.trim()}
          >
            {loading ? (
              <>
                <div className="button-spinner"></div>
                Processing...
              </>
            ) : (
              'Ask Question'
            )}
          </button>
          
          {question.trim() && (
            <button
              type="button"
              className="clear-button"
              onClick={() => setQuestion('')}
              disabled={loading}
            >
              Clear
            </button>
          )}
        </div>
      </form>

      {error && (
        <div className="error">
          <div className="error-content">
            <strong>Query Failed</strong>
            <p>{error}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentQuery;
