import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

interface Document {
  doc_id: string;
  pages_count: number;
  chunks_count: number;
  processing_time: number;
  message: string;
}

interface DocumentUploadProps {
  onDocumentUploaded: (doc: Document) => void;
}

const DocumentUpload: React.FC<DocumentUploadProps> = ({ onDocumentUploaded }) => {
  const [uploading, setUploading] = useState(false);
  const [docId, setDocId] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;
    
    const file = acceptedFiles[0];
    
    // Validate file type
    if (file.type !== 'application/pdf') {
      setError('Please upload a PDF file');
      return;
    }

    // Validate file size (250MB)
    const maxSize = 250 * 1024 * 1024; // 250MB in bytes
    if (file.size > maxSize) {
      setError('File size must be less than 250MB');
      return;
    }

    if (!docId.trim()) {
      setError('Please enter a document ID');
      return;
    }

    setUploading(true);
    setError('');
    setMessage('');

    try {
      const formData = new FormData();
      formData.append('doc_id', docId);
      formData.append('file', file);

      const response = await axios.post(`${process.env.REACT_APP_API_URL || ''}/ingest`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 300000, // 5 minutes timeout for large files
      });

      onDocumentUploaded(response.data);
      setMessage(`Document uploaded successfully! ${response.data.message}`);
      setDocId(''); // Reset form
    } catch (err: any) {
      if (err.response) {
        setError(`Upload failed: ${err.response.data.detail || err.response.data.error || 'Unknown error'}`);
      } else if (err.request) {
        setError('Upload failed: Could not connect to server');
      } else {
        setError(`Upload failed: ${err.message}`);
      }
    } finally {
      setUploading(false);
    }
  }, [docId, onDocumentUploaded]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf']
    },
    maxFiles: 1,
    disabled: uploading
  });

  return (
    <div className="upload-container">
      <div className="upload-form">
        <div className="form-group">
          <label htmlFor="docId">Document ID</label>
          <input
            id="docId"
            type="text"
            value={docId}
            onChange={(e) => setDocId(e.target.value)}
            placeholder="Enter a unique document ID (e.g., quarterly-report-2024)"
            disabled={uploading}
            className="form-input"
          />
          <div className="form-help">
            Choose a descriptive name that will help you identify this document later
          </div>
        </div>
      </div>

      <div
        {...getRootProps()}
        className={`upload-area ${isDragActive ? 'dragover' : ''} ${uploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />
        <div className="upload-content">
          <div className="upload-text">
            {uploading
              ? 'Processing document...'
              : isDragActive
              ? 'Drop the PDF file here...'
              : 'Drag & drop a PDF file here, or click to select'
            }
          </div>
          <div className="upload-subtext">
            {uploading 
              ? 'This may take a few moments for large files'
              : 'Maximum file size: 250MB • Supported format: PDF'
            }
          </div>
          {!uploading && (
            <div className="upload-features">
              <div className="feature">
                <span>AI-powered text extraction</span>
              </div>
              <div className="feature">
                <span>Fast processing</span>
              </div>
              <div className="feature">
                <span>Secure upload</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {uploading && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <div className="loading-text">Uploading and processing document...</div>
        </div>
      )}

      {error && (
        <div className="error">
          <div className="error-content">
            <strong>Upload Failed</strong>
            <p>{error}</p>
          </div>
        </div>
      )}

      {message && (
        <div className="success">
          <div className="success-content">
            <strong>Upload Successful!</strong>
            <p>{message}</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentUpload;
