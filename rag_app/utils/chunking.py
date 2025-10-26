"""
Token-aware text chunking using tiktoken.
Splits text into overlapping chunks while preserving metadata.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional

import tiktoken


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    doc_id: str
    page: int
    section: Optional[str]
    chunk_id: str
    text: str
    char_start: int
    char_end: int
    token_count: int


class TokenAwareChunker:
    """Token-aware text chunker using tiktoken."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, model_name: str = "gpt-4"):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target number of tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            model_name: OpenAI model name for tokenizer
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logger
        
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            self.encoding = tiktoken.get_encoding("cl100k_base")
            self.logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
    
    def chunk_pages(self, pages: List, doc_id: str) -> List[Chunk]:
        """
        Chunk a list of pages into overlapping text chunks.
        
        Args:
            pages: List of PageText objects
            doc_id: Document identifier
            
        Returns:
            List of Chunk objects
        """
        self.logger.info(f"Starting chunking for {doc_id}, pages_count={len(pages)}")
        
        all_chunks = []
        char_offset = 0
        
        for page in pages:
            page_chunks = self._chunk_page_text(page, char_offset)
            all_chunks.extend(page_chunks)
            char_offset += len(page.text)
        
        avg_chunk_size = sum(c.token_count for c in all_chunks) / len(all_chunks) if all_chunks else 0
        self.logger.info(f"Chunking completed for {doc_id}, chunks_count={len(all_chunks)}, avg_chunk_size={avg_chunk_size}")
        
        return all_chunks
    
    def _chunk_page_text(self, page, char_offset: int) -> List[Chunk]:
        """
        Chunk text from a single page.
        
        Args:
            page: PageText object
            char_offset: Character offset from start of document
            
        Returns:
            List of Chunk objects for this page
        """
        if not page.text.strip():
            return []
        
        # Tokenize the text
        tokens = self.encoding.encode(page.text)
        
        if len(tokens) <= self.chunk_size:
            # Text fits in one chunk
            chunk_id = str(uuid.uuid4())
            return [Chunk(
                doc_id=page.doc_id,
                page=page.page,
                section=page.section,
                chunk_id=chunk_id,
                text=page.text,
                char_start=char_offset,
                char_end=char_offset + len(page.text),
                token_count=len(tokens)
            )]
        
        # Split into overlapping chunks
        chunks = []
        start_idx = 0
        
        while start_idx < len(tokens):
            # Determine end index for this chunk
            end_idx = min(start_idx + self.chunk_size, len(tokens))
            
            # Decode tokens back to text
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Calculate character positions
            chunk_char_start = char_offset + self._get_char_position(page.text, tokens, start_idx)
            chunk_char_end = char_offset + self._get_char_position(page.text, tokens, end_idx)
            
            # Create chunk
            chunk_id = str(uuid.uuid4())
            chunk = Chunk(
                doc_id=page.doc_id,
                page=page.page,
                section=page.section,
                chunk_id=chunk_id,
                text=chunk_text,
                char_start=chunk_char_start,
                char_end=chunk_char_end,
                token_count=len(chunk_tokens)
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            if end_idx >= len(tokens):
                break
            
            start_idx = end_idx - self.chunk_overlap
            if start_idx < 0:
                start_idx = 0
        
        return chunks
    
    def _get_char_position(self, text: str, tokens: List[int], token_idx: int) -> int:
        """
        Get character position for a given token index.
        
        Args:
            text: Original text
            tokens: List of token IDs
            token_idx: Token index
            
        Returns:
            Character position in the text
        """
        if token_idx <= 0:
            return 0
        if token_idx >= len(tokens):
            return len(text)
        
        # Decode tokens up to the given index
        partial_tokens = tokens[:token_idx]
        partial_text = self.encoding.decode(partial_tokens)
        
        # Find the position in the original text
        # This is an approximation - for exact positioning, we'd need more complex logic
        return min(len(partial_text), len(text))
    
    def get_token_count(self, text: str) -> int:
        """
        Get the number of tokens in a text string.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))
    
    def validate_chunks(self, chunks: List[Chunk]) -> bool:
        """
        Validate that chunks meet the expected criteria.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            True if all chunks are valid, False otherwise
        """
        if not chunks:
            return True
        
        for chunk in chunks:
            # Check token count is within reasonable bounds
            if chunk.token_count > self.chunk_size * 1.1:  # Allow 10% tolerance
                self.logger.warning(f"Chunk {chunk.chunk_id} exceeds expected size, token_count={chunk.token_count}, expected_max={self.chunk_size}")
                return False
            
            # Check character positions are valid
            if chunk.char_start < 0 or chunk.char_end <= chunk.char_start:
                self.logger.warning(f"Chunk {chunk.chunk_id} has invalid character positions, char_start={chunk.char_start}, char_end={chunk.char_end}")
                return False
        
        return True
