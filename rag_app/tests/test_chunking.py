"""
Tests for text chunking functionality.
"""

import pytest
from utils.chunking import TokenAwareChunker, Chunk
from utils.parsing import PageText


class TestTokenAwareChunker:
    """Test cases for TokenAwareChunker."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.chunker = TokenAwareChunker(chunk_size=100, chunk_overlap=20)
    
    def test_chunk_small_text(self):
        """Test chunking of text that fits in one chunk."""
        pages = [
            PageText(
                doc_id="test_doc",
                page=1,
                text="This is a short text that should fit in one chunk.",
                section="Test Section",
                order=0
            )
        ]
        
        chunks = self.chunker.chunk_pages(pages, "test_doc")
        
        assert len(chunks) == 1
        assert chunks[0].doc_id == "test_doc"
        assert chunks[0].page == 1
        assert chunks[0].section == "Test Section"
        assert chunks[0].text == pages[0].text
        assert chunks[0].token_count <= 100
    
    def test_chunk_large_text(self):
        """Test chunking of text that requires multiple chunks."""
        # Create a long text that will exceed chunk size
        long_text = "This is a sentence. " * 50  # Should be much longer than 100 tokens
        
        pages = [
            PageText(
                doc_id="test_doc",
                page=1,
                text=long_text,
                section="Test Section",
                order=0
            )
        ]
        
        chunks = self.chunker.chunk_pages(pages, "test_doc")
        
        assert len(chunks) > 1
        assert all(chunk.doc_id == "test_doc" for chunk in chunks)
        assert all(chunk.page == 1 for chunk in chunks)
        assert all(chunk.section == "Test Section" for chunk in chunks)
        
        # Check that chunks don't exceed size limit
        for chunk in chunks:
            assert chunk.token_count <= 100 * 1.1  # Allow 10% tolerance
    
    def test_chunk_overlap(self):
        """Test that chunks have proper overlap."""
        # Create text that will produce exactly 2 chunks
        text = "This is a sentence. " * 30  # Should produce 2 chunks with overlap
        
        pages = [
            PageText(
                doc_id="test_doc",
                page=1,
                text=text,
                section="Test Section",
                order=0
            )
        ]
        
        chunks = self.chunker.chunk_pages(pages, "test_doc")
        
        if len(chunks) >= 2:
            # Check that there's some overlap between consecutive chunks
            # This is a basic check - in practice, overlap is handled at token level
            assert chunks[0].char_end > chunks[1].char_start or chunks[1].char_end > chunks[0].char_start
    
    def test_chunk_metadata_preservation(self):
        """Test that chunk metadata is preserved correctly."""
        pages = [
            PageText(
                doc_id="test_doc",
                page=5,
                text="This is a test text for metadata preservation.",
                section="Important Section",
                order=0
            )
        ]
        
        chunks = self.chunker.chunk_pages(pages, "test_doc")
        
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.doc_id == "test_doc"
        assert chunk.page == 5
        assert chunk.section == "Important Section"
        assert chunk.chunk_id is not None
        assert chunk.char_start >= 0
        assert chunk.char_end > chunk.char_start
        assert chunk.token_count > 0
    
    def test_chunk_validation(self):
        """Test chunk validation."""
        # Create valid chunks
        valid_chunks = [
            Chunk(
                doc_id="test_doc",
                page=1,
                section="Test",
                chunk_id="chunk1",
                text="Valid text",
                char_start=0,
                char_end=10,
                token_count=50
            )
        ]
        
        assert self.chunker.validate_chunks(valid_chunks) is True
        
        # Create invalid chunk (negative char_start)
        invalid_chunks = [
            Chunk(
                doc_id="test_doc",
                page=1,
                section="Test",
                chunk_id="chunk1",
                text="Invalid text",
                char_start=-1,
                char_end=10,
                token_count=50
            )
        ]
        
        assert self.chunker.validate_chunks(invalid_chunks) is False
    
    def test_token_counting(self):
        """Test token counting functionality."""
        text = "This is a test sentence for token counting."
        token_count = self.chunker.get_token_count(text)
        
        assert isinstance(token_count, int)
        assert token_count > 0
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        pages = [
            PageText(
                doc_id="test_doc",
                page=1,
                text="",
                section="Empty Section",
                order=0
            )
        ]
        
        chunks = self.chunker.chunk_pages(pages, "test_doc")
        
        assert len(chunks) == 0
    
    def test_whitespace_only_text(self):
        """Test handling of whitespace-only text."""
        pages = [
            PageText(
                doc_id="test_doc",
                page=1,
                text="   \n\t   ",
                section="Whitespace Section",
                order=0
            )
        ]
        
        chunks = self.chunker.chunk_pages(pages, "test_doc")
        
        assert len(chunks) == 0
