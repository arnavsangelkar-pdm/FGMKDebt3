"""
PDF parsing utilities using PyMuPDF (fitz).
Extracts text from PDFs with page-level metadata and section detection.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import fitz  # PyMuPDF


logger = logging.getLogger(__name__)


@dataclass
class PageText:
    """Represents text extracted from a PDF page."""
    doc_id: str
    page: int
    text: str
    section: Optional[str] = None
    order: int = 0


class PDFParser:
    """PDF parser using PyMuPDF for text extraction."""
    
    def __init__(self):
        """Initialize the PDF parser."""
        self.logger = logger
    
    def parse_pdf(self, pdf_path: Path, doc_id: str) -> List[PageText]:
        """
        Parse a PDF file and extract text with page-level metadata.
        
        Args:
            pdf_path: Path to the PDF file
            doc_id: Document identifier
            
        Returns:
            List of PageText objects containing extracted text and metadata
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If PDF parsing fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Starting PDF parsing for {doc_id}, pdf_path={str(pdf_path)}")
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = self._extract_page_text(page, doc_id, page_num)
                if page_text:
                    pages.append(page_text)
            
            doc.close()
            
            total_chars = sum(len(p.text) for p in pages)
            self.logger.info(f"PDF parsing completed for {doc_id}, pages_count={len(pages)}, total_characters={total_chars}")
            
            return pages
            
        except Exception as e:
            self.logger.error(f"Failed to parse PDF {pdf_path}: {str(e)}", exc_info=True)
            raise
    
    def _extract_page_text(self, page: fitz.Page, doc_id: str, page_num: int) -> Optional[PageText]:
        """
        Extract text from a single PDF page.
        
        Args:
            page: PyMuPDF page object
            doc_id: Document identifier
            page_num: Page number (0-indexed)
            
        Returns:
            PageText object or None if no text found
        """
        try:
            # Extract text blocks in reading order
            text_dict = page.get_text("dict")
            text_blocks = []
            
            for block in text_dict["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    for line in block["lines"]:
                        line_text = ""
                        for span in line["spans"]:
                            line_text += span["text"]
                        if line_text.strip():
                            block_text += line_text + "\n"
                    
                    if block_text.strip():
                        text_blocks.append(block_text.strip())
            
            # Combine all text blocks
            full_text = "\n\n".join(text_blocks)
            
            if not full_text.strip():
                self.logger.warning(f"No text found on page {page_num + 1}")
                return None
            
            # Detect section/heading (simple heuristic based on font size)
            section = self._detect_section(text_dict)
            
            return PageText(
                doc_id=doc_id,
                page=page_num + 1,  # Convert to 1-indexed
                text=full_text,
                section=section,
                order=page_num
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract text from page {page_num + 1}: {str(e)}", exc_info=True)
            return None
    
    def _detect_section(self, text_dict: dict) -> Optional[str]:
        """
        Detect section/heading from text blocks using font size heuristic.
        
        Args:
            text_dict: PyMuPDF text dictionary for a page
            
        Returns:
            Section name if detected, None otherwise
        """
        try:
            # Find the largest font size text (likely a heading)
            max_font_size = 0
            section_text = None
            
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_size = span["size"]
                            text = span["text"].strip()
                            
                            # Consider it a heading if:
                            # 1. Font size is larger than average
                            # 2. Text is relatively short (likely a title)
                            # 3. Text is not empty
                            if (font_size > max_font_size and 
                                len(text) < 100 and 
                                text and
                                not text.isdigit()):  # Skip page numbers
                                
                                max_font_size = font_size
                                section_text = text
            
            # Only return section if it seems like a meaningful heading
            if section_text and max_font_size > 10:  # Minimum font size threshold
                return section_text
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to detect section: {str(e)}")
            return None
    
    def get_pdf_info(self, pdf_path: Path) -> dict:
        """
        Get basic information about a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with PDF metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            page_count = len(doc)
            doc.close()
            
            return {
                "page_count": page_count,
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "modification_date": metadata.get("modDate", ""),
                "file_size": pdf_path.stat().st_size
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get PDF info for {pdf_path}: {str(e)}", exc_info=True)
            raise
