#!/usr/bin/env python3
import sys
sys.path.append('.')

from rag_app.utils.parsing import PDFParser

def test_pdf_parsing():
    parser = PDFParser()
    pdf_path = "/Users/pdm/Desktop/FGMK/Wingspire Credit Agreement.pdf"
    
    try:
        print(f"Testing PDF parsing for: {pdf_path}")
        pages = parser.parse_pdf(pdf_path, "test_doc")
        print(f"Successfully parsed {len(pages)} pages")
        if pages:
            print(f"First page text length: {len(pages[0].text)}")
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pdf_parsing()
