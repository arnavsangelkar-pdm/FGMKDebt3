"""
Answer generation with citations using OpenAI.
Generates grounded answers with proper citation formatting.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from openai import OpenAI

from .config import settings
from .models import Citation, Snippet


logger = logging.getLogger(__name__)


@dataclass
class AnswerResult:
    """Result of answer generation with citations."""
    answer: str
    citations: List[Citation]
    snippets: List[Snippet]
    found: bool
    confidence: Optional[float] = None


class AnswerGenerator:
    """Answer generator with citation support."""
    
    def __init__(self, openai_client: OpenAI):
        """
        Initialize the answer generator.
        
        Args:
            openai_client: OpenAI client for answer generation
        """
        self.openai_client = openai_client
        self.logger = logger
        self.model = settings.OPENAI_MODEL
    
    def generate_answer(self, question: str, retrieved_results: List[Dict[str, Any]], doc_id: str) -> AnswerResult:
        """
        Generate an answer with citations from retrieved results.
        
        Args:
            question: User question
            retrieved_results: List of retrieved chunks with metadata
            doc_id: Document identifier
            
        Returns:
            AnswerResult with answer, citations, and snippets
        """
        if not retrieved_results:
            self.logger.warning(f"No retrieved results for question: {question}")
            return AnswerResult(
                answer="Not found in document.",
                citations=[],
                snippets=[],
                found=False,
                confidence=0.0
            )
        
        self.logger.info(f"Generating answer for question: {question}, results_count={len(retrieved_results)}")
        
        try:
            # Prepare snippets for the prompt
            snippets = self._prepare_snippets(retrieved_results)
            
            # Generate answer using OpenAI
            answer_text = self._call_openai(question, snippets)
            
            # Extract citations from the answer
            citations = self._extract_citations(answer_text, retrieved_results, doc_id)
            
            # Create snippet objects
            snippet_objects = [
                Snippet(page=result["page"], text=result["text"])
                for result in retrieved_results
            ]
            
            # Determine confidence from the best result
            confidence = retrieved_results[0].get("confidence", 0.0) if retrieved_results else 0.0
            
            result = AnswerResult(
                answer=answer_text,
                citations=citations,
                snippets=snippet_objects,
                found=True,
                confidence=confidence
            )
            
            self.logger.info(f"Answer generation completed, question={question}, answer_length={len(answer_text)}, citations_count={len(citations)}, confidence={confidence}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate answer: {str(e)}", exc_info=True)
            return AnswerResult(
                answer="Not found in document.",
                citations=[],
                snippets=[],
                found=False,
                confidence=0.0
            )
    
    def _prepare_snippets(self, retrieved_results: List[Dict[str, Any]]) -> str:
        """
        Prepare snippets for the OpenAI prompt.
        
        Args:
            retrieved_results: List of retrieved chunks
            
        Returns:
            Formatted snippets string
        """
        formatted_snippets = []
        
        for i, result in enumerate(retrieved_results, 1):
            page = result["page"]
            text = result["text"]
            snippet = f"Snippet #{i} (Doc: p. {page}): {text}"
            formatted_snippets.append(snippet)
        
        return "\n\n".join(formatted_snippets)
    
    def _call_openai(self, question: str, formatted_snippets: str) -> str:
        """
        Call OpenAI API to generate the answer.
        
        Args:
            question: User question
            formatted_snippets: Formatted snippets for context
            
        Returns:
            Generated answer text
        """
        system_prompt = """You are a document QA assistant. You must use ONLY the provided snippets from the user's document to answer.
- If the answer is not clearly supported by the snippets, reply exactly: "Not found in document."
- For each sentence that uses evidence, add an inline citation formatted as [Doc: p. <page>].
- Keep answers concise (1–5 sentences) and faithful to the source.
- Do not speculate or use external knowledge."""

        user_prompt = f"""Question: "{question}"

You are given {len(formatted_snippets.split('Snippet #')) - 1} snippets. Each snippet includes the source page.
Snippets:
{formatted_snippets}

Instructions:
- Answer only from the snippets.
- Add [Doc: p. <page>] after each sentence that uses evidence.
- If insufficient evidence: "Not found in document." """

        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, factual responses
                max_tokens=500,   # Limit response length
            )
            
            answer = response.choices[0].message.content.strip()
            return answer
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {str(e)}", exc_info=True)
            raise
    
    def _extract_citations(self, answer_text: str, retrieved_results: List[Dict[str, Any]], doc_id: str) -> List[Citation]:
        """
        Extract citations from the generated answer.
        
        Args:
            answer_text: Generated answer text
            retrieved_results: List of retrieved chunks
            doc_id: Document identifier
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        # Create a mapping of page numbers to result metadata
        page_to_result = {}
        for result in retrieved_results:
            page = result["page"]
            if page not in page_to_result:
                page_to_result[page] = result
        
        # Find citation patterns in the answer
        import re
        citation_pattern = r'\[Doc: p\. (\d+)\]'
        matches = re.finditer(citation_pattern, answer_text)
        
        for match in matches:
            page_num = int(match.group(1))
            
            if page_num in page_to_result:
                result = page_to_result[page_num]
                citation = Citation(
                    doc_id=doc_id,
                    page=page_num,
                    chunk_id=result["chunk_id"],
                    char_start=result["char_start"],
                    char_end=result["char_end"]
                )
                citations.append(citation)
        
        # Remove duplicate citations
        unique_citations = []
        seen = set()
        for citation in citations:
            key = (citation.doc_id, citation.page, citation.chunk_id)
            if key not in seen:
                seen.add(key)
                unique_citations.append(citation)
        
        return unique_citations
    
    def validate_answer(self, answer: str) -> bool:
        """
        Validate that the answer meets quality criteria.
        
        Args:
            answer: Generated answer text
            
        Returns:
            True if answer is valid, False otherwise
        """
        if not answer or answer.strip() == "":
            return False
        
        # Check if answer is the refusal response
        if answer.strip() == "Not found in document.":
            return True
        
        # Check for reasonable length (not too short, not too long)
        if len(answer) < 10 or len(answer) > 2000:
            return False
        
        # Check for citation format
        import re
        citation_pattern = r'\[Doc: p\. \d+\]'
        citations = re.findall(citation_pattern, answer)
        
        # If answer contains claims, it should have citations
        if len(answer) > 50 and not citations:
            self.logger.warning("Answer contains claims but no citations found")
            return False
        
        return True
