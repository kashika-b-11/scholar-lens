"""
ScholarLens Document Ingestion & Parsing Engine.
Extracts, cleans, and analyzes text from PDFs and raw text papers.
"""

import io
import re
from typing import List, Tuple, Union
from pypdf import PdfReader
from .models import DocumentMetadata


class DocumentParser:
    """Handles academic paper ingestion, section tagging, and text normalization."""

    # Common academic paper section header patterns
    SECTION_PATTERNS = [
        r"(?:^|\n)\s*(?:\d+\.?\s+)?(Abstract)\b",
        r"(?:^|\n)\s*(?:\d+\.?\s+)?(Introduction)\b",
        r"(?:^|\n)\s*(?:\d+\.?\s+)?(Background|Related Work)\b",
        r"(?:^|\n)\s*(?:\d+\.?\s+)?(Methodology|Model Architecture|Methods|Approach)\b",
        r"(?:^|\n)\s*(?:\d+\.?\s+)?(Experiments|Evaluation|Results|Empirical Results)\b",
        r"(?:^|\n)\s*(?:\d+\.?\s+)?(Discussion|Limitations|Broader Impact)\b",
        r"(?:^|\n)\s*(?:\d+\.?\s+)?(Conclusion|Concluding Remarks)\b",
    ]

    @staticmethod
    def extract_from_pdf(pdf_source: Union[bytes, io.BytesIO, str]) -> str:
        """
        Extracts raw text across all pages of a PDF document.
        
        Args:
            pdf_source: Binary bytes, file stream, or filesystem path.
        """
        if isinstance(pdf_source, str):
            reader = PdfReader(pdf_source)
        elif isinstance(pdf_source, bytes):
            reader = PdfReader(io.BytesIO(pdf_source))
        else:
            reader = PdfReader(pdf_source)

        extracted_pages: List[str] = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            extracted_pages.append(page_text)

        return "\n\n".join(extracted_pages)

    @classmethod
    def clean_academic_text(cls, raw_text: str) -> str:
        """
        Normalizes academic text by repairing line breaks, hyphenated words
        across two-column layouts, and excessive whitespace.
        """
        # Repair broken hyphenations common in arXiv two-column layouts (e.g. "trans- \n former")
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", raw_text)
        
        # Replace non-breaking spaces and anomalous whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        
        # Collapse multiple empty lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        
        # Strip trailing whitespaces per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines).strip()
        
        return text

    @classmethod
    def detect_sections(cls, text: str) -> List[str]:
        """Identifies key academic structural sections detected in the text."""
        detected = []
        for pattern in cls.SECTION_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                detected.append(match.group(1).title())
        return list(dict.fromkeys(detected))  # Remove duplicates while preserving order

    @classmethod
    def parse_document(
        cls, 
        source: Union[str, bytes, io.BytesIO], 
        filename: str = "document.txt"
    ) -> Tuple[str, DocumentMetadata]:
        """
        Main entry point for document ingestion.
        
        Args:
            source: Raw string content or PDF binary stream.
            filename: Original file name.
            
        Returns:
            Tuple of (cleaned_text, DocumentMetadata)
        """
        if filename.lower().endswith(".pdf") or isinstance(source, (bytes, io.BytesIO)):
            raw_text = cls.extract_from_pdf(source)
        else:
            raw_text = str(source)

        cleaned_text = cls.clean_academic_text(raw_text)
        char_count = len(cleaned_text)
        words = cleaned_text.split()
        word_count = len(words)
        
        # Standard rough heuristic: 1 token ~= 4 characters in English
        estimated_tokens = max(1, char_count // 4)
        detected_sections = cls.detect_sections(cleaned_text)

        metadata = DocumentMetadata(
            filename=filename,
            char_count=char_count,
            word_count=word_count,
            estimated_tokens=estimated_tokens,
            detected_sections=detected_sections
        )

        return cleaned_text, metadata
