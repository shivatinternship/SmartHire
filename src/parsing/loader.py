"""Document loader for PDF and DOCX files."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_pdf(file_path: str) -> Optional[str]:
    """Extract text from a PDF file using PyMuPDF.

    Args:
        file_path: Path to the PDF file.

    Returns:
        Extracted text content or None if failed.
    """
    try:
        import fitz
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        logger.info(f"Successfully loaded PDF: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error loading PDF {file_path}: {e}")
        return None


def load_docx(file_path: str) -> Optional[str]:
    """Extract text from a DOCX file using python-docx.

    Args:
        file_path: Path to the DOCX file.

    Returns:
        Extracted text content or None if failed.
    """
    try:
        from docx import Document
        doc = Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        logger.info(f"Successfully loaded DOCX: {file_path}")
        return text
    except Exception as e:
        logger.error(f"Error loading DOCX {file_path}: {e}")
        return None


def load_document(file_path: str) -> Optional[str]:
    """Load a document and extract text based on file extension.

    Args:
        file_path: Path to the document file.

    Returns:
        Extracted text content or None if failed.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)
    elif suffix == ".docx":
        return load_docx(file_path)
    else:
        logger.error(f"Unsupported file format: {suffix}")
        return None
