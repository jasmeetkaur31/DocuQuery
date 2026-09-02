"""
Handles PDF text extraction and chunking.
"""
from typing import List, Dict
from pypdf import PdfReader
from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_by_page(file_path: str) -> List[Dict]:
    """
    Extracts text from a PDF, page by page.
    Returns a list of dicts: [{"page": 1, "text": "..."}, ...]
    """
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Fixed-size chunking with overlap, splitting on character count
    but snapping to the nearest sentence/whitespace boundary where possible.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # try to snap to a sentence boundary (period + space) near the end
        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + (chunk_size // 2):
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        start = end - overlap  # step back for overlap
        if start < 0:
            start = 0

    return chunks


def process_pdf(file_path: str, source_name: str) -> List[Dict]:
    """
    Full ingestion pipeline for one PDF: extract -> chunk -> attach metadata.
    Returns a list of dicts ready to be embedded:
    [{"text": ..., "source": ..., "page": ..., "chunk_id": ...}, ...]
    """
    pages = extract_text_by_page(file_path)
    all_chunks = []
    chunk_counter = 0

    for page in pages:
        page_chunks = chunk_text(page["text"])
        for chunk in page_chunks:
            all_chunks.append({
                "text": chunk,
                "source": source_name,
                "page": page["page"],
                "chunk_id": f"{source_name}_p{page['page']}_c{chunk_counter}",
            })
            chunk_counter += 1

    return all_chunks
