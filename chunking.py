from typing import List


def fixed_chunking(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Strategy 1: Splits text into fixed-size character chunks with overlap."""
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def paragraph_chunking(text: str, max_length: int = 800) -> List[str]:
    """Strategy 2: Splits text along paragraph boundaries to preserve context."""
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) <= max_length:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks
