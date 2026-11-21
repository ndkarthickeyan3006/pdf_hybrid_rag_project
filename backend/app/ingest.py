import re
import uuid
from dataclasses import dataclass
from typing import List, Optional
import fitz



@dataclass
class Chunk:
    chunk_id: str
    document: str
    page: int
    text: str
    section: Optional[str] = None


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()



def extract_pages(pdf_bytes: bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for idx, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        pages.append({"page": idx, "text": text, "is_empty": not bool(text)})
    metadata = {"page_count": len(doc), "title": doc.metadata.get("title") or "", "author": doc.metadata.get("author") or ""}
    doc.close()
    return pages, metadata



def chunk_pages(pages, document_name: str, chunk_size: int = 900, overlap: int = 160) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page in pages:
        text = page["text"]
        if not text:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        section = None
        buffer = ""
        for para in paragraphs:
            if len(para) <= 140 and (para.isupper() or para.endswith(":")):
                section = para.strip().rstrip(":")
            # Break unusually long paragraphs into bounded windows so chunk sizes stay predictable.
            pieces = [para[i:i + chunk_size] for i in range(0, len(para), max(1, chunk_size - overlap))] if len(para) > chunk_size else [para]
            for piece in pieces:
                if buffer and len(buffer) + len(piece) + 2 > chunk_size:
                    chunks.append(Chunk(str(uuid.uuid4()), document_name, page["page"], buffer.strip(), section))
                    tail = buffer[-overlap:] if overlap else ""
                    buffer = (tail + "\n" + piece).strip()
                else:
                    buffer = (buffer + "\n\n" + piece).strip()
        if buffer:
            chunks.append(Chunk(str(uuid.uuid4()), document_name, page["page"], buffer.strip(), section))
    return chunks
