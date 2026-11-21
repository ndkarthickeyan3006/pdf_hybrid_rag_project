from pathlib import Path
from dotenv import load_dotenv
import os

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(ENV_FILE, override=True)

print("ENV file found:", ENV_FILE.exists())
print("API key configured:", bool(os.getenv("OPENAI_API_KEY")))
import uuid
from typing import Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .schemas import AskRequest, AskResponse, Source
from .ingest import extract_pages, chunk_pages
from .retrieval import HybridIndex
from .llm import generate_answer, FALLBACK



app = FastAPI(title="PDF-Only Hybrid RAG", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
# Session-keyed in-memory document indexes. For production multi-worker deployment, replace with
# persistent storage keyed by authenticated user/session and a shared vector service.
SESSIONS: Dict[str, dict] = {}



@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...), x_session_id: str = Header(default="")):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted.")
    contents = await file.read()
    if len(contents) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"PDF exceeds {settings.max_upload_mb} MB limit.")
    if not contents.startswith(b"%PDF"):
        raise HTTPException(400, "File does not appear to be a valid PDF.")
    try:
        pages, metadata = extract_pages(contents)
    except Exception:
        raise HTTPException(400, "Could not read this PDF. It may be corrupted or encrypted.")
    empty_pages = [p["page"] for p in pages if p["is_empty"]]
    chunks = chunk_pages(pages, file.filename, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return {"status": "needs_ocr", "filename": file.filename, "pages": metadata["page_count"], "empty_pages": empty_pages, "message": "No selectable text was extracted. This may be a scanned PDF; OCR is not configured in this build."}
    try:
        index = HybridIndex(chunks)
    except Exception as exc:
        raise HTTPException(500, f"Embedding/indexing failed: {exc.__class__.__name__}")
    session_id = x_session_id or str(uuid.uuid4())
    SESSIONS[session_id] = {"filename": file.filename, "pages": pages, "metadata": metadata, "chunks": chunks, "index": index}
    return {"session_id": session_id, "status": "ready", "filename": file.filename, "page_count": metadata["page_count"], "chunk_count": len(chunks), "empty_pages": empty_pages, "metadata": metadata}



@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest, x_session_id: str = Header(default="")):
    session = SESSIONS.get(x_session_id)
    if not session:
        raise HTTPException(400, "Upload a PDF before asking questions.")
    results = session["index"].retrieve(req.question, top_k=req.top_k or settings.top_k, rrf_k=settings.rrf_k)
    if not results or max(r["cosine"] for r in results) < settings.min_relevance:
        return AskResponse(answer=FALLBACK, grounded=False, citations=[], sources=[], confidence=0)
    # Only chunks above the relevance threshold are supplied as evidence.
    evidence = [r for r in results if r["cosine"] >= settings.min_relevance]
    if not evidence:
        return AskResponse(answer=FALLBACK, grounded=False, citations=[], sources=[], confidence=0)
    try:
        answer = generate_answer(req.question, evidence)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    # If the LLM abstains, don't expose citations as if an answer was supported.
    grounded = FALLBACK not in answer
    citations = sorted(set(r["chunk"].page for r in evidence)) if grounded else []
    sources = [Source(chunk_id=r["chunk"].chunk_id, page=r["chunk"].page, text=r["chunk"].text, score=r["score"], document=r["chunk"].document, section=r["chunk"].section) for r in evidence]
    confidence = round(sum(r["cosine"] for r in evidence) / len(evidence), 3) if evidence else 0
    return AskResponse(answer=answer, grounded=grounded, citations=citations, sources=sources, confidence=confidence)



@app.get("/api/document")
def document_info(x_session_id: str = Header(default="")):
    session = SESSIONS.get(x_session_id)
    if not session:
        raise HTTPException(404, "No PDF loaded.")
    return {"filename": session["filename"], "pages": session["metadata"]["page_count"], "metadata": session["metadata"]}



@app.delete("/api/document")
def clear_document(x_session_id: str = Header(default="")):
    SESSIONS.pop(x_session_id, None)
    return {"status": "cleared"}
