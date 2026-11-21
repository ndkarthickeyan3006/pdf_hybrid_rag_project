# Papertrail — PDF-Only Hybrid RAG

A PDF-grounded question-answering app with a React/TypeScript UI and FastAPI backend. It uses page-aware chunking, Sentence Transformers + FAISS semantic retrieval, BM25 lexical retrieval, Reciprocal Rank Fusion, and a configurable OpenAI chat model. No web search or external retrieval is implemented.


## Features
- PDF upload validation, page extraction and metadata
- Detects pages with no extractable text; scanned-PDF OCR is not included yet
- Paragraph-aware page chunks with overlap and page metadata
- Semantic retrieval (normalized embeddings + FAISS inner-product search)
- BM25 lexical retrieval + RRF ranking
- Grounding prompt, citations and explicit fallback
- Expandable retrieved evidence source snippets
- Clear document and conversation controls
- Docker Compose development deployment


## Quick start (local)

### Backend
```bash
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in backend/.env
uvicorn app.main:app --reload --port 8000
```

The first request/upload may download the configured Sentence Transformers model from its model registry. After the model is cached, embedding is local. Set `EMBEDDING_MODEL` to another compatible sentence-transformer model if desired.


### Frontend
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173. Set `VITE_API_BASE` if backend runs elsewhere (default http://localhost:8000).


## Docker
```bash
cp backend/.env.example backend/.env
# edit backend/.env with OPENAI_API_KEY
 docker compose up --build
```
- Frontend: http://localhost:5173
- API docs: http://localhost:8000/docs


## Configuration
All settings are environment variables; see `backend/.env.example`.\n- `CHUNK_SIZE`, `CHUNK_OVERLAP`: page-local chunk parameters
- `TOP_K`: number of fused chunks returned
- `RRF_K`: RRF smoothing constant
- `MIN_RELEVANCE`: cosine threshold for abstention (tune against your own test corpus)
- `MAX_CONTEXT_CHARS`: cap evidence context passed to LLM
- `MAX_UPLOAD_MB`: upload size limit
- `LLM_MODEL`: OpenAI chat model name
- `EMBEDDING_MODEL`: Sentence Transformer model name


## API
- `GET /health`
- `POST /api/upload` — multipart PDF file
- `POST /api/ask` — JSON `{ "question": "...", "top_k": 6 }`; pass session ID in `X-Session-Id`
- `GET /api/document` — metadata for current session
- `DELETE /api/document` — clear session document index


## Testing
```bash
cd backend
pytest
```
A starter test suite should cover PDF parsing/chunking, retrieval behavior, answer abstention, citations, and injection-resistance. Add a test PDF in `tests/fixtures` or generate PDFs in tests.


## Production considerations / limitations
- **Session isolation:** current indexes are held in process memory, and the session ID is not authentication. Do not expose this build to untrusted users as-is. Production deployment should use authenticated sessions, shared durable storage, encrypted document storage, quotas, TTL cleanup, and tenant-bound index lookups.
- **Grounding:** prompting and similarity threshold reduce unsupported answers, but LLM text is not mechanically proof of support. For production-grade claim verification, add an entailment/claim-check stage or require quote-level evidence for every sentence and abstain when claims cannot be verified.
- **Reranking:** optional reranker is not included in this starter.
- **OCR:** scanned PDF detection is present (empty-text pages); OCR is not yet implemented.
- **PDF preview:** not implemented in this starter UI; citations and extracted source snippets are shown.
- **Index lifecycle:** indexes are cleared when `/api/document` is called; process restart clears all current indexes.
- **Privacy:** avoid logging document text; the application doesn't send retrieved context to any model until a question is submitted. On query, retrieved excerpts are sent to configured LLM provider.


## Grounding prompt
The LLM is instructed to answer only from provided PDF excerpts, cite pages, point out conflicts, and ignore instructions embedded in the PDF. If evidence is inadequate it should return exactly: `I couldn't find this information in the uploaded PDF.`
