import io
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncGenerator, Dict
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from pypdf import PdfReader

from chunking import fixed_chunking, paragraph_chunking
from database import init_db, save_document_metadata
from rag import ask_rag_assistant, store_chunks_in_qdrant


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    init_db()
    yield


app = FastAPI(
    title="Palm Mind AI - Document Ingestion & Conversational RAG API",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


class ChunkStrategy(str, Enum):
    fixed = "fixed"
    paragraph = "paragraph"


class ChatInput(BaseModel):
    session_id: str
    query: str


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    strategy: ChunkStrategy = Form(ChunkStrategy.paragraph),
) -> Dict[str, Any]:
    filename = file.filename or ""
    raw_bytes = await file.read()

    if filename.endswith(".txt"):
        text = raw_bytes.decode("utf-8", errors="ignore")
    elif filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(raw_bytes))
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        raise HTTPException(status_code=400, detail="Only .pdf or .txt files are allowed.")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded file contains no text.")

    chunks = fixed_chunking(text) if strategy == ChunkStrategy.fixed else paragraph_chunking(text)

    store_chunks_in_qdrant(chunks, filename)
    save_document_metadata(filename, strategy.value, len(chunks))

    return {
        "message": "Document uploaded and indexed successfully.",
        "filename": filename,
        "strategy": strategy.value,
        "chunks_created": len(chunks),
    }


@app.post("/chat")
def chat(payload: ChatInput) -> Dict[str, Any]:
    return ask_rag_assistant(session_id=payload.session_id, user_query=payload.query)

