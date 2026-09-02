"""
FastAPI entrypoint for the RAG PDF Q&A Assistant.

Endpoints:
  GET  /                -> health check
  POST /upload           -> upload + index a PDF
  POST /ask               -> ask a question over indexed documents
  GET  /stats             -> number of chunks currently indexed
  POST /reset              -> wipe the vector store
"""
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.config import UPLOAD_DIR
from app.ingestion import process_pdf
from app.embeddings import add_chunks_to_store, query_store, get_indexed_document_count, reset_store
from app.generation import generate_answer

app = FastAPI(
    title="RAG PDF Q&A Assistant",
    description="Upload PDFs and ask questions answered with retrieval-augmented generation.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None


@app.get("/")
def health_check():
    return {"status": "RAG PDF Assistant is running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = process_pdf(save_path, source_name=file.filename)
    num_stored = add_chunks_to_store(chunks)

    return {
        "filename": file.filename,
        "chunks_indexed": num_stored,
        "message": f"Indexed {num_stored} chunks from {file.filename}",
    }


@app.post("/ask")
async def ask_question(payload: AskRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    top_k = payload.top_k or None
    retrieved = query_store(payload.question, top_k=top_k) if top_k else query_store(payload.question)
    result = generate_answer(payload.question, retrieved)

    return {
        "question": payload.question,
        "answer": result["answer"],
        "sources": result["sources"],
    }


@app.get("/stats")
def stats():
    return {"indexed_chunks": get_indexed_document_count()}


@app.post("/reset")
def reset():
    reset_store()
    return {"message": "Vector store cleared."}
