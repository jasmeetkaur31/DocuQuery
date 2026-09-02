"""
Handles answer generation via the Google Gemini API, grounded in retrieved chunks.
"""
from typing import List, Dict
from google import genai
from app.config import GEMINI_API_KEY, GEMINI_MODEL

_client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = """You are a precise document Q&A assistant. Answer the user's \
question using ONLY the information in the provided context excerpts.

Rules:
- If the context does not contain enough information to answer, say so explicitly \
— do not guess or use outside knowledge.
- Cite the source and page number for each fact you use, like (source.pdf, p.3).
- Be concise and direct. Do not pad the answer with filler.
"""


def build_context_block(chunks: List[Dict]) -> str:
    """Formats retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(
            f"[{i}] Source: {c['source']} (page {c['page']})\n{c['text']}"
        )
    return "\n\n".join(blocks)


def generate_answer(question: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Calls Gemini with the question + retrieved context.
    Returns {"answer": str, "sources": [...]}
    """
    if not retrieved_chunks:
        return {
            "answer": "I couldn't find any relevant content in the indexed documents to answer this.",
            "sources": [],
        }

    context_block = build_context_block(retrieved_chunks)

    user_message = f"""Context excerpts:
{context_block}

Question: {question}

Answer using only the context above."""

    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": 1200,
        },
    )

    answer_text = response.text or ""

    sources = [
        {"source": c["source"], "page": c["page"]} for c in retrieved_chunks
    ]

    return {"answer": answer_text, "sources": sources}
