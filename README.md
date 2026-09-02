# RAG PDF Q&A Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions grounded in the content of uploaded PDF documents. Built with FastAPI, ChromaDB, sentence-transformers, and the Google Gemini API.

## Why this project

Most Q&A demos hallucinate or answer from the model's general knowledge instead of the actual document. This assistant grounds every answer in retrieved chunks from the source PDF and cites the page number for each fact, so answers are traceable and verifiable.

## Architecture

```
PDF Upload → Text Extraction (pypdf) → Chunking (fixed-size, overlap)
    → Embedding (sentence-transformers, all-MiniLM-L6-v2)
    → Vector Storage (ChromaDB)

Question → Embedding → Similarity Search (top-k retrieval)
    → Context-grounded prompt → Gemini API → Cited answer
```

## Features

- PDF upload and automatic chunking/indexing via a REST API
- Semantic retrieval over indexed documents using vector similarity search
- Answers grounded strictly in retrieved context, with source + page citations
- Explicit "not found in documents" response when context is insufficient (reduces hallucination)
- Retrieval evaluation harness for measuring accuracy and latency
- Streamlit UI for demoing the system end-to-end

## Tech Stack

- **Backend:** FastAPI
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB (persistent, local)
- **Generation:** Google Gemini API (`gemini-2.5-flash`)
- **Frontend:** Streamlit
- **PDF parsing:** pypdf

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   git clone <your-repo-url>
   cd rag-pdf-assistant
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API key (get a free one at https://aistudio.google.com/apikey):
   ```bash
   cp .env.example .env
   # then edit .env and add your GEMINI_API_KEY
   ```

4. Run the backend:
   ```bash
   uvicorn app.main:app --reload
   ```
   Visit `http://127.0.0.1:8000/docs` for interactive API docs.

5. (Optional) Run the Streamlit UI in a second terminal:
   ```bash
   streamlit run streamlit_app.py
   ```

## API Endpoints

| Method | Endpoint  | Description                          |
|--------|-----------|---------------------------------------|
| GET    | `/`       | Health check                          |
| POST   | `/upload` | Upload and index a PDF                |
| POST   | `/ask`    | Ask a question over indexed documents |
| GET    | `/stats`  | Number of chunks currently indexed    |
| POST   | `/reset`  | Clear the vector store                |

**Example: ask a question**
```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings in section 2?"}'
```

## Evaluation

`tests/evaluate_retrieval.py` measures retrieval accuracy (was the correct source document retrieved in the top-k results?) and average retrieval latency. Fill in `EVAL_SET` with real question/expected-source pairs from your own indexed PDFs, then run:

```bash
python tests/evaluate_retrieval.py
```

This produces the kind of quantified metrics (accuracy %, latency) worth citing on a resume or in a project writeup — run it against your own test documents and record the numbers.

## Design Decisions

- **Fixed-size chunking with overlap (800 chars, 150 overlap):** simple, predictable baseline; overlap prevents losing context at chunk boundaries. Snaps to sentence boundaries where possible to avoid cutting mid-sentence.
- **all-MiniLM-L6-v2 embeddings:** fast and lightweight (384 dimensions) — good baseline for a local, low-latency demo without needing GPU inference.
- **Explicit grounding in the system prompt:** the model is instructed to say when it can't answer from context rather than filling gaps with outside knowledge, directly reducing hallucination risk.

## Possible Extensions

- Recursive/semantic chunking instead of fixed-size
- Hybrid search (keyword + semantic)
- Multi-document cross-referencing in a single answer
- Deploy backend + Streamlit UI to a public host (e.g. Streamlit Cloud + Render)

## License

MIT
