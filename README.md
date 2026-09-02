# DocuQuery

A RAG (Retrieval-Augmented Generation) system that answers questions grounded in the content of uploaded PDFs. Built with FastAPI, ChromaDB, sentence-transformers, and the Google Gemini API.

## How it works

Upload a PDF → it's chunked and embedded into a vector store. Ask a question → relevant chunks are retrieved and passed to Gemini, which answers using only that context and cites the source page — reducing hallucination compared to general-purpose Q&A.

## Tech Stack

- **Backend:** FastAPI
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)
- **Vector store:** ChromaDB
- **Generation:** Google Gemini API (`gemini-3.6-flash`)
- **Frontend:** Streamlit
- **PDF parsing:** pypdf

## Setup

```bash
git clone <https://github.com/jasmeetkaur31/DocuQuery>
cd DocuQuery
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env       # add your GEMINI_API_KEY
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the API, or run `streamlit run streamlit_app.py` for the UI.

## Endpoints

| Method | Endpoint  | Description                |
|--------|-----------|-----------------------------|
| POST   | `/upload` | Upload and index a PDF      |
| POST   | `/ask`    | Ask a question               |
| GET    | `/stats`  | Chunks currently indexed    |
| POST   | `/reset`  | Clear the vector store       |
