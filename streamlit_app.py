"""
Simple Streamlit UI for the RAG PDF Q&A Assistant.
Talks to the FastAPI backend over HTTP.

Run the FastAPI backend first:
  uvicorn app.main:app --reload

Then run this:
  streamlit run streamlit_app.py
"""
import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG PDF Q&A Assistant", page_icon="📄")
st.title("📄 RAG PDF Q&A Assistant")
st.caption("Upload PDFs, then ask questions grounded in their content.")

# --- Upload section ---
st.header("1. Upload a PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    if st.button("Index this PDF"):
        with st.spinner("Extracting, chunking, and embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            resp = requests.post(f"{API_URL}/upload", files=files)
        if resp.status_code == 200:
            data = resp.json()
            st.success(data["message"])
        else:
            st.error(f"Upload failed: {resp.text}")

# --- Stats ---
try:
    stats = requests.get(f"{API_URL}/stats").json()
    st.sidebar.metric("Chunks indexed", stats["indexed_chunks"])
except requests.exceptions.ConnectionError:
    st.sidebar.warning("Backend not reachable. Is `uvicorn app.main:app --reload` running?")

if st.sidebar.button("Reset index"):
    requests.post(f"{API_URL}/reset")
    st.sidebar.success("Index cleared.")

# --- Ask section ---
st.header("2. Ask a question")
question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving and generating answer..."):
        resp = requests.post(f"{API_URL}/ask", json={"question": question})
    if resp.status_code == 200:
        data = resp.json()
        st.subheader("Answer")
        st.write(data["answer"])

        if data["sources"]:
            st.subheader("Sources")
            for s in data["sources"]:
                st.markdown(f"- {s['source']}, page {s['page']}")
    else:
        st.error(f"Request failed: {resp.text}")
