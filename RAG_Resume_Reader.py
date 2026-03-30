# app.py
import streamlit as st
import PyPDF2
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# ---------------------------
# Groq API settings
# ---------------------------

GROQ_API_KEY = "Insert_api_key"  
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"


# ---------------------------
# Embedding model (local)
# ---------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def local_embedding(texts):
    return embedder.encode(texts, convert_to_numpy=True).tolist()


# ---------------------------
# Groq Chat
# ---------------------------
def groq_chat(messages, max_tokens=300):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": GROQ_CHAT_MODEL, "messages": messages, "max_tokens": max_tokens}
    resp = requests.post(url, headers=headers, json=payload)
    return resp.json()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="PDF Chat UI", layout="wide")
st.title("📄 Chat with Your PDFs (UI Demo)")

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_model()

def get_embeddings(texts):
    return embedder.encode(texts, convert_to_numpy=True)

# ---------------------------
# Sidebar - Upload PDFs
# ---------------------------
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# ---------------------------
# Session State
# ---------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []

# ---------------------------
# Process PDFs
# ---------------------------
if uploaded_files:
    st.sidebar.info("Processing PDFs...")

    all_chunks = []

    for file in uploaded_files:
        reader = PyPDF2.PdfReader(file)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        # Chunking
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        all_chunks.extend(chunks)

    if all_chunks:
        embeddings = get_embeddings(all_chunks)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings.astype("float32"))

        st.session_state.index = index
        st.session_state.chunks = all_chunks

        st.sidebar.success(f"{len(all_chunks)} chunks indexed")

# ---------------------------
# Layout
# ---------------------------
col1, col2 = st.columns([1, 2])

# Left: File info
with col1:
    st.subheader("📂 Files")
    if uploaded_files:
        for f in uploaded_files:
            st.write(f"📄 {f.name}")
    else:
        st.write("No files uploaded")

# Right: Chat UI
with col2:
    st.subheader("💬 Ask Questions")

    query = st.text_input("Enter your question:")

    if query and st.session_state.index:
        # Embed query
        q_emb = get_embeddings([query]).astype("float32")

        # Search
        D, I = st.session_state.index.search(q_emb, k=3)

        results = [st.session_state.chunks[i] for i in I[0]]

        st.write("### 🔍 Top Matches:")
        for i, r in enumerate(results):
            st.write(f"**Chunk {i+1}:**")
            st.write(r[:300] + "...")

        # Dummy answer (UI only)
        st.write("### 🤖 Answer:")
        st.success("This is a demo response based on retrieved chunks.")

    elif query:
        st.warning("Upload PDF first!")
