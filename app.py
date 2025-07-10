# --- Imports ---
import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
from typing import List, Tuple
import time
import fitz  # PyMuPDF for reading PDFs

# --- Page Configuration ---
st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# --- Load Embedding Model (cached) ---
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# --- Text Cleaning Function ---
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
    return text.strip()

# --- Split Cleaned Text into Chunks ---
def chunk_text(text: str, chunk_size: int = 200) -> List[str]:
    text = clean_text(text)
    sentences = re.split(r'[.!?]+', text)
    chunks, current_chunk = [], ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk + sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "."
        else:
            current_chunk += sentence + "."

    if current_chunk:
        chunks.append(current_chunk.strip())

    return [chunk for chunk in chunks if len(chunk) > 20]

# --- Process Uploaded Document: Split + Embed ---
@st.cache_data
def process_document(file_content: str) -> Tuple[List[str], np.ndarray]:
    chunks = chunk_text(file_content)
    model = load_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=True)
    return chunks, embeddings

# --- Create FAISS Vector Store ---
@st.cache_resource
def create_vector_db(_embeddings: np.ndarray):
    dimension = _embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(_embeddings.astype('float32'))
    return index

# --- Search for Most Relevant Chunks ---
def search_similar_chunks(query: str, index, chunks: List[str], k: int = 3, threshold: float = 1.0) -> List[Tuple[str, float]]:
    model = load_embedding_model()
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding.astype('float32'), k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(chunks) and dist < threshold:
            results.append((chunks[idx], float(dist)))
    return results

# --- Generate Answer from Retrieved Chunks ---
def generate_response(query: str, relevant_chunks: List[str]) -> str:
    if not relevant_chunks:
        return "I don't have enough information to answer this question based on the provided document."

    context = "\n\n".join(relevant_chunks)

    response = f"""Based on the document, here's what I found:

{context[:500]}...

This information is directly extracted from the provided document and should help answer your question about: "{query}"
"""
    return response

# --- Simulate Typing Effect ---
def simulate_streaming(text: str, placeholder):
    displayed_text = ""
    for char in text:
        displayed_text += char
        placeholder.markdown(displayed_text + "▌")
        time.sleep(0.01)
    placeholder.markdown(displayed_text)

# --- Extract Text from PDF using PyMuPDF ---
def extract_text_from_pdf(file) -> str:
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Main App Logic ---
def main():
    # --- Branding Header ---
    st.markdown("""
        <div style='text-align:center; padding:15px; background-color:#f8f9fa; color:#000000; border-radius:10px; margin-bottom:20px; border: 1px solid #ccc;'>
            <h2 style='color:#0A2647;'> AI-Powered RAG Chatbot</h2>
            <h4>Document-Based Q&A System using <strong>Embeddings</strong> + <strong>FAISS</strong></h4>
            <p style='font-size:15px;'>Assignment for <strong>Amlgo Labs</strong> | Built by <strong>Naman Jain</strong></p>
        </div>
    """, unsafe_allow_html=True)

    # --- Light Theme CSS ---
    st.markdown("""
        <style>
            .main {
                background-color: #ffffff;
                color: #000000;
            }
            .css-1d391kg, .css-1r6slb0 {
                background-color: #f4f4f4 !important;
            }
            .stChatMessage {
                background-color: #f9f9f9 !important;
                border-radius: 10px;
                padding: 10px;
                color: #000;
            }
            .stTextInput > div > input {
                color: #000000 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("Upload a document (.txt, .md, .pdf) and ask questions about it!")

    # --- Sidebar for File Upload ---
    with st.sidebar:
        st.header("📊 System Information")

        uploaded_file = st.file_uploader("Upload Document", type=['txt', 'md', 'pdf'])

        if uploaded_file is not None:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'pdf':
                file_content = extract_text_from_pdf(uploaded_file)
            else:
                file_content = uploaded_file.read().decode('utf-8')

            if 'processed_doc' not in st.session_state or st.session_state.get('current_file') != uploaded_file.name:
                with st.spinner("Processing document..."):
                    chunks, embeddings = process_document(file_content)
                    index = create_vector_db(embeddings)
                    st.session_state.chunks = chunks
                    st.session_state.embeddings = embeddings
                    st.session_state.index = index
                    st.session_state.processed_doc = True
                    st.session_state.current_file = uploaded_file.name

            st.success("✅ Document processed successfully!")
            st.info(f"📄 **File:** {uploaded_file.name}")
            st.info(f"🔢 **Chunks:** {len(st.session_state.chunks)}")
            st.info(f" **Model:** all-MiniLM-L6-v2")
        else:
            st.warning("⚠️ Please upload a document to start")
            if 'processed_doc' in st.session_state:
                del st.session_state['processed_doc']
            st.stop()

        st.divider()

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Chat Logic ---
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Source Documents"):
                    if not message["sources"]:
                        st.markdown("_No relevant chunks found._")
                    for i, (chunk, score) in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** (Relevance Score: {score:.3f})")
                        st.markdown(f"*{chunk[:300]}...*")
                        st.divider()

    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching through document..."):
                similar_chunks = search_similar_chunks(
                    prompt,
                    st.session_state.index,
                    st.session_state.chunks,
                    k=3,
                    threshold=1.0
                )
                relevant_texts = [chunk for chunk, _ in similar_chunks]
                response = generate_response(prompt, relevant_texts)

            response_placeholder = st.empty()
            simulate_streaming(response, response_placeholder)

            with st.expander("📚 Source Documents"):
                if not similar_chunks:
                    st.markdown("_No relevant chunks found._")
                for i, (chunk, score) in enumerate(similar_chunks):
                    st.markdown(f"**Source {i+1}** (Relevance Score: {score:.3f})")
                    st.markdown(f"*{chunk[:300]}...*")
                    st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": similar_chunks
            })

    st.markdown("---")
    st.markdown("💡 **Tip:** Ask specific questions about the document content for best results!")

# --- Run App ---
if __name__ == "__main__":
    main()
