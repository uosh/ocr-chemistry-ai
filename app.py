import os
import glob
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq

import re


def clean_pdf_text(text: str) -> str:
    """Removes weird PDF artifacts, line breaks, and broken hyphens."""
    # Fix hyphenated words broken across lines (e.g. "enthal-\npy" -> "enthalpy")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Replace single line breaks with spaces, preserving paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize consecutive spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 1. Page Config
st.set_page_config(
    page_title="OCR Chemistry AI (RAG)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. System Instruction
SYSTEM_PROMPT = """You are an expert OCR A Level Chemistry AI Assistant, specialized in helping students master the OCR Chemistry specifications.

STRICT GROUNDING & EXAM RULES:
1. Base your answers primarily on the official OCR specification and mark scheme context provided below.
2. Align all definitions, key terms, and reaction mechanisms directly with official OCR guidelines.
3. Highlight required exam keywords in **bold** (e.g., **heterolytic fission**, **lone pair on nitrogen**).
4. Use LaTeX ($...$ for inline, $$...$$ for block equations) for calculations and chemical formulas.
"""

# 3. Sidebar Setup & Key Verification
st.sidebar.title("🧪 OCR Chemistry AI")

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.sidebar.error("⚠️ `GROQ_API_KEY` missing in Streamlit Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

selected_model = st.sidebar.selectbox(
    "Active Groq Model",
    options=[
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "llama-3.1-8b-instant"
    ],
    index=0
)

# 4. RAG Engine: PDF Loading & Embedding Pipeline
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    # Lightweight CPU-friendly embedding model
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

@st.cache_data(show_spinner="Processing PDF Knowledge Base...")
def index_pdf_documents(folder_path="ocr_files", chunk_size=600, overlap=100):
    """Parses all PDFs in folder_path, chunks text, and creates embeddings."""
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    
    # Fallback to root directory if folder doesn't exist
    if not pdf_files:
        pdf_files = glob.glob("*.pdf")

    if not pdf_files:
        return None, []

    chunks = []
    
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        try:
            reader = PdfReader(pdf_path)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                
                # Simple character-level sliding window chunking
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        chunks.append({
                            "source": f"{filename} (p. {page_num + 1})",
                            "text": chunk_text
                        })
                    start += (chunk_size - overlap)
        except Exception as e:
            st.sidebar.error(f"Error reading {filename}: {e}")

    if not chunks:
        return None, []

    # Generate embeddings matrix
    texts_to_embed = [f"Source: {c['source']}\n{c['text']}" for c in chunks]
    embeddings = embedder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)

    return embeddings, chunks

embeddings_matrix, chunks_db = index_pdf_documents()

# Knowledge Base Status in Sidebar
with st.sidebar.expander("📚 Knowledge Base Status", expanded=True):
    if chunks_db:
        sources = sorted(list(set(c["source"].split(" (")[0] for c in chunks_db)))
        st.success(f"✅ Indexed {len(chunks_db)} document chunks.")
        st.markdown("**Active PDFs:**")
        for src in sources:
            st.markdown(f"- `{src}`")
    else:
        st.warning("⚠️ No PDFs found in `ocr_files/`. Place your OCR PDFs there to enable document search.")

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# Search Helper Function
def retrieve_relevant_context(query, top_k=3):
    if embeddings_matrix is None or not chunks_db:
        return ""
    
    # Query embedding
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    
    # Cosine similarity via dot product
    scores = np.dot(embeddings_matrix, query_emb.T).squeeze()
    
    # Top-K indices
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    retrieved_texts = []
    for idx in top_indices:
        chunk = chunks_db[idx]
        retrieved_texts.append(f"--- [From {chunk['source']}] ---\n{chunk['text']}")
        
    return "\n\n".join(retrieved_texts)

# 5. Main Chat Interface
st.title("🧪 OCR A Level Chemistry AI Assistant")
st.caption("Grounded on official OCR A specifications, data sheets, and mark scheme context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about OCR Chemistry..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Search PDFs for relevant context
    context = retrieve_relevant_context(user_input, top_k=3)

    with st.chat_message("assistant"):
        # Display retrieved context expander for transparency
        if context:
            with st.expander("🔍 Retreived Specification & Mark Scheme Context"):
                st.markdown(context)

        # Construct system message with context
        augmented_system_prompt = SYSTEM_PROMPT
        if context:
            augmented_system_prompt += f"\n\nRELEVANT OCR SPECIFICATION & MARK SCHEME CONTEXT:\n{context}"

        api_messages = [{"role": "system", "content": augmented_system_prompt}] + [
            {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
        ]

        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=selected_model,
                messages=api_messages,
                temperature=0.1,
                stream=True
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as err:
            st.error(f"API Error encountered: {str(err)}")
