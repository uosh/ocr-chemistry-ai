import glob
import os
import re
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st
from groq import Groq

# ------------------------------------------------------------------------------
# 1. Page Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="OCR Chemistry AI (RAG)",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------------
# 2. Text Cleaning Helpers
# ------------------------------------------------------------------------------
def clean_pdf_text(text: str) -> str:
    """Removes line-break hyphens and normalizes whitespace in raw PDF text."""
    if not text:
        return ""
    # Fix hyphenated words broken across lines (e.g. "enthal-\npy" -> "enthalpy")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Replace single line breaks with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Normalize consecutive spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_display_text(text: str) -> str:
    """Converts HTML break tags to newlines and fixes raw LaTeX delimiters for Streamlit rendering."""
    if not text:
        return ""
    # Convert HTML break tags to Markdown newlines
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

    # Convert LaTeX wrapped in parentheses like ( \text{NO}_2^{+} ) to $\text{NO}_2^{+}$
    text = re.sub(r"\(\s*(\\text\{.*?\}.*?)\s*\)", r"$\1$", text)

    # Convert standard LaTeX delimiters \(...\) or \[...\] to $...$ and $$...$$
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)
    text = re.sub(r"\\\[(.*?)\\]", r"$$\1$$", text)

    return text.strip()


# ------------------------------------------------------------------------------
# 3. System Instruction
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert OCR A Level Chemistry AI Assistant, specialized in helping students master the OCR Chemistry specifications.

STRICT GROUNDING & EXAM RULES:
1. Base your answers primarily on the official OCR specification and mark scheme context provided below.
2. Align all definitions, key terms, and reaction mechanisms directly with official OCR guidelines.
3. Highlight required exam keywords in **bold** (e.g., **heterolytic fission**, **lone pair on nitrogen**).
4. Use LaTeX ($...$ for inline, $$...$$ for block equations) for calculations and chemical formulas.
"""

# ------------------------------------------------------------------------------
# 4. Sidebar Setup & Credentials
# ------------------------------------------------------------------------------
st.sidebar.title("🧪 OCR Chemistry AI")

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.sidebar.error(
        "⚠️ `GROQ_API_KEY` missing in Streamlit Secrets or Environment."
    )
    st.stop()

client = Groq(api_key=groq_api_key)

selected_model = st.sidebar.selectbox(
    "Active Groq Model",
    options=[
        "openai/gpt-oss-120b",
        "qwen/qwen3.6-27b",
        "llama-3.1-8b-instant",
    ],
    index=0,
)


# ------------------------------------------------------------------------------
# 5. RAG Engine: PDF Processing & Embeddings
# ------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()


@st.cache_data(show_spinner="Indexing OCR Knowledge Base...")
def index_pdf_documents(folder_path="ocr_files", chunk_size=600, overlap=100):
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

    # Fallback to root directory if ocr_files is empty or missing
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
                raw_text = page.extract_text() or ""
                text = clean_pdf_text(raw_text)
                if not text:
                    continue

                # Sliding window chunking
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk_text = text[start:end].strip()
                    if chunk_text:
                        chunks.append({
                            "source": f"{filename} (p. {page_num + 1})",
                            "text": chunk_text,
                        })
                    start += chunk_size - overlap
        except Exception as e:
            st.sidebar.error(f"Error reading {filename}: {e}")

    if not chunks:
        return None, []

    texts_to_embed = [f"Source: {c['source']}\n{c['text']}" for c in chunks]
    embeddings = embedder.encode(
        texts_to_embed, convert_to_numpy=True, normalize_embeddings=True
    )

    return embeddings, chunks


embeddings_matrix, chunks_db = index_pdf_documents()

# Knowledge Base Status Display
with st.sidebar.expander("📚 Knowledge Base Status", expanded=True):
    if chunks_db:
        sources = sorted(
            list(set(c["source"].split(" (")[0] for c in chunks_db))
        )
        st.success(f"✅ Indexed {len(chunks_db)} document chunks.")
        st.markdown("**Active Files:**")
        for src in sources:
            st.markdown(f"- `{src}`")
    else:
        st.warning(
            "⚠️ No PDFs found. Create an `ocr_files/` folder in your repo and upload your PDFs."
        )

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


def retrieve_relevant_context(query, top_k=3):
    if embeddings_matrix is None or not chunks_db:
        return []

    query_emb = embedder.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True
    )
    scores = np.dot(embeddings_matrix, query_emb.T).squeeze()

    if np.ndim(scores) == 0:
        top_indices = [0]
    else:
        top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx]) if np.ndim(scores) > 0 else float(scores)
        chunk = chunks_db[idx]
        results.append({
            "source": chunk["source"],
            "text": chunk["text"],
            "score": round(score * 100, 1),
        })

    return results


# ------------------------------------------------------------------------------
# 6. Main Chat Interface
# ------------------------------------------------------------------------------
st.title("🧪 OCR A Level Chemistry AI Assistant")
st.caption(
    "Grounded on official OCR A specifications, data sheets, and mark schemes."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about OCR Chemistry..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    retrieved_chunks = retrieve_relevant_context(user_input, top_k=3)

    with st.chat_message("assistant"):
        # Format and display retrieved context cards
        if retrieved_chunks:
            with st.expander(
                f"🔍 Retrieved Specification Context ({len(retrieved_chunks)} matches)"
            ):
                for idx, chunk in enumerate(retrieved_chunks, 1):
                    st.markdown(
                        f"**Match #{idx}** · `{chunk['source']}` &nbsp;|&nbsp; **Relevance:** `{chunk['score']}%`"
                    )
                    formatted_text = clean_display_text(chunk["text"])
                    st.info(formatted_text)
                    if idx < len(retrieved_chunks):
                        st.divider()

        context_str = "\n\n".join([
            f"--- Source: {c['source']} ---\n{c['text']}"
            for c in retrieved_chunks
        ])

        augmented_system_prompt = SYSTEM_PROMPT
        if context_str:
            augmented_system_prompt += f"\n\nRELEVANT OCR SPECIFICATION & MARK SCHEME CONTEXT:\n{context_str}"

        api_messages = [{"role": "system", "content": augmented_system_prompt}] + [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]

        response_placeholder = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=selected_model,
                messages=api_messages,
                temperature=0.1,
                stream=True,
            )

            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as err:
            st.error(f"API Error encountered: {str(err)}")
