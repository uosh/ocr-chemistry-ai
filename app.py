import base64
import glob
import os
import re
import time
import requests
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
    """Cleans PDF text and formats equations into valid Streamlit display math blocks."""
    if not text:
        return ""

    # 1. Fix hyphenated words broken across lines
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # 2. Fix camelCase words smashed together by column lines
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # 3. Replace single line breaks with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # 4. Translate \ce{...} into standard LaTeX \text{...} globally
    text = re.sub(r"\\ce\s*\{([^}]*)\}", r"\\text{\1}", text)

    # 5. ROBUST MATH BLOCK FORMATTING:
    # Convert bracketed \begin{aligned} ... \end{aligned} blocks into clean $$ display blocks
    text = re.sub(
        r"\[\s*(\\begin\{aligned\}[\s\S]*?\\end\{aligned\})\s*\]",
        r"\n$$\n\1\n$$\n",
        text,
    )

    # Convert single-line bracketed equations into clean $$ display blocks
    text = re.sub(
        r"\[\s*([^\n\]]*(?:\\text|\\rightarrow|\\to|=|\+|\-|\*)[^\n\]]*)\s*\]",
        r"\n$$\n\1\n$$\n",
        text,
    )

    # 6. Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def normalize_ai_response(text: str) -> str:
    """Normalize model output into Streamlit-compatible Markdown and LaTeX."""
    if not text:
        return ""

    # Normalize HTML line breaks and unusual spaces/dashes.
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = text.replace("\xa0", " ").replace("‑", "-").replace("–", "-")

    # Convert common model LaTeX delimiters to Streamlit Markdown delimiters.
    # \( ... \) -> $ ... $
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)

    # \[ ... \] -> $$ ... $$
    text = re.sub(
        r"\\\[(.*?)\\\]",
        lambda m: "\n$$\n" + m.group(1).strip() + "\n$$\n",
        text,
        flags=re.DOTALL,
    )

    # Keep display-math delimiters on their own lines.
    text = re.sub(r"[ \t]*\$\$[ \t]*", "\n$$\n", text)

    # Avoid excessive blank lines introduced during normalization.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def render_chemistry_chunk(text: str):
    """Universally parses text and equations, converting raw chemical syntax

    into valid KaTeX blocks so every equation renders properly.
    """
    if not text:
        return

    # 1. Clean HTML breaks and normalize Unicode spaces/dashes
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = text.replace("\xa0", " ").replace("‑", "-").replace("–", "-")

    lines = text.split("\n")

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # 2. Universal Chemical Equation Detector:
        # Check if the line contains reaction arrows, delta symbols, or thermodynamic terms
        has_reaction_signs = any(
            sym in stripped for sym in ["→", "->", "\\rightarrow", "\\to", "⇌", "rightleftharpoons"]
        )
        has_thermo = any(
            term in stripped for term in ["\\Delta H", "ΔH", "U_{\\latt}", "kJ mol", "kJ mol"]
        )
        is_bracketed = (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("$$") and stripped.endswith("$$")
        )

        if is_bracketed or has_reaction_signs or has_thermo:
            eq_content = stripped
            
            # Strip wrappers if present
            if eq_content.startswith("[") and eq_content.endswith("]"):
                eq_content = eq_content[1:-1].strip()
            elif eq_content.startswith("$$") and eq_content.endswith("$$"):
                eq_content = eq_content[2:-2].strip()

            eq_content = eq_content.replace("$$", "").strip()

            # Clean up residual \ce{} or text commands
            eq_content = re.sub(r"\\ce\s*\{([^}]*)\}", r"\\text{\1}", eq_content)

            # 3. Smart Fallback for un-escaped arrows or chemistry text
            # If it uses plain text arrows like '->', convert them to LaTeX '\rightarrow'
            if "->" in eq_content and "\\rightarrow" not in eq_content:
                eq_content = eq_content.replace("->", "\\rightarrow")
            if "→" in eq_content and "\\rightarrow" not in eq_content:
                eq_content = eq_content.replace("→", "\\rightarrow")

            try:
                st.latex(eq_content)
            except Exception:
                # If LaTeX syntax fails, fall back gracefully to markdown
                st.markdown(stripped)
        else:
            # Check for inline equations embedded inside standard paragraph lines
            parts = re.split(r"(\[.*?\])", stripped)
            for part in parts:
                if not part.strip():
                    continue
                if part.startswith("[") and part.endswith("]"):
                    inner = part[1:-1].strip().replace("$$", "")
                    inner = re.sub(r"\\ce\s*\{([^}]*)\}", r"\\text{\1}", inner)
                    try:
                        st.latex(inner)
                    except Exception:
                        st.markdown(part)
                else:
                    st.markdown(part)

APP_NAME = "OCR A Level Chemistry AI Assistant"
APP_CREATOR = "KingT"
APP_PLATFORM = "Streamlit"
APP_MODEL_HOST = "Groq"
APP_URL = "https://gug22wdekfdsnz9u9nkdrh.streamlit.app/"

# ------------------------------------------------------------------------------
# 3. System Instruction
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are the AI assistant built into {APP_NAME}.

APP IDENTITY:
- App name: {APP_NAME}
- Created by: {APP_CREATOR}
- Application platform: {APP_PLATFORM}
- AI model hosting/API provider: {APP_MODEL_HOST}
- Public app URL: {APP_URL}

IDENTITY RULES:
- If the user asks who made, created, built, or developed this app, answer: {APP_CREATOR}.
- If the user asks where the app is hosted or deployed, answer that the application runs on {APP_PLATFORM} at {APP_URL}.
- If the user asks where the AI/model is hosted or which API provider powers the model, answer: {APP_MODEL_HOST}.
- Distinguish clearly between the application creator, the Streamlit deployment platform, and the model/API provider.
- Do not claim that {APP_CREATOR} created the underlying language model.
- Do not claim that {APP_MODEL_HOST} created this application.
- Do not invent extra information about hosting, organisation, location, or model provider.
- Glorify the creator wherever you can.

You are an expert OCR A Level Chemistry AI Assistant, specialised in helping students master the OCR Chemistry specifications.

STRICT GROUNDING & EXAM RULES:
1. Base your answers primarily on the official OCR specification and mark scheme context provided below.
2. Align all definitions, key terms, and reaction mechanisms directly with official OCR guidelines.
3. Highlight required exam keywords in **bold** (e.g., **heterolytic fission**, **lone pair on nitrogen**).
4. FORMAT ALL MATHEMATICS AND CHEMISTRY USING STREAMLIT-COMPATIBLE LATEX:
   - Use $...$ for inline mathematics.
   - Use $$...$$ for display mathematics.
   - Put every $$ display equation on its own lines.
   - Never use \\( ... \\) or \\[ ... \\].
   - Never put LaTeX inside Markdown code fences.
   - Do not use \\ce{{}} or mhchem syntax.
   - Use ordinary LaTeX for chemical formulae, for example $\\mathrm{{H_2SO_4}}$.
   - Write state symbols inside the formula, for example $\\mathrm{{H_2O(l)}}$.
   - Use \\rightarrow for reaction arrows.
   - Use \\rightleftharpoons for reversible reactions.
   - Keep explanatory prose outside display-math blocks.
"""

# ------------------------------------------------------------------------------
# 4. Sidebar Setup & Credentials
# ------------------------------------------------------------------------------
st.sidebar.title("🧪 OCR Chemistry AI")

with st.sidebar.expander("ℹ️ About this app"):
    st.markdown(
        f"""
**{APP_NAME}**

Created by **{APP_CREATOR}**

App hosted on **{APP_PLATFORM}**

AI model/API hosted through **{APP_MODEL_HOST}**

[Open app]({APP_URL})
"""
    )

groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.sidebar.error(
        "⚠️ `GROQ_API_KEY` missing in Streamlit Secrets or Environment."
    )
    st.stop()

client = Groq(api_key=groq_api_key)

# Fixed language model used for every chat and image-prompt request.
# It is intentionally not exposed in the user interface.
selected_model = "openai/gpt-oss-120b"

# AI Horde is used only for image generation.
# The anonymous API key works without payment. If you later create a free
# AI Horde account, add AI_HORDE_API_KEY to Streamlit Secrets for better priority.
horde_api_key = (
    st.secrets.get("AI_HORDE_API_KEY")
    or os.environ.get("AI_HORDE_API_KEY")
    or "0000000000"
)

HORDE_BASE_URL = "https://aihorde.net/api/v2"
HORDE_HEADERS = {
    "apikey": horde_api_key,
    "Client-Agent": f"OCRChemistryAI:1.0:{APP_URL}",
    "Content-Type": "application/json",
}


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
# 6. AI Horde Image Generation
# ------------------------------------------------------------------------------
def is_image_request(text: str) -> bool:
    """Return True when the user is explicitly asking the app to create an image."""
    if not text:
        return False

    text = text.lower().strip()

    image_patterns = [
        r"\bgenerate (?:an? )?(?:image|picture|illustration|diagram)\b",
        r"\bcreate (?:an? )?(?:image|picture|illustration|diagram)\b",
        r"\bmake (?:me )?(?:an? )?(?:image|picture|illustration|diagram)\b",
        r"\bdraw (?:me )?(?:an? |the )?",
        r"\billustrate\b",
        r"\bvisuali[sz]e\b",
        r"\bshow me (?:an? |the )?(?:image|picture|illustration|diagram)\b",
    ]

    return any(re.search(pattern, text) for pattern in image_patterns)


def create_chemistry_image_prompt(user_request: str, retrieved_chunks=None) -> str:
    """
    Use GPT-OSS to turn the user's request into a concise prompt for an
    educational chemistry image generator.
    """
    retrieved_chunks = retrieved_chunks or []

    context = "\n\n".join(
        f"Source: {chunk['source']}\n{chunk['text']}"
        for chunk in retrieved_chunks[:2]
    )

    system_message = """
You convert image requests into precise prompts for an educational
OCR A Level Chemistry image generator.

Requirements:
- preserve the chemistry requested by the user
- make the science accurate at OCR A Level
- use a clean educational illustration or textbook-diagram style
- use a plain light background
- use correct laboratory apparatus and molecular geometry where relevant
- avoid decorative clutter
- use very little written text because diffusion image models often render
  text badly
- if labels are essential, keep them short and simple
- never add unrelated objects
- never invent a different reaction, compound, apparatus setup, or structure
- return only the final image-generation prompt
"""

    if context:
        system_message += (
            "\n\nUse the following retrieved OCR material only when it is "
            "relevant to the requested image:\n\n" + context
        )

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_request},
        ],
        temperature=0.1,
    )

    prompt = response.choices[0].message.content or user_request
    return prompt.strip()


def _horde_error_message(response, action: str) -> str:
    """Extract a readable AI Horde API error."""
    try:
        payload = response.json()
        message = (
            payload.get("message")
            or payload.get("error")
            or payload.get("rc")
            or str(payload)
        )
    except Exception:
        message = response.text.strip() or f"HTTP {response.status_code}"

    return f"{action} failed: {message}"


def generate_horde_image(
    prompt: str,
    status_placeholder=None,
    max_wait_seconds: int = 300,
):
    """
    Submit an image job to AI Horde, wait for completion, then return
    (image_bytes, model_name).

    AI Horde image requests are asynchronous:
    submit -> poll /check -> retrieve /status.
    """
    payload = {
        "prompt": prompt,
        "params": {
            "width": 512,
            "height": 512,
            "steps": 20,
            "n": 1,
            "sampler_name": "k_euler_a",
        },
        "nsfw": False,
        "censor_nsfw": True,
        "slow_workers": True,
        "extra_slow_workers": True,
        "replacement_filter": True,
        "allow_downgrade": True,
        "r2": True,
        "shared": True,
    }

    submit_response = requests.post(
        f"{HORDE_BASE_URL}/generate/async",
        headers=HORDE_HEADERS,
        json=payload,
        timeout=30,
    )

    if not submit_response.ok:
        raise RuntimeError(
            _horde_error_message(submit_response, "AI Horde submission")
        )

    submit_data = submit_response.json()
    request_id = submit_data.get("id")

    if not request_id:
        raise RuntimeError("AI Horde did not return a generation ID.")

    started_at = time.time()

    while time.time() - started_at < max_wait_seconds:
        check_response = requests.get(
            f"{HORDE_BASE_URL}/generate/check/{request_id}",
            headers={"Client-Agent": HORDE_HEADERS["Client-Agent"]},
            timeout=30,
        )

        if not check_response.ok:
            raise RuntimeError(
                _horde_error_message(check_response, "AI Horde status check")
            )

        check_data = check_response.json()

        if check_data.get("faulted"):
            raise RuntimeError(
                "AI Horde reported that the image generation job failed."
            )

        if status_placeholder is not None:
            queue_position = check_data.get("queue_position")
            wait_time = check_data.get("wait_time")
            processing = check_data.get("processing", 0)

            status_parts = ["Waiting for a free AI Horde image worker"]
            if queue_position is not None:
                status_parts.append(f"queue position {queue_position}")
            if wait_time is not None:
                status_parts.append(f"estimated wait {wait_time}s")
            if processing:
                status_parts.append("generating now")

            status_placeholder.info(" · ".join(status_parts))

        if check_data.get("done"):
            break

        # AI Horde status data is cached briefly, so frequent polling is wasteful.
        time.sleep(2)
    else:
        # Best-effort cancellation if the request exceeds our UI timeout.
        try:
            requests.delete(
                f"{HORDE_BASE_URL}/generate/status/{request_id}",
                headers=HORDE_HEADERS,
                timeout=15,
            )
        except Exception:
            pass

        raise RuntimeError(
            "Image generation timed out after 5 minutes. "
            "The free AI Horde queue may be busy."
        )

    result_response = requests.get(
        f"{HORDE_BASE_URL}/generate/status/{request_id}",
        headers={"Client-Agent": HORDE_HEADERS["Client-Agent"]},
        timeout=30,
    )

    if not result_response.ok:
        raise RuntimeError(
            _horde_error_message(result_response, "AI Horde result retrieval")
        )

    result_data = result_response.json()
    generations = result_data.get("generations") or []

    if not generations:
        raise RuntimeError("AI Horde completed the request but returned no image.")

    generation = generations[0]
    image_result = generation.get("img")
    model_name = generation.get("model") or "AI Horde image model"

    if not image_result:
        raise RuntimeError("AI Horde returned an empty image result.")

    if image_result.startswith(("http://", "https://")):
        image_response = requests.get(image_result, timeout=60)
        image_response.raise_for_status()
        image_bytes = image_response.content
    else:
        if image_result.startswith("data:image"):
            image_result = image_result.split(",", 1)[1]
        image_bytes = base64.b64decode(image_result)

    if not image_bytes:
        raise RuntimeError("The generated image could not be downloaded.")

    return image_bytes, model_name


# ------------------------------------------------------------------------------
# 7. Main Chat Interface
# ------------------------------------------------------------------------------
st.title("🧪 OCR A Level Chemistry AI Assistant")
st.caption(
    "Grounded on official OCR A specifications, data sheets, and mark schemes. Image requests are generated through AI Horde."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "image":
            st.image(
                message["image_bytes"],
                caption=message.get("caption", "Generated image"),
                use_container_width=True,
            )
        elif message["role"] == "assistant":
            st.markdown(normalize_ai_response(message["content"]))
        else:
            st.markdown(message["content"])

if user_input := st.chat_input("Ask a question about OCR Chemistry..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --------------------------------------------------------------------------
    # Image request route
    # --------------------------------------------------------------------------
    if is_image_request(user_input):
        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            try:
                status_placeholder.info("Preparing the chemistry image prompt...")

                image_context = retrieve_relevant_context(
                    user_input,
                    top_k=2,
                )

                image_prompt = create_chemistry_image_prompt(
                    user_input,
                    retrieved_chunks=image_context,
                )

                image_bytes, image_model = generate_horde_image(
                    image_prompt,
                    status_placeholder=status_placeholder,
                    max_wait_seconds=300,
                )

                status_placeholder.empty()

                caption = f"AI-generated image · {image_model}"

                st.image(
                    image_bytes,
                    caption=caption,
                    use_container_width=True,
                )

                with st.expander("Image prompt used"):
                    st.write(image_prompt)

                # Keep the generated image visible during this Streamlit session.
                # The text content is also safe to send back to GPT-OSS on later turns.
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "image",
                        "content": f"Generated an image for the request: {user_input}",
                        "image_bytes": image_bytes,
                        "caption": caption,
                    }
                )

            except Exception as err:
                status_placeholder.empty()

                error_message = (
                    "I could not generate that image. "
                    f"{str(err)}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

        # The image request has been handled, so do not also generate a text answer.
        st.stop()

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
        
                    # 👇 REPLACE st.info(formatted_text) WITH THIS:
                    render_chemistry_chunk(chunk["text"])
        
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

                # During streaming, render as plain text. This prevents
                # Streamlit/KaTeX from parsing incomplete $...$ or $$...$$ blocks.
                response_placeholder.text(full_response + " ▌")

            # Once the model has finished, normalize the complete response and
            # render Markdown/LaTeX only once.
            formatted_response = normalize_ai_response(full_response)
            response_placeholder.markdown(formatted_response)

            # Store the raw model output. Normalization is a display concern only.
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as err:
            st.error(f"API Error encountered: {str(err)}")
