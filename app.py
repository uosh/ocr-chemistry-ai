import base64
import glob
import html
import json
import os
import re
import time
import requests
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import streamlit as st
import streamlit.components.v1 as components
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
- Do not invent extra information about the hosting organisation, location, or model provider.
- If the user says "ADMIN" then you should ask for password which is "Tazveed", not case-sensitive, to which if entered correctly lets you know that the user is the creator AKA Kingt.

You are an expert OCR A Level Chemistry tutor.

OCR SOURCE RULES:
You may receive retrieved material from OCR specifications, OCR mark schemes,
OCR question papers, data sheets, and other reference files.

Use each source according to its purpose.

SPECIFICATION MATERIAL:
- Use specifications to determine what students are expected to know.
- Use them to control the correct depth and scope of explanations.
- Prefer specification terminology when stating required knowledge.

MARK SCHEME MATERIAL:
Treat OCR mark schemes as especially useful evidence for:
- accepted definitions
- marking points
- exam keywords
- required distinctions
- acceptable alternative wording
- common wording that earns marks
- wording that is too vague or incomplete
- the level of precision OCR expects

When mark-scheme material is relevant:
- explain the chemistry first so the student understands it
- then explain how to phrase the answer safely for an OCR exam
- identify important exam keywords in **bold**
- state what idea actually earns the mark when the evidence supports it
- point out vague wording that could lose a mark
- suggest a stronger exam-style answer where useful

You may use wording such as:
- "For OCR, the key marking point is..."
- "A safer exam answer is..."
- "This is chemically reasonable, but the mark scheme expects..."
- "Include **...** because that is the marking point."

IMPORTANT MARK-SCHEME LIMITS:
- Do not blindly copy mark schemes.
- Do not claim that wording from one question is mandatory for every question.
- Do not invent OCR marking rules.
- Do not claim an exact phrase is required unless the retrieved OCR material supports that.
- Treat ALLOW, ACCEPT, IGNORE, NOT, DO NOT ALLOW, and equivalent mark-scheme
  instructions as question-specific unless repeated evidence supports a wider rule.
- If retrieved sources disagree or are insufficient, say so rather than guessing.

QUESTION PAPER MATERIAL:
- Use question papers to understand OCR command words, question style, expected
  depth, and typical ways concepts are assessed.
- Do not treat a question paper itself as evidence that a particular answer earns a mark.

ANSWERING STUDENTS:
- Answer the student's actual question first.
- Teach the underlying chemistry clearly.
- Add exam advice only when it is relevant and useful.
- If a student gives an answer, explain what is correct, what is vague or missing,
  and how to improve it using the retrieved OCR evidence.
- Keep the distinction clear between general chemistry knowledge and specific
  OCR mark-scheme expectations.

GROUNDING:
1. Base OCR-specific claims primarily on the retrieved official OCR context.
2. Never invent a mark-scheme requirement.
3. When retrieved mark-scheme evidence is present, use it actively rather than ignoring it.
4. If no relevant mark-scheme evidence was retrieved, do not pretend that a particular
   wording is definitely required by OCR.
5. Highlight important exam terminology in **bold**.

LATEX AND CHEMISTRY FORMAT:
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
def detect_document_type(filename: str) -> str:
    """
    Classify OCR PDFs from their filenames.

    Recommended filenames include obvious labels such as:
      2024_H432_01_MS.pdf
      2024_H432_01_QP.pdf
      OCR_Chemistry_A_Specification.pdf
    """
    name = filename.lower()
    stem = os.path.splitext(name)[0]

    mark_scheme_patterns = [
        r"(^|[_\-\s])ms($|[_\-\s])",
        r"mark[_\-\s]*scheme",
        r"markscheme",
    ]

    question_paper_patterns = [
        r"(^|[_\-\s])qp($|[_\-\s])",
        r"question[_\-\s]*paper",
    ]

    specification_patterns = [
        r"specification",
        r"(^|[_\-\s])spec($|[_\-\s])",
    ]

    data_sheet_patterns = [
        r"data[_\-\s]*sheet",
        r"datasheet",
    ]

    if any(re.search(pattern, stem) for pattern in mark_scheme_patterns):
        return "mark_scheme"

    if any(re.search(pattern, stem) for pattern in question_paper_patterns):
        return "question_paper"

    if any(re.search(pattern, stem) for pattern in specification_patterns):
        return "specification"

    if any(re.search(pattern, stem) for pattern in data_sheet_patterns):
        return "data_sheet"

    return "reference"


def document_type_label(document_type: str) -> str:
    return {
        "mark_scheme": "Mark Scheme",
        "question_paper": "Question Paper",
        "specification": "Specification",
        "data_sheet": "Data Sheet",
        "reference": "Reference",
    }.get(document_type, "Reference")


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
        document_type = detect_document_type(filename)

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
                            "filename": filename,
                            "page": page_num + 1,
                            "document_type": document_type,
                            "text": chunk_text,
                        })
                    start += chunk_size - overlap
        except Exception as e:
            st.sidebar.error(f"Error reading {filename}: {e}")

    if not chunks:
        return None, []

    texts_to_embed = [
        (
            f"Document type: {document_type_label(c['document_type'])}\n"
            f"Source: {c['source']}\n"
            f"{c['text']}"
        )
        for c in chunks
    ]
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

        type_files = {}
        for chunk in chunks_db:
            doc_type = chunk.get("document_type", "reference")
            type_files.setdefault(doc_type, set()).add(chunk.get("filename", "Unknown"))

        st.markdown("**Knowledge base:**")
        for doc_type in [
            "specification",
            "mark_scheme",
            "question_paper",
            "data_sheet",
            "reference",
        ]:
            files_for_type = type_files.get(doc_type, set())
            if files_for_type:
                st.markdown(
                    f"- **{document_type_label(doc_type)}:** "
                    f"{len(files_for_type)} file(s)"
                )

        with st.expander("Active files"):
            for src in sources:
                detected_type = detect_document_type(src)
                st.markdown(
                    f"- `{src}` · {document_type_label(detected_type)}"
                )
    else:
        st.warning(
            "⚠️ No PDFs found. Create an `ocr_files/` folder in your repo and upload your PDFs."
        )

if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()


def retrieve_relevant_context(query, top_k=6):
    """
    Retrieve semantically relevant OCR chunks, with a small preference for
    official specification and mark-scheme evidence.

    The preference is deliberately small: relevance still dominates.
    """
    if embeddings_matrix is None or not chunks_db:
        return []

    query_emb = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scores = np.dot(embeddings_matrix, query_emb.T).squeeze()

    if np.ndim(scores) == 0:
        scores = np.array([float(scores)])

    query_lower = query.lower()

    exam_intent_terms = [
        "define",
        "definition",
        "mark",
        "marks",
        "mark scheme",
        "markscheme",
        "exam",
        "ocr",
        "keyword",
        "wording",
        "state",
        "explain",
        "answer",
        "would this get",
        "how many marks",
        "improve my answer",
    ]

    has_exam_intent = any(term in query_lower for term in exam_intent_terms)

    reranked = []

    for idx, base_score in enumerate(scores):
        chunk = chunks_db[idx]
        doc_type = chunk.get("document_type", "reference")

        adjusted_score = float(base_score)

        # Small source-type preferences. They should never overpower a
        # substantially more relevant chunk.
        if doc_type == "specification":
            adjusted_score += 0.018

        if doc_type == "mark_scheme":
            adjusted_score += 0.025
            if has_exam_intent:
                adjusted_score += 0.025

        if doc_type == "question_paper" and not has_exam_intent:
            adjusted_score -= 0.008

        reranked.append((adjusted_score, float(base_score), idx))

    reranked.sort(key=lambda item: item[0], reverse=True)

    # Look at more candidates than we finally return so we can include useful
    # mark-scheme evidence without filling the context with near-duplicate chunks.
    candidate_count = min(len(reranked), max(top_k * 4, 16))
    candidates = reranked[:candidate_count]

    selected = []
    selected_indices = set()

    def add_candidate(candidate):
        adjusted_score, base_score, idx = candidate
        if idx in selected_indices:
            return
        selected.append((adjusted_score, base_score, idx))
        selected_indices.add(idx)

    # For exam/definition questions, try to include relevant mark-scheme evidence
    # if it exists among the strong candidates.
    if has_exam_intent:
        mark_scheme_candidates = [
            candidate
            for candidate in candidates
            if chunks_db[candidate[2]].get("document_type") == "mark_scheme"
        ]

        for candidate in mark_scheme_candidates[:2]:
            add_candidate(candidate)

    # Include a strong specification chunk where available.
    specification_candidates = [
        candidate
        for candidate in candidates
        if chunks_db[candidate[2]].get("document_type") == "specification"
    ]

    if specification_candidates:
        add_candidate(specification_candidates[0])

    # Fill remaining slots by reranked relevance.
    for candidate in candidates:
        if len(selected) >= top_k:
            break
        add_candidate(candidate)

    # Keep final output in adjusted relevance order.
    selected = sorted(selected, key=lambda item: item[0], reverse=True)[:top_k]

    results = []

    for adjusted_score, base_score, idx in selected:
        chunk = chunks_db[idx]

        results.append({
            "source": chunk["source"],
            "filename": chunk.get("filename", ""),
            "page": chunk.get("page"),
            "document_type": chunk.get("document_type", "reference"),
            "document_type_label": document_type_label(
                chunk.get("document_type", "reference")
            ),
            "text": chunk["text"],
            "score": round(base_score * 100, 1),
        })

    return results




# ------------------------------------------------------------------------------
# 6. Deterministic Chemistry Diagram Engine
# ------------------------------------------------------------------------------
def is_diagram_request(text: str) -> bool:
    """Detect requests that are better rendered as clean vector diagrams."""
    if not text:
        return False

    text = text.lower().strip()

    strong_terms = [
        "diagram",
        "labelled diagram",
        "labeled diagram",
        "apparatus",
        "experimental setup",
        "experimental set-up",
        "reaction setup",
        "reaction set-up",
        "electrolysis cell",
        "electrochemical cell",
        "galvanic cell",
        "voltaic cell",
        "titration setup",
        "titration apparatus",
        "distillation apparatus",
        "fractional distillation",
        "simple distillation",
        "calorimetry setup",
        "calorimeter",
    ]

    if any(term in text for term in strong_terms):
        return True

    # In this chemistry app, "draw" normally means a teaching diagram.
    draw_phrases = [
        "draw ",
        "sketch ",
        "show the setup",
        "show the set-up",
        "show the apparatus",
    ]

    return any(phrase in text for phrase in draw_phrases)


def _extract_json_object(text: str):
    """Extract one JSON object even if the model accidentally adds prose."""
    if not text:
        raise ValueError("The model returned an empty diagram specification.")

    text = text.strip()

    # Remove common Markdown fences.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("The model did not return valid JSON for the diagram.")

    return json.loads(text[start:end + 1])


def create_chemistry_diagram_spec(user_request: str, retrieved_chunks=None):
    """
    Ask GPT-OSS for structured drawing instructions rather than an image.
    The model is never allowed to emit raw SVG or HTML.
    """
    retrieved_chunks = retrieved_chunks or []

    context = "\n\n".join(
        (
            f"Document type: {chunk.get('document_type_label', 'Reference')}\n"
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )
        for chunk in retrieved_chunks[:2]
    )

    schema = r"""
Return ONE JSON object only.

Coordinate system:
- canvas width: 1000
- canvas height: 680
- x increases left to right
- y increases top to bottom

Schema:
{
  "title": "short diagram title",
  "caption": "one short explanatory sentence",
  "elements": [
    {
      "type": "text",
      "x": 100,
      "y": 100,
      "text": "label",
      "size": 24,
      "anchor": "start"
    },
    {
      "type": "line",
      "x1": 100,
      "y1": 100,
      "x2": 300,
      "y2": 100,
      "width": 4,
      "dashed": false
    },
    {
      "type": "arrow",
      "x1": 100,
      "y1": 100,
      "x2": 300,
      "y2": 100,
      "width": 4,
      "label": ""
    },
    {
      "type": "rect",
      "x": 100,
      "y": 100,
      "w": 200,
      "h": 100,
      "label": ""
    },
    {
      "type": "circle",
      "cx": 200,
      "cy": 200,
      "r": 20,
      "label": ""
    },
    {
      "type": "beaker",
      "x": 100,
      "y": 250,
      "w": 300,
      "h": 250,
      "label": "electrolyte"
    },
    {
      "type": "electrode",
      "x": 180,
      "y": 180,
      "w": 35,
      "h": 220,
      "label": "Anode (+)"
    },
    {
      "type": "battery",
      "x": 400,
      "y": 80,
      "w": 170,
      "h": 70,
      "label": "d.c. supply"
    },
    {
      "type": "burette",
      "x": 420,
      "y": 90,
      "h": 300,
      "label": "burette"
    },
    {
      "type": "flask",
      "x": 350,
      "y": 390,
      "w": 220,
      "h": 190,
      "label": "conical flask"
    },
    {
      "type": "test_tube",
      "x": 200,
      "y": 180,
      "w": 90,
      "h": 280,
      "label": ""
    },
    {
      "type": "thermometer",
      "x": 300,
      "y": 120,
      "h": 250,
      "label": "thermometer"
    },
    {
      "type": "condenser",
      "x1": 400,
      "y1": 220,
      "x2": 760,
      "y2": 340,
      "label": "condenser"
    }
  ]
}

Rules:
- Use only the element types listed above.
- Maximum 45 elements.
- Keep all coordinates inside the canvas.
- Use clean textbook-style layout with generous spacing.
- Use short labels.
- Use Unicode chemistry text instead of LaTeX in labels:
  H₂O, Cl₂, Na⁺, Cl⁻, e⁻, SO₄²⁻.
- Never generate raw SVG, HTML, Markdown, or code fences.
- Never invent decorative objects.
- For electrolysis, clearly distinguish anode and cathode and show ion movement
  with arrows only when chemically relevant.
- For apparatus, use standard A Level laboratory arrangements.
- Put long explanations in the caption, not inside the drawing.
"""

    system_message = (
        "You design precise OCR A Level Chemistry teaching diagrams. "
        "The output will be drawn by a deterministic SVG renderer, so you must "
        "describe the diagram using the JSON schema exactly.\n\n" + schema
    )

    if context:
        system_message += (
            "\n\nRelevant OCR material follows. Use it only where it directly "
            "supports the requested diagram:\n\n" + context
        )

    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_request},
        ],
        temperature=0.0,
    )

    raw = response.choices[0].message.content or ""
    spec = _extract_json_object(raw)

    if not isinstance(spec, dict):
        raise ValueError("Diagram specification must be a JSON object.")

    elements = spec.get("elements", [])
    if not isinstance(elements, list):
        raise ValueError("Diagram elements must be a JSON list.")

    spec["elements"] = elements[:45]
    spec["title"] = str(spec.get("title", "Chemistry diagram"))[:120]
    spec["caption"] = str(spec.get("caption", ""))[:300]

    return spec


def _num(value, default=0.0, minimum=0.0, maximum=1000.0):
    """Safely clamp numeric model output."""
    try:
        value = float(value)
    except Exception:
        value = float(default)
    return max(minimum, min(maximum, value))


def _svg_text(x, y, text_value, size=22, anchor="start", weight="normal"):
    text_value = html.escape(str(text_value))
    anchor = anchor if anchor in {"start", "middle", "end"} else "start"
    size = _num(size, 22, 10, 46)

    return (
        f'<text x="{_num(x)}" y="{_num(y, maximum=680)}" '
        f'font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" fill="currentColor">{text_value}</text>'
    )


def render_chemistry_svg(spec: dict) -> str:
    """
    Convert the safe structured diagram spec into SVG.
    Text is escaped, and the model never controls raw HTML/SVG.
    """
    title = html.escape(str(spec.get("title", "Chemistry diagram")))
    elements = spec.get("elements", [])

    parts = [
        """
<div style="
    width:100%;
    overflow-x:auto;
    border:1px solid rgba(128,128,128,.28);
    border-radius:12px;
    padding:10px;
    background:white;
">
<svg viewBox="0 0 1000 680"
     width="100%"
     xmlns="http://www.w3.org/2000/svg"
     role="img">
<defs>
  <marker id="arrowhead"
          markerWidth="10"
          markerHeight="7"
          refX="9"
          refY="3.5"
          orient="auto">
    <polygon points="0 0, 10 3.5, 0 7" fill="#222"/>
  </marker>
</defs>
<style>
  text { fill:#111; }
  .shape { fill:none; stroke:#222; stroke-width:4; }
  .thin { fill:none; stroke:#444; stroke-width:3; }
  .liquid { fill:#d9eef9; stroke:#222; stroke-width:3; }
</style>
""",
        _svg_text(500, 42, title, size=30, anchor="middle", weight="bold"),
    ]

    allowed = {
        "text", "line", "arrow", "rect", "circle", "beaker", "electrode",
        "battery", "burette", "flask", "test_tube", "thermometer", "condenser"
    }

    for raw_element in elements:
        if not isinstance(raw_element, dict):
            continue

        kind = str(raw_element.get("type", "")).lower().strip()
        if kind not in allowed:
            continue

        if kind == "text":
            parts.append(
                _svg_text(
                    raw_element.get("x", 0),
                    raw_element.get("y", 0),
                    raw_element.get("text", ""),
                    raw_element.get("size", 22),
                    raw_element.get("anchor", "start"),
                )
            )

        elif kind in {"line", "arrow"}:
            x1 = _num(raw_element.get("x1", 0))
            y1 = _num(raw_element.get("y1", 0), maximum=680)
            x2 = _num(raw_element.get("x2", 0))
            y2 = _num(raw_element.get("y2", 0), maximum=680)
            width = _num(raw_element.get("width", 4), 4, 1, 10)
            dashed = bool(raw_element.get("dashed", False))
            dash = ' stroke-dasharray="10 8"' if dashed else ""
            marker = ' marker-end="url(#arrowhead)"' if kind == "arrow" else ""

            parts.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="#222" stroke-width="{width}"{dash}{marker}/>'
            )

            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(
                    _svg_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2 - 10,
                        label,
                        18,
                        "middle",
                    )
                )

        elif kind == "rect":
            x = _num(raw_element.get("x", 0))
            y = _num(raw_element.get("y", 0), maximum=680)
            w = _num(raw_element.get("w", 100), 100, 10, 900)
            h = _num(raw_element.get("h", 80), 80, 10, 600)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                f'rx="10" class="shape"/>'
            )
            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(x + w / 2, y + h / 2 + 7, label, 20, "middle"))

        elif kind == "circle":
            cx = _num(raw_element.get("cx", 0))
            cy = _num(raw_element.get("cy", 0), maximum=680)
            r = _num(raw_element.get("r", 20), 20, 5, 120)
            parts.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" class="shape"/>'
            )
            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(cx, cy + 7, label, 18, "middle"))

        elif kind == "beaker":
            x = _num(raw_element.get("x", 100))
            y = _num(raw_element.get("y", 250), maximum=680)
            w = _num(raw_element.get("w", 300), 300, 100, 650)
            h = _num(raw_element.get("h", 250), 250, 100, 400)

            # Beaker outline with open top.
            path = (
                f"M {x} {y} "
                f"L {x} {y+h-25} "
                f"Q {x} {y+h} {x+25} {y+h} "
                f"L {x+w-25} {y+h} "
                f"Q {x+w} {y+h} {x+w} {y+h-25} "
                f"L {x+w} {y}"
            )
            parts.append(f'<path d="{path}" class="shape"/>')

            liquid_y = y + h * 0.45
            parts.append(
                f'<rect x="{x+5}" y="{liquid_y}" width="{w-10}" '
                f'height="{y+h-liquid_y-5}" class="liquid" opacity="0.75"/>'
            )

            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(x + w / 2, y + h - 45, label, 20, "middle"))

        elif kind == "electrode":
            x = _num(raw_element.get("x", 100))
            y = _num(raw_element.get("y", 100), maximum=680)
            w = _num(raw_element.get("w", 35), 35, 15, 100)
            h = _num(raw_element.get("h", 220), 220, 80, 400)
            parts.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                f'rx="5" fill="#777" stroke="#222" stroke-width="3"/>'
            )
            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(x + w / 2, y - 14, label, 19, "middle", "bold"))

        elif kind == "battery":
            x = _num(raw_element.get("x", 400))
            y = _num(raw_element.get("y", 80), maximum=680)
            w = _num(raw_element.get("w", 170), 170, 100, 300)
            h = _num(raw_element.get("h", 70), 70, 50, 160)

            parts.append(
                f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
                f'fill="#f5f5f5" stroke="#222" stroke-width="4"/>'
            )
            parts.append(_svg_text(x + 30, y + h / 2 + 8, "+", 30, "middle", "bold"))
            parts.append(_svg_text(x + w - 30, y + h / 2 + 7, "−", 30, "middle", "bold"))

            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(x + w / 2, y - 12, label, 18, "middle"))

        elif kind == "burette":
            x = _num(raw_element.get("x", 420))
            y = _num(raw_element.get("y", 90), maximum=680)
            h = _num(raw_element.get("h", 300), 300, 180, 480)

            parts.append(
                f'<rect x="{x-18}" y="{y}" width="36" height="{h}" '
                f'rx="10" fill="#eef8ff" stroke="#222" stroke-width="3"/>'
            )
            parts.append(
                f'<line x1="{x}" y1="{y+h}" x2="{x}" y2="{y+h+65}" '
                f'stroke="#222" stroke-width="4"/>'
            )
            parts.append(
                f'<line x1="{x-35}" y1="{y+h+30}" x2="{x+35}" y2="{y+h+30}" '
                f'stroke="#222" stroke-width="4"/>'
            )

            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(x + 42, y + h / 2, label, 18, "start"))

        elif kind == "flask":
            x = _num(raw_element.get("x", 350))
            y = _num(raw_element.get("y", 390), maximum=680)
            w = _num(raw_element.get("w", 220), 220, 120, 380)
            h = _num(raw_element.get("h", 190), 190, 120, 280)

            neck_w = w * 0.22
            neck_x = x + (w - neck_w) / 2
            neck_h = h * 0.28

            path = (
                f"M {neck_x} {y} "
                f"L {neck_x} {y+neck_h} "
                f"L {x+20} {y+h-25} "
                f"Q {x+10} {y+h} {x+45} {y+h} "
                f"L {x+w-45} {y+h} "
                f"Q {x+w-10} {y+h} {x+w-20} {y+h-25} "
                f"L {neck_x+neck_w} {y+neck_h} "
                f"L {neck_x+neck_w} {y} Z"
            )
            parts.append(f'<path d="{path}" fill="#eef8ff" stroke="#222" stroke-width="4"/>')

            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(x + w / 2, y + h + 30, label, 18, "middle"))

        elif kind == "test_tube":
            x = _num(raw_element.get("x", 200))
            y = _num(raw_element.get("y", 180), maximum=680)
            w = _num(raw_element.get("w", 90), 90, 50, 180)
            h = _num(raw_element.get("h", 280), 280, 120, 420)

            path = (
                f"M {x} {y} "
                f"L {x} {y+h-w/2} "
                f"A {w/2} {w/2} 0 0 0 {x+w} {y+h-w/2} "
                f"L {x+w} {y}"
            )
            parts.append(f'<path d="{path}" fill="#eef8ff" stroke="#222" stroke-width="4"/>')

            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(x + w / 2, y + h + 28, label, 18, "middle"))

        elif kind == "thermometer":
            x = _num(raw_element.get("x", 300))
            y = _num(raw_element.get("y", 120), maximum=680)
            h = _num(raw_element.get("h", 250), 250, 120, 420)

            parts.append(
                f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y+h}" '
                f'stroke="#555" stroke-width="8"/>'
            )
            parts.append(
                f'<line x1="{x}" y1="{y+30}" x2="{x}" y2="{y+h}" '
                f'stroke="#c62828" stroke-width="4"/>'
            )
            parts.append(
                f'<circle cx="{x}" cy="{y+h}" r="14" fill="#c62828" stroke="#555" stroke-width="3"/>'
            )

            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(_svg_text(x + 25, y + h / 2, label, 18, "start"))

        elif kind == "condenser":
            x1 = _num(raw_element.get("x1", 400))
            y1 = _num(raw_element.get("y1", 220), maximum=680)
            x2 = _num(raw_element.get("x2", 760))
            y2 = _num(raw_element.get("y2", 340), maximum=680)

            parts.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="#9fd3ef" stroke-width="34" stroke-linecap="round"/>'
            )
            parts.append(
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                f'stroke="#222" stroke-width="4" stroke-linecap="round"/>'
            )

            label = str(raw_element.get("label", "")).strip()
            if label:
                parts.append(
                    _svg_text((x1+x2)/2, (y1+y2)/2 - 28, label, 18, "middle")
                )

    parts.append("</svg></div>")
    return "".join(parts)


def display_chemistry_diagram(svg: str, height: int = 720):
    """Render the generated SVG safely inside Streamlit."""
    components.html(svg, height=height, scrolling=True)


# ------------------------------------------------------------------------------
# 7. AI Horde Image Generation
# ------------------------------------------------------------------------------
def is_image_request(text: str) -> bool:
    """Detect non-diagram image requests for AI Horde."""
    if not text:
        return False

    # Diagram requests must always use the deterministic SVG renderer.
    if is_diagram_request(text):
        return False

    text = text.lower().strip()

    image_patterns = [
        r"\bgenerate (?:an? )?(?:image|picture|illustration)\b",
        r"\bcreate (?:an? )?(?:image|picture|illustration)\b",
        r"\bmake (?:me )?(?:an? )?(?:image|picture|illustration)\b",
        r"\bshow me (?:an? |the )?(?:image|picture|illustration)\b",
    ]

    return any(re.search(pattern, text) for pattern in image_patterns)


def create_chemistry_image_prompt(user_request: str, retrieved_chunks=None) -> str:
    """
    Use GPT-OSS to turn the user's request into a concise prompt for an
    educational chemistry image generator.
    """
    retrieved_chunks = retrieved_chunks or []

    context = "\n\n".join(
        (
            f"Document type: {chunk.get('document_type_label', 'Reference')}\n"
            f"Source: {chunk['source']}\n"
            f"{chunk['text']}"
        )
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
# 8. Main Chat Interface
# ------------------------------------------------------------------------------
st.title("🧪 OCR A Level Chemistry AI Assistant")
st.caption(
    "Grounded on official OCR A specifications, data sheets, and mark schemes. Chemistry diagrams are rendered as clean vectors; other image requests can use AI Horde."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("type") == "diagram":
            display_chemistry_diagram(message["svg"])
            caption = message.get("caption", "")
            if caption:
                st.caption(caption)
        elif message.get("type") == "image":
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
    # Deterministic chemistry diagram route
    # --------------------------------------------------------------------------
    if is_diagram_request(user_input):
        with st.chat_message("assistant"):
            status_placeholder = st.empty()

            try:
                status_placeholder.info("Building a clean chemistry diagram...")

                diagram_context = retrieve_relevant_context(
                    user_input,
                    top_k=2,
                )

                diagram_spec = create_chemistry_diagram_spec(
                    user_input,
                    retrieved_chunks=diagram_context,
                )

                svg = render_chemistry_svg(diagram_spec)
                status_placeholder.empty()

                display_chemistry_diagram(svg)

                diagram_caption = diagram_spec.get("caption", "")
                if diagram_caption:
                    st.caption(diagram_caption)

                with st.expander("Diagram specification"):
                    st.json(diagram_spec)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "diagram",
                        "content": (
                            f"Generated a chemistry diagram for: {user_input}"
                        ),
                        "svg": svg,
                        "caption": diagram_caption,
                    }
                )

            except Exception as err:
                status_placeholder.empty()

                error_message = (
                    "I could not build that chemistry diagram. "
                    f"{str(err)}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )

        st.stop()

    # --------------------------------------------------------------------------
    # AI Horde image request route
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

    retrieved_chunks = retrieve_relevant_context(user_input, top_k=6)

    with st.chat_message("assistant"):
        # Format and display retrieved context cards
        if retrieved_chunks:
            with st.expander(
                f"🔍 Retrieved OCR Context ({len(retrieved_chunks)} matches)"
            ):
                for idx, chunk in enumerate(retrieved_chunks, 1):
                    source_type = chunk.get("document_type_label", "Reference")
                    st.markdown(
                        f"**Match #{idx}** · **{source_type}** · "
                        f"`{chunk['source']}` &nbsp;|&nbsp; "
                        f"**Relevance:** `{chunk['score']}%`"
                    )
        
                    # 👇 REPLACE st.info(formatted_text) WITH THIS:
                    render_chemistry_chunk(chunk["text"])
        
                    if idx < len(retrieved_chunks):
                        st.divider()

        context_str = "\n\n".join([
            (
                f"--- {c.get('document_type_label', 'Reference').upper()} ---\n"
                f"Source: {c['source']}\n"
                f"{c['text']}"
            )
            for c in retrieved_chunks
        ])

        augmented_system_prompt = SYSTEM_PROMPT

        if context_str:
            augmented_system_prompt += (
                "\n\nRETRIEVED OFFICIAL OCR CONTEXT:\n"
                "Use the document-type labels below when deciding how much "
                "authority to give each passage.\n\n"
                f"{context_str}"
            )

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
