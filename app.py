import os
import streamlit as st
from google import genai
from google.genai import types

# Page Config
st.set_page_config(page_title="OCR A Chemistry Tutor", page_icon="🧪")
st.title("🧪 OCR A Level Chemistry AI Assistant")
st.caption("Grounding strictly on official OCR H432 Specification & Mark Schemes")

# 1. API Key Setup
api_key = os.environ.get("GEMINI_API_KEY") or st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.info("Please enter your Gemini API Key in the sidebar or set GEMINI_API_KEY in your environment.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=api_key)

# 2. System Instructions for Strict OCR Grounding
SYSTEM_INSTRUCTION = """
You are an elite OCR A Level Chemistry (H432) specialist tutor.
Your single purpose is to help students strictly according to the official OCR A specification, data sheet, and past mark schemes uploaded in context.

STRICT OPERATIONAL RULES:
1. OCR ONLY: Base your answers ONLY on OCR A guidelines. If a topic, phrasing, or mechanism convention differs from other boards (AQA/Edexcel), enforce the OCR A standard.
2. Mark Scheme Keywords: OCR mark schemes require specific phrasing. Always highlight required exam keywords in **bold** (e.g., **heterolytic fission**, **curly arrow starting from lone pair/bond**, **electron pair donor**).
3. Spec Codes: Include official OCR specification references (e.g., Module 3.1.2 (a)) when explaining concepts.
4. Out of Scope: If a question is outside the OCR A specification, state clearly: "This topic is outside the official OCR A Chemistry (H432) specification."
5. Mathematical Precision: For physical chemistry calculations (enthalpy, Kc/Kp, pH, rate equations), present full step-by-step working matching OCR mark scheme layouts.
"""

# 3. Document Processing (Uploads to Gemini File API)
@st.cache_resource
def upload_ocr_documents():
    """Uploads local OCR PDFs to Gemini File API once."""
    uploaded_files = []
    folder = "ocr_files"
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".pdf"):
                filepath = os.path.join(folder, filename)
                st.sidebar.text(f"Uploading {filename}...")
                # Upload PDF directly to Gemini File API
                g_file = client.files.upload(file=filepath)
                uploaded_files.append(g_file)
    return uploaded_files

with st.sidebar:
    st.header("OCR Knowledge Base")
    if st.button("Index OCR PDFs"):
        files = upload_ocr_documents()
        st.success(f"Indexed {len(files)} OCR PDFs into Gemini context!")

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask an OCR A Chemistry question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching OCR Specification & Mark Schemes..."):
            # Gather uploaded files if any exist
            ocr_files = upload_ocr_documents()
            
            # Combine documents with prompt for grounded generation
            contents = ocr_files + [prompt]
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,  # Low temperature prevents non-OCR hallucinations
                )
            )
            
            st.markdown(response.text)
            st.session_state.messages.append({"role": "assistant", "content": response.text})
