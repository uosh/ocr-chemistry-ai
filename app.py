import os
import streamlit as st
from google import genai
from google.genai import types

# Page Config
st.set_page_config(page_title="OCR A Chemistry Tutor", page_icon="🧪")
st.title("🧪 OCR A Level Chemistry AI Assistant")
st.caption("Grounding strictly on official OCR H432 Specification & Mark Schemes")

# 1. API Key Setup (Explicitly check Streamlit Secrets first)
api_key = None

if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("⚠️ API Key missing. Please set GEMINI_API_KEY in Streamlit Cloud Secrets or enter it in the sidebar.")
    st.stop()

# Initialize Gemini Client
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Failed to initialize Gemini client: {e}")
    st.stop()

# 2. System Instructions for OCR A Grounding
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

# 3. Document Processing
def get_ocr_documents():
    """Uploads local OCR PDFs from 'ocr_files' folder to Gemini File API."""
    uploaded_files = []
    folder = "ocr_files"
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(".pdf"):
                filepath = os.path.join(folder, filename)
                try:
                    g_file = client.files.upload(file=filepath)
                    uploaded_files.append(g_file)
                except Exception as e:
                    st.sidebar.error(f"Error uploading {filename}: {e}")
    return uploaded_files

with st.sidebar:
    st.header("OCR Knowledge Base")
    if st.button("Index OCR PDFs"):
        with st.spinner("Uploading OCR files..."):
            files = get_ocr_documents()
            if files:
                st.success(f"Indexed {len(files)} OCR PDFs into Gemini context!")
            else:
                st.info("No PDF files found in 'ocr_files' folder.")

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask an OCR A Chemistry question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            try:
                ocr_files = get_ocr_documents()
                contents = ocr_files + [prompt]
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_INSTRUCTION,
                        temperature=0.1,
                    )
                )
                
                st.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                st.error(f"Gemini API Error: {e}")
