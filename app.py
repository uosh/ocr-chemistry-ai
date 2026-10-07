import os
import streamlit as st
from google import genai
from google.genai import types

# Page Config
st.set_page_config(page_title="OCR A Chemistry Tutor", page_icon="🧪")
st.title("🧪 OCR A Level Chemistry AI Assistant")
st.caption("Grounding strictly on official OCR H432 Specification & Mark Schemes")

# 1. API Key Setup
api_key = None
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    api_key = os.environ["GEMINI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

if not api_key:
    st.warning("⚠️ API Key missing. Please set GEMINI_API_KEY in Streamlit Cloud Secrets.")
    st.stop()

# Cache Gemini Client so it doesn't re-instantiate on every Streamlit rerun
@st.cache_resource
def get_gemini_client(key):
    return genai.Client(api_key=key)

client = get_gemini_client(api_key)

# 2. System Instruction
SYSTEM_INSTRUCTION = """
You are an elite OCR A Level Chemistry (H432) specialist tutor.
Your single purpose is to help students strictly according to the official OCR A specification, data sheet, and past mark schemes.

STRICT OPERATIONAL RULES:
1. OCR ONLY: Base answers strictly on OCR A guidelines. Enforce OCR conventions over AQA/Edexcel.
2. Mark Scheme Keywords: Always highlight required exam keywords in **bold** (e.g., **heterolytic fission**, **curly arrow starting from lone pair/bond**).
3. Spec Codes: Include official OCR specification references (e.g., Module 3.1.2 (a)).
4. Out of Scope: State clearly if a question is outside the official OCR A Chemistry specification.
5. Mathematical Precision: For physical chemistry calculations, present full step-by-step working matching OCR mark scheme layouts.
"""

# 3. Chat State Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Fast Streaming Generator
def stream_gemini_response(prompt):
    """Streams tokens in real time to eliminate perceived latency."""
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.1,
        )
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text

# 5. Chat Input & Streamed Rendering
if prompt := st.chat_input("Ask an OCR A Chemistry question..."):
    # Render user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream assistant response instantly
    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_gemini_response(prompt))
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
