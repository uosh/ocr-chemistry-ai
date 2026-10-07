import os
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError

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
    st.warning("⚠️ GEMINI_API_KEY is missing. Please add it to your Streamlit Secrets (`.streamlit/secrets.toml`) or enter it in the sidebar.")
    st.stop()

# Initialize Cached Client
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

# 3. Chat History Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Safe Fast Streaming Generator Function
def stream_gemini_response(prompt):
    """Streams response from Gemini 3.8 Flash with error handling for API issues."""
    try:
        response = client.models.generate_content_stream(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.1,
            )
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except APIError as e:
        yield f"\n\n⚠️ **Gemini API Error ({e.code})**: {e.message}"
    except Exception as e:
        yield f"\n\n⚠️ **Unexpected Error**: {str(e)}"

# 5. User Input and Handling
if prompt := st.chat_input("Ask an OCR A Chemistry question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        full_response = st.write_stream(stream_gemini_response(prompt))
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
