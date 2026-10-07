import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="OCR Chemistry Tutor", page_icon="🧪")
st.title("🧪 OCR A Level Chemistry AI Assistant")

# Initialize OpenAI client with Groq's endpoint
api_key = st.secrets.get("GROQ_API_KEY") or st.sidebar.text_input("Groq API Key", type="password")

if not api_key:
    st.warning("Please provide a Groq API Key.")
    st.stop()

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key
)

SYSTEM_INSTRUCTION = """
You are an elite OCR A Level Chemistry (H432) specialist tutor.
Base answers strictly on the official OCR A specification and past mark schemes.
Highlight required exam keywords in **bold**.
"""

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask an OCR Chemistry question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                *st.session_state.messages
            ],
            stream=True,
            temperature=0.1
        )
        
        def response_generator():
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        full_response = st.write_stream(response_generator())

    st.session_state.messages.append({"role": "assistant", "content": full_response})
