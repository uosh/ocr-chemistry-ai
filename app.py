import os
import streamlit as st
from openai import OpenAI

# 1. Streamlit Page Configuration
st.set_page_config(
    page_title="OCR Chemistry AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. OCR A Level Chemistry System Prompt
SYSTEM_PROMPT = """You are an expert OCR A Level Chemistry AI Assistant, specialized in helping students master the OCR Chemistry A and B specifications.

Key Guidelines:
1. Provide precise, step-by-step explanations for Physical, Inorganic, and Organic Chemistry topics.
2. Align all definitions, key terms, and reaction mechanisms directly with official OCR mark schemes (e.g., exact wording for enthalpy definitions, electron pair repulsion theory, curly arrow mechanisms).
3. Use LaTeX formatting ($...$ for inline formulas, $$...$$ for standalone equations) for mathematical calculations (e.g., pH, Ka, Arrhenius, rate equations, mole calculations).
4. Highlight common student pitfalls and OCR exam tips whenever relevant.
"""

# 3. Sidebar Configuration
st.sidebar.title("🧪 OCR Chemistry AI")
st.sidebar.markdown("Powered by **Llama 3.3 70B** via Groq")

# Retrieve API key from secrets or environment variables
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.sidebar.error("⚠️ `GROQ_API_KEY` not detected.")
    st.info("Add `GROQ_API_KEY = \"your_api_key_here\"` in Streamlit Secrets (`.streamlit/secrets.toml`).")
    st.stop()

# Initialize OpenAI client with Groq base URL
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=groq_api_key
)

# Clear Conversation Button
if st.sidebar.button("🗑️ Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("💡 Quick Prompts")
prompt_suggestions = [
    "Explain the nucleophilic substitution mechanism of haloalkanes.",
    "How do I calculate the pH of a weak acid buffer solution?",
    "What is the official OCR definition for first ionisation energy?",
    "Summarise transition metal ligand substitution reactions and color changes."
]

for suggestion in prompt_suggestions:
    if st.sidebar.button(suggestion, use_container_width=True):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": suggestion})
        st.rerun()

# 4. Main Chat Interface
st.title("🧪 OCR A Level Chemistry Assistant")
st.caption("Ask questions on mechanisms, calculations, OCR mark scheme definitions, or practical skills.")

# Initialize chat history state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Process User Input
if user_input := st.chat_input("Ask a chemistry question (e.g., 'Explain optical isomerism')..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate Response from Groq
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Construct full context for the model
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
            {"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages
        ]

        try:
            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=api_messages,
                temperature=0.2,
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
