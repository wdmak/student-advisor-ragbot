"""
Streamlit chat UI for the internal knowledge bank chatbot.

Run with:
    streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from agent import build_agent

CHROMA_DIR = Path(__file__).parent / "chroma_db"


def extract_text_from_message(message):
    """Return readable assistant text from LangChain/Gemini messages.

    Some model/tool-call responses arrive as a list of dict blocks rather than
    a plain string. Rendering those directly can produce JSON-like content in the
    UI instead of the final answer.
    """
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    text_parts.append(block["text"])
                elif isinstance(block.get("content"), str):
                    text_parts.append(block["content"])
        joined = "\n".join(part for part in text_parts if part).strip()
        if joined:
            return joined

    if isinstance(content, dict):
        for key in ("text", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value

    return str(content) if content is not None else ""

st.set_page_config(page_title="Job Hunt Knowledge Bank", page_icon="💼")
st.title("Job hunt knowledge bank")
st.caption(
    "Ask about interview questions, internship experiences, or techniques "
    "students have shared. Answers are grounded in the knowledge base and cited."
)

if not CHROMA_DIR.exists():
    st.warning("No knowledge base found yet. Run `python ingest.py` first, then restart this app.")
    st.stop()


@st.cache_resource
def get_agent():
    return build_agent()


agent_executor = get_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask about job hunting experiences..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the knowledge base..."):
            # create_agent takes/returns a "messages" list in OpenAI-style
            # role dicts (or BaseMessage objects) — the full running
            # conversation is passed in each time since this app has no
            # checkpointer.
            result = agent_executor.invoke({"messages": st.session_state.messages})

            answer = ""
            for message in reversed(result["messages"]):
                if getattr(message, "type", None) == "ai":
                    extracted = extract_text_from_message(message)
                    if extracted.strip():
                        answer = extracted
                        break

            if not answer:
                for message in reversed(result["messages"]):
                    extracted = extract_text_from_message(message)
                    if extracted.strip():
                        answer = extracted
                        break

            if not answer:
                answer = "I couldn't produce a clear answer from the knowledge base."

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
