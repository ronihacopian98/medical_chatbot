import uuid

import requests
import streamlit as st

API_URL = "http://localhost:9000/api/v1/chat"

# --- Page config ---
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Medical Chatbot")
st.caption("AI-powered medical information assistant. Not a substitute for professional medical advice.")

# --- Session state ---
# Session state persists data between reruns (like a page refresh)
# Every time user sends a message, Streamlit reruns the whole script
# Without session state, we'd lose the chat history on every rerun
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# --- Display chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if it's an assistant message
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.caption(f"**{source['source']}**")
                    st.caption(source["content"])
                    st.divider()

# --- Chat input ---
if prompt := st.chat_input("Ask a medical question..."):

    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call our FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": prompt,
                        "conversation_id": st.session_state.conversation_id,
                    },
                    timeout=120,  # LLM can be slow locally
                )
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander("📚 Sources"):
                        for source in sources:
                            st.caption(f"**{source['source']}**")
                            st.caption(source["content"])
                            st.divider()

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the backend. Make sure the API is running on port 9000.")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model is taking too long.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
