import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Mental Health Support", page_icon="💙", layout="centered")

st.title("💙 Mental Health Support Chatbot")
st.caption("A safe space to talk. Everything is confidential.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "meta" not in st.session_state:
    st.session_state.meta = []

# Display chat history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and i // 2 < len(st.session_state.meta):
            meta = st.session_state.meta[i // 2]
            st.caption(
                f"Language: `{meta.get('language', '')}` | "
                f"Emotion: `{meta.get('emotion', '')}` | "
                f"Intent: `{meta.get('intent', '')}`"
            )

# Chat input
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # Show user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Build history for API (exclude current message)
    history = st.session_state.messages[:-1]

    # Call API
    with st.chat_message("assistant"):
        with st.spinner("..."):
            try:
                response = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "text": user_input,
                        "history": history
                    },
                    timeout=60
                )
                data = response.json()
                answer = data.get("response", "Sorry, something went wrong.")
                
                st.write(answer)
                st.caption(
                    f"Language: `{data.get('language', '')}` | "
                    f"Emotion: `{data.get('emotion', '')}` | "
                    f"Intent: `{data.get('intent', '')}`"
                )

                # Save to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "language": data.get("language", "")
                })
                st.session_state.meta.append({
                    "language": data.get("language", ""),
                    "emotion" : data.get("emotion", ""),
                    "intent"  : data.get("intent", "")
                })

            except Exception as e:
                st.error(f"Could not reach the API: {e}")

# Clear button
if st.session_state.messages:
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.meta     = []
        st.rerun()