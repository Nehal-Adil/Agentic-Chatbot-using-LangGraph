import streamlit as st
import uuid
from langchain_core.messages import HumanMessage
from chatbot_backend import chatbot

st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="centered")

st.title("🤖 Agentic Chatbot with LangGraph")

# 1. Initialize persistent session variables
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar options
with st.sidebar:
    st.subheader("Session Controls")
    st.caption(f"Thread ID: `{st.session_state.thread_id[:8]}...`")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# 2. Render previous messages in UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle new user input
if user_input := st.chat_input("Type your message..."):
    # Display and record user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare config for LangGraph memory
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Stream response from LangGraph
    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""

        # Stream chunks directly from the chat node
        for chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
            stream_mode="messages"
        ):
            if chunk.content:
                full_response += chunk.content
                response_container.markdown(full_response + "▌")

        response_container.markdown(full_response)

    # Record assistant message in UI state
    st.session_state.messages.append({"role": "assistant", "content": full_response})