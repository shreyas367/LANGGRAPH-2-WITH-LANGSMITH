import sys
from pathlib import Path
import uuid

# Ensure local module directory is in sys.path
_current_dir = str(Path(__file__).resolve().parent)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot


# Utility functions
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state['thread_id'] = new_thread_id
    add_thread(new_thread_id)
    st.session_state['message_history'] = []


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('chatmessages', []) if state and state.values else []


st.title("Chatbot")

# Initialize message history
if "message_history" not in st.session_state:
    st.session_state["message_history"] = [
        {"role": "assistant", "content": "How can I assist you today?"}
    ]

# Initialize threads list
if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = []

# Dynamic thread ID for conversation in memory
if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

add_thread(st.session_state['thread_id'])


# Sidebar
st.sidebar.title('LANGGRAPH CHATBOT')

st.sidebar.button('new chat', on_click=reset_chat)

st.sidebar.header('my conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(thread_id, key=f"btn_{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []
        for m in messages:
            if m.type == 'human':
                temp_messages.append({'role': 'user', 'content': m.content})
            else:
                temp_messages.append({'role': 'assistant', 'content': m.content})
        st.session_state['message_history'] = temp_messages
        st.rerun()


# Display all past messages on each rerun
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# User input box
user_input = st.chat_input("Say something...")

if user_input:
    
    # Display & save user message
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    # Prepare configuration and invoke chatbot
    config = {"configurable": {"thread_id": st.session_state['thread_id']}}
    state = {"chatmessages": [HumanMessage(content=user_input)]}



    # Display & save assistant reply with streaming
    with st.chat_message("assistant"):
        reply = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                state, config, stream_mode="messages"
            )
            if hasattr(message_chunk, "content")
        )
    st.session_state["message_history"].append({"role": "assistant", "content": reply})
