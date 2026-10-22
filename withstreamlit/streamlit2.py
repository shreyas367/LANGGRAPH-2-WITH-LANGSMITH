import sys
from pathlib import Path
import uuid
import importlib

# Ensure local module directory is in sys.path
_current_dir = str(Path(__file__).resolve().parent)
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)


import os 
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

os.environ['LANGCHAIN_PROJECT']="langgraph_with_sqlite_in_streamlit"

import streamlit as st
from langchain_core.messages import HumanMessage

# Reload module to ensure updates are picked up by Streamlit
import langgraph_database
importlib.reload(langgraph_database)
from langgraph_database import chatbot, checkpointer


# Utility functions
def generate_thread_id():
    return str(uuid.uuid4())


def retrieve_all_threads():
    all_threads = []
    seen = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id not in seen:
            seen.add(thread_id)
            all_threads.append(thread_id)
    return all_threads


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    return state.values.get('chatmessages', []) if state and state.values else []


def reset_chat():
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['message_history'] = [
        {"role": "assistant", "content": "How can I assist you today?"}
    ]


st.title("Chatbot")

# Always fetch all saved conversations from the SQLite database
all_saved_threads = retrieve_all_threads()
st.session_state['chat_threads'] = all_saved_threads


# On fresh load/refresh, load the latest conversation if available
if 'thread_id' not in st.session_state or 'message_history' not in st.session_state:
    if all_saved_threads:
        latest_thread = all_saved_threads[0]
        st.session_state['thread_id'] = latest_thread
        saved_messages = load_conversation(latest_thread)
        st.session_state['message_history'] = [
            {'role': 'user' if m.type == 'human' else 'assistant', 'content': m.content}
            for m in saved_messages
        ]
    else:
        st.session_state['thread_id'] = generate_thread_id()
        st.session_state['message_history'] = [
            {"role": "assistant", "content": "How can I assist you today?"}
        ]


# Sidebar
st.sidebar.title('LANGGRAPH CHATBOT')

st.sidebar.button('➕ New Chat', on_click=reset_chat, use_container_width=True)

st.sidebar.header('My Conversations')

# Display all conversations in the sidebar
for thread_id in st.session_state['chat_threads']:
    messages = load_conversation(thread_id)
    
    first_msg = next((m.content for m in messages if m.type == 'human' and m.content), None)
    if first_msg:
        clean_title = str(first_msg).strip().replace("\n", " ")
        short_title = (clean_title[:24] + "..") if len(clean_title) > 24 else clean_title
        btn_label = f"💬 {short_title}"
    else:
        btn_label = f"💬 Chat ({thread_id[:6]})"

    # Mark the currently active conversation
    if thread_id == st.session_state['thread_id']:
        btn_label = f"👉 {btn_label}"

    if st.sidebar.button(btn_label, key=f"btn_{thread_id}", use_container_width=True):
        st.session_state['thread_id'] = thread_id
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
    # config = {"configurable": {"thread_id": st.session_state['thread_id']}}
    CONFIG={
        "configurable":{
            "thread_id":st.session_state['thread_id']
        },
        "metadata":{
            "thread_id":st.session_state['thread_id'],
        },
        "run_name":"streamlitchat"
    }
    state = {"chatmessages": [HumanMessage(content=user_input)]}

    # Display & save assistant reply with streaming
    with st.chat_message("assistant"):
        reply = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                state, config=CONFIG, stream_mode="messages"
            )
            if hasattr(message_chunk, "content")
        )
    st.session_state["message_history"].append({"role": "assistant", "content": reply})
    st.rerun()
