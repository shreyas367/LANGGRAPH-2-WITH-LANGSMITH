import sys
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, Annotated
from typing import TypedDict
import sqlite3


load_dotenv(find_dotenv())


class botstate(TypedDict):
    chatmessages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(model="openai/gpt-oss-120b")


def chat_node(state: botstate):
    messages = state["chatmessages"]
    response = llm.invoke(messages)
    return {"chatmessages": [response]}


# conn = sqlite3.connect(database='chatbot.db',check_same_thread=False)
# checkpointer = SqliteSaver(conn)
checkpointer = SqliteSaver(sqlite3.connect("chatbot.db", check_same_thread=False))


graph = StateGraph(botstate)
graph.add_node("llmnode", chat_node)
graph.add_edge(START, "llmnode")
graph.add_edge("llmnode", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = []
    seen = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config['configurable']['thread_id']
        if thread_id not in seen:
            seen.add(thread_id)
            all_threads.append(thread_id)
    return all_threads


