import sys
from dotenv import load_dotenv, find_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, Annotated
from typing import TypedDict



load_dotenv(find_dotenv())


class botstate(TypedDict):
    chatmessages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(model="openai/gpt-oss-120b")


def chat_node(state: botstate):
    messages = state["chatmessages"]
    response = llm.invoke(messages)
    return {"chatmessages": [response]}


checkpointer = MemorySaver()

graph = StateGraph(botstate)
graph.add_node("llmnode", chat_node)
graph.add_edge(START, "llmnode")
graph.add_edge("llmnode", END)

chatbot = graph.compile(checkpointer=checkpointer)