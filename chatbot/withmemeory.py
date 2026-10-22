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

thread_id = "1"
config = {"configurable": {"thread_id": thread_id}}

print("Chatbot initialized! Type 'exit', 'quit', or press Ctrl+C to stop.\n")

try:
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ["exit", "quit"]:
            print("\nExiting the chatbot. Goodbye!")
            break

        state = {"chatmessages": [HumanMessage(content=user_input)]}
        response = chatbot.invoke(state, config)
        print(f"Agent: {response['chatmessages'][-1].content}\n")

except KeyboardInterrupt:
    print("\nChat stopped by user (Ctrl+C).")



# Retrieve and inspect the memory snapshot
state_snapshot = chatbot.get_state(config)

print("\n" + "=" * 50)
print("--- COMPLETE STATE SNAPSHOT ---")
print("=" * 50)
print(f"Next Node to Run: {state_snapshot.next}")
print(f"Config / Checkpoint: {state_snapshot.config}")
print(f"Created At: {state_snapshot.created_at}")

print("\n--- MESSAGES STORED IN MEMORY ---")
for i, msg in enumerate(state_snapshot.values.get("chatmessages", []), start=1):
    role = msg.__class__.__name__
    print(f"[{i}] {role}: {msg.content}")
