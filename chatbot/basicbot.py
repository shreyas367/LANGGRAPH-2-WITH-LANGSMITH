from langchain_groq import ChatGroq
from langgraph.graph import StateGraph,START,END
from typing import TypedDict
from langchain_core.messages import BaseMessage,HumanMessage
from langgraph.graph.message import add_messages,Annotated
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class botstate(TypedDict):
    chatmessages: Annotated[list[BaseMessage], add_messages]


llm = ChatGroq(model="openai/gpt-oss-120b")


def chat_node(state: botstate):
    messages = state['chatmessages']
    response = llm.invoke(messages)
   
    return {'chatmessages': [response]}


graph = StateGraph(botstate)

graph.add_node("llmnode", chat_node)

graph.add_edge(START, "llmnode")
graph.add_edge("llmnode", END)

chatbot = graph.compile()


while True:
    user_input=input("You: ")
    if user_input.lower() == "exit" or user_input.lower() == "quit":
        print("Exiting the chatbot. Goodbye!")
        break
    else:
        state = {"chatmessages": [HumanMessage(content=user_input)]}
        response = chatbot.invoke(state)
        print(f"Agent: {response['chatmessages'][-1].content}")





# initital_state = {
#     'chatmessages': [HumanMessage(content='what is capital of india')]
# }

# result = chatbot.invoke(initital_state)

# print(result['chatmessages'][-1].content)




