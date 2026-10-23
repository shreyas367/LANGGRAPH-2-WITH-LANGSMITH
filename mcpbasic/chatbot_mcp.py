import asyncio
import sys
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient





load_dotenv()
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1000,
)


mcp_client=MultiServerMCPClient(
    {
        "Arithmatic":{
            "transport":"stdio",
            "command":"uv",
            "args":["run","mcpserver.py"]
        }
    }
)





# tools = [calculator]



# llm_with_tools = llm.bind_tools(tools)




class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():
    tools=await mcp_client.get_tools()

    print(tools)

    llm_with_tools=llm.bind_tools(tools)



    
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}


    tool_node = ToolNode(tools)

    # Build the graph
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Flow: START -> chat_node -> (conditional) -> tools -> chat_node OR -> END
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()
    return chatbot


async def main():
    chatbot=await build_graph()

    result=await chatbot.ainvoke({'messages': [HumanMessage(content="hey, what is the square root of 1546? ")]})
    print(result["messages"][-1].content)


if __name__ == '__main__':
    asyncio.run(main())

