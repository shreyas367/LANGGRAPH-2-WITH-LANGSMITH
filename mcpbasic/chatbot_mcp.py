import asyncio
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1000,
)

mcp_client = MultiServerMCPClient(
    {
        "Arithmatic": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "mcpserver.py"],
        }
    }
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():
    tools = await mcp_client.get_tools()

    print("===Available MCP tools ===")
    for t in tools:
        print(t.name)
    print("\n")

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]

        system_message = SystemMessage(
            content=(
                "You are an arithmetic assistant with access to tools. "
                "Use tools for arithmetic operations. "
                "Perform only operations explicitly requested by the user. "
                "Do not introduce additional calculations or expressions. "
                "Use each tool result as the input for subsequent requested "
                "operations. After all requested operations are complete, "
                "provide the final answer."
            )
        )

        response = await llm_with_tools.ainvoke([system_message] + messages)

        print("\n=== LLM RESPONSE ===")
        print("Content:", response.content)
        print("Tool calls:", response.tool_calls)

        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()
    return chatbot


async def main():
    chatbot = await build_graph()

    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the square root of 1546? Do not calculate it yourself; use tools."
                )
            ]
        }
    )

    print("\n=== COMPLETE MESSAGE HISTORY ===")

    for message in result["messages"]:
        print("\nMessage type:", type(message).__name__)
        print("Content:", message.content)

        if hasattr(message, "tool_calls") and message.tool_calls:
            print("Tool calls:", message.tool_calls)

        if message.type == "tool":
            print("Tool result:", message.content)


if __name__ == "__main__":
    asyncio.run(main())