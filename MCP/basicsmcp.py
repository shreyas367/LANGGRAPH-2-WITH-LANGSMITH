from basicagent import llm_with_tools
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage,HumanMessage

from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,ToolNode
from langchain_core.tools import tool


load_dotenv()

@tool
def calculator(first_num: float,second_num:float,operator:str)->float:
    """
    this function is used to perform basic arithmetic operations like 
    addition, subtraction, multiplication and division on two numbers
    """
    if operator == "add":
        return first_num + second_num
    elif operator == "sub":
        return first_num - second_num
    elif operator == "mul":
        return first_num * second_num
    elif operator == "div":
        if second_num == 0:
            raise ValueError("Division by zero is not allowed")
        return first_num / second_num
    else:
        raise ValueError("Invalid operator. Please use 'add', 'sub', 'mul', or 'div'")

tools=[calculator]

llm=ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1000,
)


llm_with_tools=llm.bind_tools(tools)


