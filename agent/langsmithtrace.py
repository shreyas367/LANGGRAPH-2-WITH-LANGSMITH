import os
from dotenv import load_dotenv
load_dotenv()
import requests

os.environ['LANGCHAIN_PROJECT']="myagentDEMO"



from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


@tool
def get_weather(city:str) ->str:
    """get current weather for a city"""
    
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        raise ValueError("OPENWEATHER_API_KEY is not set in the environment variables.")
    
    url=(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric")
    if not url:
        raise ValueError("URL is not set correctly.")
    
    response=requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Error fetching weather data: {response.status_code} - {response.text}")
    
    data=response.json()
    if "main" not in data or "temp" not in data["main"]:
        raise ValueError("Temperature data is missing in the response.")
    
    temperature=data["main"]["temp"]
    return f"The current temperature in {city} is {temperature}°C."



tavily=TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=3,
    topic="news")

@tool
def search_news(query:str) ->str:
    """geting news articles based on a query"""
    if not query:
        print("Query is empty. Please provide a valid query.")
        return "No query provided."
    
    return tavily.invoke({
        "query":f"latest news about {query}",
    })

llmgroq=ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-120b",
    verbose=True,
    temperature=0.4,
    max_tokens=500
)

tools = [get_weather, search_news]
llm_with_tools = llmgroq.bind_tools(tools)


def call_model(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


workflow = StateGraph(MessagesState)
workflow.add_node("model", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_edge(START, "model")
workflow.add_conditional_edges("model", tools_condition)
workflow.add_edge("tools", "model")

myagent = workflow.compile(debug=True)

config={
    'tags':['my first agent','weather and news tools'],
    'run_name':'basicagentwithsmithtrace'
}

while True:
    user_input=input("user : ")
    if user_input.lower() == "exit" or user_input.lower() == "quit":
        print("Exiting the agent. Goodbye!")
        break
    
    else:
        response=myagent.invoke({"messages": [{"role":"user","content":user_input}]},config=config)
        print(f"Agent: {response['messages'][-1].content}")