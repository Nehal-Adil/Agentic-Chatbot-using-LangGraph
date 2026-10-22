from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph.message import add_messages
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import requests
import math
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

# NVIDIA LLM Model
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-super-vl-120b-a12b",
    temperature=0,
)

# Tools
search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced"
)

@tool
def calculator(expression: str) -> str:
    """
    Useful for simple math calculations.
    """ 

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation Error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol. (e.g. "AAPL", "TSLA")
    using Alpha Vantage with API key in the URL.
    """ 
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=INVO20V2O2GYTRC2"
    response = requests.get(url)
    return response.json()

# Make a tool list
tools = [get_stock_price, search_tool, calculator]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

# graph nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state['messages']
    
    response = llm_with_tools.invoke(messages)
    
    return {'messages': [response]}

tool_node = ToolNode(tools) # Execute tool calls

# connect to database
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

# Add Graph
graph = StateGraph(ChatState)

graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

#add edges
graph.add_edge(START, 'chat_node')

# If the LLM asked for a tool, go to ToolNode; else finish
graph.add_conditional_edges('chat_node', tools_condition)

graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()
    for thread in checkpoint.list(None):
        all_threads.add(thread.config['configurable']['thread_id'])

    return list(all_threads)
     
