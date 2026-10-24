from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command
from dotenv import load_dotenv

from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import requests
import math
import os

import pypdf
from langchain_community.vectorstores import FAISS 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

load_dotenv()

# NVIDIA LLM Model
llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-super-vl-120b-a12b",
    temperature=0,
)

# Embedding Model
embeddings_model = NVIDIAEmbeddings(
    model="nvidia/nemotron-3.5-embed-1b",
    truncate="END"
)

# document loader
def load_pdf_modern(file_path: str):
    reader = pypdf.PdfReader(file_path)
    documents = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            # Wrap raw text straight into LangChain's core stable Document object
            documents.append(Document(page_content=text, metadata={"page": page_num}))
    return documents

def ingest_rag_document(file_path):
    DB_PATH = "faiss_db"
    raw_docs = load_pdf_modern(file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)

    if os.path.exists(DB_PATH):
        vector_store = FAISS.load_local(
            DB_PATH, embeddings_model, allow_dangerous_deserialization=True
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings_model)

    vector_store.save_local(DB_PATH)

def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    return retriever

# Tools
search_tool = TavilySearch(
    max_results=3,
    topic="general",
    search_depth="advanced"
)

# rag tool
@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions that may
    be answered by a PDF document. 

    Args:
        query: The question or search query used to retrieve information from the PDF document.
    """
    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information found."

    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page", "Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}\n"
        )

    return "\n\n".join(formatted_documents)

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

@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of stocks.
    HUMAN-IN-THE-LOOP:
    Before purchasing, ask the user for confirmation.
    """

    # This pauses the graph and returns control to the caller
    decision = interrupt("Approve buying {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} cancelled by user.",
            "symbol": symbol,
            "quantity": quantity,
        }

# Make a tool list
tools = [get_stock_price, search_tool, calculator, rag_tool, purchase_stock]

# Make the LLM tool-aware
llm_with_tools = llm.bind_tools(tools)

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

# graph nodes
def chat_node(state: ChatState):
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to serveral tools.\n\n"

            "Tool usage instructions:\n"
            "- Use `rag_tool` for questions about the uploaded PDF or document."
            "Always retrieve relevant document content before answering PDF-related questions.\n"
            "- Use `search_tool` for current events, recent information, or information that requires an internet search.\n"
            "- Use `calculator` for mathematical calculations. Do not calculate complex expressions manually when the calculator is available.\n"
            "- Use `get_stock_price` when the user asks for the current price of a stock.\n"
            "Answer general questions directly when no tool is required."
            "Do not invent information from the uploaded document."
            "If the user asks about a PDF but no document is available, ask them to upload a PDF."
            "After receiving a tool result, provide a clear and helpful final answer."
        )
    )

    messages = [
        system_message,
        *state['messages']
    ]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

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
