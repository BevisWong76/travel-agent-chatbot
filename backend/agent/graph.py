import os
import re
from typing import Literal
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from tools import ACTION_TOOLS
from agent.state import TravelAgentState, build_agent_context
from services.rag_engine import query_rag
from services.crawler import search_and_crawl

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/dbname")

# =====================================================================
# 1. LLM & Tools Setup
# =====================================================================
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY")
)
fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    api_key=os.getenv("GEMINI_API_KEY")
)
# for lite models, sampling parameters are not supported, so we don't set temperature or top_p here

llm = primary_llm.with_fallbacks([fallback_llm])

# Bind the LLM with all available tools for parallel execution
llm_with_tools = llm.bind_tools(ACTION_TOOLS)

SYSTEM_INSTRUCTION = """
You are an elite, highly efficient AI Travel Agent.

CRITICAL OPERATIONAL RULES:
1. Ground your answers using the provided [Retrieved Context] and state details.
2. PARALLEL TOOL CALLING: If you need live action data (e.g., weather forecast, map routes), call all necessary tools in a single turn.
3. STATE UPDATES: Whenever the user provides new travel details/preferences (destination, dates, budget, food style) OR when you retrieve weather information, use `update_travel_state_tool` to update the state immediately (including `weather_forecast`).
4. ITINERARY OUTPUT RULE: Whenever you generate or update an itinerary, put the ENTIRE itinerary structure (including its title, headers, and daily breakdown) INSIDE the <itinerary> and </itinerary> tags. Do NOT put the itinerary title outside the tags.
5. CONVERSATIONAL INTRO RULE: Before opening the <itinerary> tag, ALWAYS provide a friendly, concise response in normal text. Directly answer any user questions (e.g., weather forecasts, budget tips, local recommendations) or highlight key adjustments made to their trip in this intro text. Never output ONLY the <itinerary> block without conversational text preceding it.

Example Output:
I've checked the forecast for Hong Kong—temperatures will be around 19°C-24°C with a chance of light spring drizzle, so I've added a few cozy indoor alternatives alongside the street food spots! 

<itinerary>
# 3-Day Low-Budget Romantic Dating Itinerary
## Day 1...
</itinerary>
"""

# Notes:
# We use XML tags to allow the backend to extract the itinerary from the LLM's response.
# Another approach could be to have the LLM return a structured JSON object.
# JSON Mode have slightly higher stability, but also higher latency, so we prefer the XML tag approach for now.
# JSON Mode is better for API-to-API communication, while XML tag extraction is better for human-facing chat interfaces.

# =====================================================================
# 2. Nodes Definition
# =====================================================================

# Helper Function to Extract Text Content from LLM Responses
def extract_text_content(content) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return "".join(text_parts)
    return str(content) if content else ""

# Node 1: Code-level CRAG (Process Knowledge Base / Crawler)
async def retrieve_or_crawl_node(state: TravelAgentState):
    messages = state.get("messages", [])
    raw_content = messages[-1].content if messages else ""
    user_query = extract_text_content(raw_content)
    
    # 1. Check Pinecone for relevant context
    rag_context = await query_rag(user_query)
    
    # 2. If Pinecone returns relevant context, use it
    if rag_context and "No relevant internal guides" not in rag_context:
        print("[CRAG] Pinecone Hit!")
        return {"retrieved_context": rag_context}
    
    # 3. If Pinecone returns no relevant context, trigger the web crawler
    print("[CRAG] Pinecone Miss -> Triggering Crawler...")
    crawled_docs = await search_and_crawl(user_query, max_results=2)
    crawl_context = "\n\n".join([f"Source: {doc.metadata.get('source_url', 'Unknown')}\nContent: {doc.text.strip()}" for doc in crawled_docs])
    
    return {"retrieved_context": crawl_context}


# Node 2: Agent Reasoning Node (Process Tools Calls and Generate Final Response)
async def call_agent_node(state: TravelAgentState, config: RunnableConfig):
    print("[Agent] Building context for LLM...")
    context_str = build_agent_context(state)
    retrieved_info = state.get("retrieved_context", "No retrieved docs.")
    
    # Prepare the system prompt with the current state context and retrieved info
    full_system_prompt = SystemMessage(
        content=f"{SYSTEM_INSTRUCTION}\n\n[Retrieved Context]\n{retrieved_info}\n\n[State Context]\n{context_str}"
    )

    # Context Windowing: Trim messages to fit within LLM's context window
    messages = state.get("messages", [])
    trimmed_messages = trim_messages(
        messages,
        max_tokens=5000,
        strategy="last",
        token_counter=len,
        allow_partial=True,
        start_on="human",
        include_system=False,
    )
    prompt_messages = [full_system_prompt] + trimmed_messages
    
    # Invoke the LLM with tools
    print("[Agent] Invoking LLM with tools...")
    response = await llm_with_tools.ainvoke(prompt_messages, config)

    # Extract the content text from the response, handling both string and list formats
    content_text = extract_text_content(response.content)

    # Extract Itinerary by locating XML Tag
    print("[Agent] Extracting itinerary from LLM response...")
    new_itinerary = None
    if content_text:
        match = re.search(r"<itinerary>(.*?)</itinerary>", content_text, re.DOTALL)
        if match:
            new_itinerary = match.group(1).strip()

    return {
        "messages": [response],
        "draft_itinerary": new_itinerary
    }

# =====================================================================
# 3. Conditional Routing Function
# =====================================================================
def should_continue(state: TravelAgentState) -> Literal["tools", "__end__"]:
    print("[Routing] Evaluating whether to continue to tools or end the conversation...")
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    # if the last message contains tool calls, route to "tools" node; otherwise, end the conversation
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("[Routing] Tool calls detected in the last message. Routing to 'tools' node.")
        return "tools"
    
    print("[Routing] No tool calls found. Ending conversation.")
    return END

# =====================================================================
# 4. StateGraph Assembly
# =====================================================================
builder = StateGraph(TravelAgentState)

# Add Nodes
builder.add_node("retrieve_or_crawl", retrieve_or_crawl_node)
builder.add_node("agent", call_agent_node)
builder.add_node("tools", ToolNode(ACTION_TOOLS)) 

# Add Edges
builder.add_edge(START, "retrieve_or_crawl")      # 1. Always start with CRAG to retrieve context or crawl
builder.add_edge("retrieve_or_crawl", "agent")    # 2. Pass the context to the Agent for reasoning and tool invocation

# 3. Conditional Routing: If the Agent calls tools, route to "tools" node; otherwise, end the conversation
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)
builder.add_edge("tools", "agent")                 # 4. Tool results are passed back to the Agent for final response generation

# =====================================================================
# 5. Checkpointer & Compile Graph
# =====================================================================

# Checkpointer: In-memory for development; 
# checkpointer = MemorySaver()
# travel_agent_app = builder.compile(checkpointer=checkpointer)

# Checkpointer: PostgreSQL for production
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
pool = AsyncConnectionPool(conninfo=DATABASE_URL, kwargs=connection_kwargs, open=False)

async def init_app():
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    return builder.compile(checkpointer=checkpointer)