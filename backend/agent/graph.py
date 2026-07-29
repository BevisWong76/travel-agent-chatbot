import os
import re
from typing import Literal
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from tools import ACTION_TOOLS
from agent.state import TravelAgentState, build_agent_context
from services.rag_engine import query_rag
from services.crawler import search_and_crawl

load_dotenv()

# =====================================================================
# 1. LLM & Tools Setup
# =====================================================================
primary_llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.3,
    api_key=os.getenv("GEMINI_API_KEY")
)
fallback_llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.3,
    api_key=os.getenv("GEMINI_API_KEY")
)
llm = primary_llm.with_fallbacks([fallback_llm])

# Bind the LLM with all available tools for parallel execution
llm_with_tools = llm.bind_tools(ACTION_TOOLS)

SYSTEM_INSTRUCTION = """
You are an elite, highly efficient AI Travel Agent.

CRITICAL OPERATIONAL RULES:
1. Ground your answers using the provided [Retrieved Context] and state details.
2. PARALLEL TOOL CALLING: If you need live action data (e.g., weather forecast, map routes, updating state), call all necessary tools in a single turn.
3. STATE UPDATES: Whenever the user provides new travel details or preferences (destination, dates, budget, food style), use `update_travel_state_tool` to update the state immediately.
4. ITINERARY OUTPUT RULE: Whenever you generate an itinerary, wrap it inside <itinerary> and </itinerary> tags.
   Example:
   <itinerary>
   # 3-Day Trip to Tokyo
   - Day 1: ...
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

# Node 1: Code-level CRAG (Process Knowledge Base / Crawler)
async def retrieve_or_crawl_node(state: TravelAgentState):
    messages = state.get("messages", [])
    user_query = messages[-1].content if messages else ""
    
    # 1. Check Pinecone for relevant context
    rag_context = await query_rag(user_query)
    
    # 2. If Pinecone returns relevant context, use it
    if rag_context and "No relevant internal guides" not in rag_context:
        print("[CRAG] Pinecone Hit!")
        return {"retrieved_context": rag_context}
    
    # 3. If Pinecone returns no relevant context, trigger the web crawler
    print("[CRAG] Pinecone Miss -> Triggering Crawler...")
    crawled_docs = await search_and_crawl(user_query, max_results=2)
    crawl_context = "\n\n".join([f"Source: {doc.url}\nContent: {doc.text}" for doc in crawled_docs])
    
    return {"retrieved_context": crawl_context}


# Node 2: Agent Reasoning Node (Process Tools Calls and Generate Final Response)
async def call_agent_node(state: TravelAgentState, config: RunnableConfig):
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
        start_on="human",
        include_system=False,
    )

    prompt_messages = [full_system_prompt] + trimmed_messages
    
    # Invoke the LLM with tools
    response = await llm_with_tools.ainvoke(prompt_messages, config)

    state_update = {
        "messages": [response],
        "draft_itinerary": state.get("draft_itinerary")
    }

    # Extract Itinerary XML Tag
    content_text = response.content if isinstance(response.content, str) else ""
    if content_text:
        match = re.search(r"<itinerary>(.*?)</itinerary>", content_text, re.DOTALL)
        if match:
            state_update["draft_itinerary"] = match.group(1).strip()

    return state_update


# =====================================================================
# 3. Conditional Routing Function
# =====================================================================
def should_continue(state: TravelAgentState) -> Literal["tools", "__end__"]:
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    # if the last message contains tool calls, route to "tools" node; otherwise, end the conversation
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
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
# 5. Checkpointer (MemorySaver / Postgres) & Compile Graph
# =====================================================================

# Checkpointer: In-memory for development; 
# can switch to AsyncPostgresSaver or RedisSaver for production
checkpointer = MemorySaver()
travel_agent_app = builder.compile(checkpointer=checkpointer)