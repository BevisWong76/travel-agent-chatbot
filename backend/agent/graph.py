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

from backend.tools import ALL_TOOLS 
from backend.agent.state import TravelAgentState, build_agent_context

load_dotenv()

# =====================================================================
# 1. Initialize LLM, Tools, and System Instructions
# =====================================================================

# Initialize the Google Gemini LLM
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
llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_INSTRUCTION = """
You are an elite, highly efficient AI Travel Agent. Your core objective is to resolve user requests with the ABSOLUTE MINIMUM number of API turns and tool calls (ideally completing the entire task in 1-2 turns).

CRITICAL OPERATIONAL RULES:
1. PARALLEL TOOL CALLING: If a user request requires multiple actions (e.g., updating travel state AND checking weather, transit, or searching for spots), you MUST invoke ALL necessary tools simultaneously in a SINGLE turn. Do NOT call tools sequentially one by one across multiple turns.
2. STATE UPDATES: Whenever the user provides new travel details or preferences (destination, dates, budget, food style), bundle `update_travel_state_tool` together with any retrieval tools in your very first turn.
3. STRICT TOOL BUDGET: You have a strict limit of AT MOST 1 round of tool calls. Gather all necessary information immediately in one batch.
4. ITINERARY OUTPUT RULE: Whenever you generate or revise a day-by-day travel itinerary for the user, you MUST wrap the complete itinerary markdown inside <itinerary> and </itinerary> tags.
   Example:
   <itinerary>
   # 3-Day Trip to Tokyo
   - Day 1: ...
   </itinerary>
5. IMMEDIATE TERMINATION: As soon as the tool execution results are returned to you, you MUST synthesize the final response directly. Do NOT issue any further tool calls under any circumstances.
"""

# Notes:
# We use XML tags to allow the backend to extract the itinerary from the LLM's response.
# Another approach could be to have the LLM return a structured JSON object.
# JSON Mode have slightly higher stability, but also higher latency, so we prefer the XML tag approach for now.
# JSON Mode is better for API-to-API communication, while XML tag extraction is better for human-facing chat interfaces.

# =====================================================================
# 2. Node Function for Agent Reasoning
# =====================================================================
async def call_agent_node(state: TravelAgentState, config: RunnableConfig):
    """
        Agent Reasoning Node:
    1. Builds a structured context from the current TravelAgentState and the system instruction.
    2. Trims the conversation history to fit within the LLM's context window.
    3. Invokes the LLM with the context and conversation history.
    4. Extracts the LLM's response and updates the TravelAgentState accordingly.
    """
    # 1. Prepare the system prompt with the current state context
    context_str = build_agent_context(state)
    full_system_prompt = SystemMessage(content=f"{SYSTEM_INSTRUCTION}\n\n{context_str}")

    # 2. Context Windowing
    messages = state.get("messages", [])
    trimmed_messages = trim_messages(
        messages,
        max_tokens=5000,
        strategy="last",
        token_counter=len,
        start_on="human",
        include_system=False,
    )

    # 3. Invoke LLM
    prompt_messages = [full_system_prompt] + trimmed_messages
    response = await llm_with_tools.ainvoke(prompt_messages, config)

    # 4. Extract content text from the LLM response
    content_text = ""
    if isinstance(response.content, str):
        content_text = response.content
    elif isinstance(response.content, list):
        content_text = "".join(
            chunk.get("text", "") for chunk in response.content if isinstance(chunk, dict)
        )

    # Initialize the state update with the new message and current step
    state_update = {
        "messages": [response],
        "draft_itinerary": state.get("draft_itinerary")
    }

    if content_text:
        match = re.search(r"<itinerary>(.*?)</itinerary>", content_text, re.DOTALL)
        if match:
            state_update["draft_itinerary"] = match.group(1).strip()

    return state_update

# =====================================================================
# 3. Conditional Routing
# =====================================================================
def should_continue(state: TravelAgentState) -> Literal["tools", "__end__"]:
    """
        Conditional Routing Function:
    Checks the current state to determine if the agent should invoke tools or end the conversation.
    - If the last message contains tool calls, it routes to the "tools" node.
    - If there are no messages or no tool calls, it ends the conversation. 
    """
    messages = state.get("messages", [])
    if not messages:
        return END

    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return END


# =====================================================================
# 4. StateGraph workflow Definition
# =====================================================================

# Initialize the StateGraph with the TravelAgentState schema
builder = StateGraph(TravelAgentState)

# Add Nodes
builder.add_node("agent", call_agent_node)
builder.add_node("tools", ToolNode(ALL_TOOLS))  # ToolNode supports parallel execution of multiple tools

# Set Edges
builder.add_edge(START, "agent")    # Start the workflow with the Agent Node
builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END,
    },
)
builder.add_edge("tools", "agent")  # After tool execution, results are returned to the Agent for further reasoning

# =====================================================================
# 5. Checkpointer (MemorySaver / Postgres) & Compile Graph
# =====================================================================
# Checkpointer: In-memory for development; 
# can switch to AsyncPostgresSaver or RedisSaver for production
checkpointer = MemorySaver()


# Compile the Graph with the checkpointer for state persistence
travel_agent_app = builder.compile(checkpointer=checkpointer)


