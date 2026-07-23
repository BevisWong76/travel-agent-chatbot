import os
import re
from typing import Literal
from dotenv import load_dotenv

from langchain_core.messages import SystemMessage, trim_messages, BaseMessage
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
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,    # Low temperature for more deterministic responses
    api_key=os.getenv("GEMINI_API_KEY")
)

# Bind the LLM with all available tools for parallel execution
llm_with_tools = llm.bind_tools(ALL_TOOLS)

SYSTEM_INSTRUCTION = """
You are a professional, highly personalized AI Travel Agent.
You have access to state updating tools and external retrieval tools.

CRITICAL INSTRUCTIONS:
1. STATE UPDATES: Whenever the user provides new travel details or preferences (e.g., destination, dates, budget, food preferences), you MUST call `update_travel_state_tool` with ONLY the updated key-value pairs.
2. ITINERARY OUTPUT RULE: Whenever you generate or revise a day-by-day travel itinerary for the user, you MUST wrap the complete itinerary markdown inside <itinerary> and </itinerary> tags.
   Example:
   <itinerary>
   # 3-Day Trip to Tokyo
   - Day 1: ...
   </itinerary>"""

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

    # 4. Extract & Return State Updates
    state_update = {
        "messages": [response],
        "current_step": "agent_reasoning",
    }

    # Extract itinerary from LLM response if present
    if isinstance(response.content, str):
        # The LLM is asked to wrap the itinerary in <itinerary>...</itinerary> tags. 
        # We can extract it using regex.
        match = re.search(r"<itinerary>(.*?)</itinerary>", response.content, re.DOTALL)
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


