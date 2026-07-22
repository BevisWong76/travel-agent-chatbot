import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.tools import ALL_TOOLS
from backend.agent.state import TravelAgentState

load_dotenv()

# --- 1. Initialize LLM ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    api_key=os.getenv("GEMINI_API_KEY")
)
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# Define the system prompt for the AI model
SYSTEM_PROMPT = """
You are a professional and enthusiastic global travel planning expert.
Your task is to assist users in planning itineraries, recommending attractions, local cuisine, and transportation.

Rules for responding:
1. Use a friendly, professional, and organized tone.
2. When recommending itineraries, present daily plans using Markdown tables or bullet lists.
3. Proactively remind users of travel considerations (e.g., seasonal weather, visa requirements, transportation tickets, etc.).
"""


# --- 2. Define Graph Nodes ---
async def call_model_node(state: TravelAgentState):
    """
    LLM Core Node:
    1. Receives the current state (including conversation history).
    2. Calls the LLM with the current messages and any relevant context.
    3. Returns the LLM's response, which will be appended to the history automatically by LangGraph .
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    
    return {"messages": [response]}


def should_continue(state: TravelAgentState) -> str:
    """
    Determines whether the agent should continue the conversation or terminate.
    1. If the last message contains a tool call, the agent should continue to process the tool call.
    2. If no tool calls are present, the agent can terminate the conversation.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return END

# --- 4. Build & Compile StateGraph ---

workflow = StateGraph(TravelAgentState)

# 1. 新增 Nodes
workflow.add_node("agent", call_model_node)
workflow.add_node("tools", ToolNode(ALL_TOOLS))

# 2. 設定進入點 (Entry Point)
workflow.set_entry_point("agent")

# 3. 新增條件邊 (Conditional Edge)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# 4. 工具執行完成後，自動跳回 'agent' 讓 LLM 分析結果
workflow.add_edge("tools", "agent")

# 5. 編譯 Graph (可傳入 checkpointer 支援持久化記憶)
travel_agent_graph = workflow.compile()


# --- 5. Local Test Execution ---
if __name__ == "__main__":
    import asyncio

    async def run_test():
        print("\n🚀 [Testing LangGraph Agent Workflow]...")
        
        # 模擬用戶發問
        test_query = "Help me plan a trip to Tokyo. What's the weather on 2026-09-15 and any famous street food?"
        
        initial_state: TravelAgentState = {
            "messages": [("user", test_query)],
            "destination": "Tokyo",
            "start_date": "2026-09-15",
            "end_date": "2026-09-22",
            "require_human_feedback": False
        }
        
        print(f"\nUser Query: {test_query}\n" + "-"*50)
        
        # 執行 Agent Workflow
        async for event in travel_agent_graph.astream(initial_state):
            for node_name, output in event.items():
                print(f"\n📍 [Node Executed]: {node_name}")
                if "messages" in output:
                    last_msg = output["messages"][-1]
                    # 打印 Tool Call 或 LLM 最終回覆
                    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                        print(f"🛠️ Tool Calls Requested: {last_msg.tool_calls}")
                    else:
                        print(f"💬 Agent Response Snippet: {last_msg.content[:200]}...")

    asyncio.run(run_test())

# uv run python -m backend.agent.graph 