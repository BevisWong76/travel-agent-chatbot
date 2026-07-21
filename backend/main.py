from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.agent.graph import travel_agent_graph
from backend.agent.state import TravelAgentState

app = FastAPI(
    title="AI Travel Planner API",
    description="FastAPI Backend powered by LangGraph Search-driven RAG Agent",
    version="2.0.0"
)

# Request & Response Models
class QueryRequest(BaseModel):
    query: str
    thread_id: str = "default_session"  # 供對話 Session/Memory 使用

class QueryResponse(BaseModel):
    query: str
    answer: str

@app.get("/")
def health_check():
    return {"status": "healthy", "service": "AI Travel Planner Backend"}

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Core Chat Endpoint:
    Processes user queries, invokes the LangGraph agent workflow, and returns the final response.
    """
    try:
        print(f"📩 [FastAPI] Received query: '{request.query}'")
        
        # 1. Create initial_state
        initial_state: TravelAgentState = {
            "messages": [("user", request.query)]
        }
        
        # 2. Create LangGraph execution Config (supports Thread ID)
        config = {"configurable": {"thread_id": request.thread_id}}
        
        # 3. Asynchronously invoke Agent Graph computation
        final_state = await travel_agent_graph.ainvoke(initial_state, config=config)
        
        # 4. Extract LLM final output (last Message)
        last_message = final_state["messages"][-1]
        
        # Handle cases where the last message content might be a list of chunks (e.g., from streaming responses)
        answer_text = last_message.content
        if isinstance(answer_text, list):
            answer_text = "".join([chunk.get("text", "") for chunk in answer_text if isinstance(chunk, dict)])

        return QueryResponse(
            query=request.query,
            answer=answer_text
        )

    except Exception as e:
        print(f"❌ [FastAPI Error]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


# uv run python -m backend.main
# http://127.0.0.1:8000/docs
# {
#   "query": "What are the recommended attractions in Shibuya?",
#   "thread_id": "session_001"
# }