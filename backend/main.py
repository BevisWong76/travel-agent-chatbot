import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.agent import travel_agent_app, TravelAgentState

app = FastAPI(
    title="AI Travel Planner API",
    description="FastAPI Backend powered by LangGraph Search-driven RAG Agent",
    version="2.0.0"
)

# Request & Response Models
class QueryRequest(BaseModel):
    query: str
    thread_id: str = "default_session" 

class QueryResponse(BaseModel):
    query: str
    answer: str
    draft_itinerary: Optional[str] = None


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
        print(f"[FastAPI] Received query: '{request.query}' for thread_id: '{request.thread_id}'")
        
        # 1. Create initial_state
        initial_state: TravelAgentState = {
            "messages": [("user", request.query)]
        }
        
        # 2. Create LangGraph execution Config
        # Added recursion_limit to prevent infinite tool-calling loops
        config = {
            "configurable": {
                "thread_id": request.thread_id  # Allows for session-based state management across multiple queries
            },
            "recursion_limit": 10  # allows up to 10 recursive tool calls to prevent infinite loops
        }
        
        # 3. Asynchronously invoke Agent Graph computation
        final_state = await travel_agent_app.ainvoke(initial_state, config=config)
        
        # 4. Extract LLM final output (last Message)
        last_message = final_state["messages"][-1]
        
        # Handle cases where content is structured list of text chunks
        answer_text = last_message.content
        if isinstance(answer_text, list):
            answer_text = "".join([chunk.get("text", "") for chunk in answer_text if isinstance(chunk, dict)])

        return QueryResponse(
            query=request.query,
            answer=answer_text,
            draft_itinerary=final_state.get("draft_itinerary")
        )

    except Exception as e:
        print(f"❌ [FastAPI Error]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


# uv run python -m backend.main
# http://127.0.0.1:8000/docs

# Test Cases:

# {
#   "query": "I want a 3 days itinerary to Tokyo. I like to eat sushi. Please check the weather and transit for me.",
#   "thread_id": "test_session_tools_check"
# }

# {
#   "query": "Please search and read the latest opening hours or official event page for Tsukiji Outer Market for this month.",
#   "thread_id": "test_session_tools_check"
# }
