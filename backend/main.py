import os
import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse

from agent import travel_agent_app, TravelAgentState

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


@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    """
    Streaming Chat Endpoint via Server-Sent Events (SSE).
    Streams LLM tokens live and sends final state metadata (like draft_itinerary).
    """
    try:
        print(f"[FastAPI] Received query: '{request.query}' for thread_id: '{request.thread_id}'")
        
        # 1. Create initial_state & Config
        initial_state: TravelAgentState = {"messages": [("user", request.query)]}
        config = {
            "configurable": {"thread_id": request.thread_id},
            "recursion_limit": 10
        }

        # 2. Define a Server-Sent Events (SSE) generator
        async def event_generator():
            try:
                print("[FastAPI] Starting SSE event stream...")

                async for event in travel_agent_app.astream_events(initial_state, version="v2", config=config):
                    kind = event["event"]
                    node_name = event.get("metadata", {}).get("langgraph_node", "")
                    
                    # Event: on_chain_start for the "retrieve_or_crawl" node
                    if kind == "on_chain_start" and node_name == "retrieve_or_crawl":
                        yield f"data: {json.dumps({'type': 'status', 'message': 'Checking knowledge base & searching web...'})}\n\n"

                    # Event: on_tool_start for any tool invocation
                    elif kind == "on_tool_start":
                        tool_name = event.get("name", "tool")
                        yield f"data: {json.dumps({'type': 'status', 'message': f'Calling tool: {tool_name}...'})}\n\n"
                            
                    # Event: on_chat_model_stream for streaming LLM output
                    elif kind == "on_chat_model_stream":
                        # Only stream content from the "agent" node to the frontend
                        if node_name == "agent":
                            chunk = event["data"]["chunk"]
                            content_text = ""
                            
                            # Support both string and list formats for chunk content
                            if hasattr(chunk, "content") and chunk.content:
                                if isinstance(chunk.content, str):
                                    content_text = chunk.content
                                elif isinstance(chunk.content, list):
                                    for block in chunk.content:
                                        if isinstance(block, str):
                                            content_text += block
                                        elif isinstance(block, dict) and block.get("type") == "text":
                                            content_text += block.get("text", "")

                            if content_text:
                                yield f"data: {json.dumps({'type': 'content', 'delta': content_text})}\n\n"

                # 3. After streaming is complete, fetch the final state to extract draft_itinerary
                final_state_snapshot = await travel_agent_app.aget_state(config)
                draft = final_state_snapshot.values.get("draft_itinerary", None)
                
                print(f"[FastAPI METADATA] Sending draft_itinerary length: {len(draft) if draft else 0}")
                yield f"data: {json.dumps({'type': 'metadata', 'draft_itinerary': draft})}\n\n"
                yield "data: [DONE]\n\n"

            except Exception as e:
                print(f"❌ [FastAPI Stream Error]: {str(e)}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        # 4. Return StreamingResponse
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    except Exception as e:
        print(f"❌ [FastAPI Initialization Error]: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    
    uvicorn.run("main:app", host=host, port=port)

# uv run python -m backend.main
# uv run python .\backend\main.py
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
