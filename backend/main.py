import sys
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import the custom modules for Crawler and RAG Engine
from backend.crawler import search_and_crawl
from backend.rag_engine import ingest_documents, query_rag

# FastAPI Application Setup
app = FastAPI(
    title="AI Travel Planner API",
    description="FastAPI Backend for Search-driven RAG Travel Assistant",
    version="1.0.0"
)

# Define Request / Response data formats (Pydantic Models)
class QueryRequest(BaseModel):
    query: str
    auto_scrape: bool = True  
    
class QueryResponse(BaseModel):
    query: str
    answer: str
    scraped_documents_count: int

# FastAPI Endpoints for Health Check
@app.get("/")
def health_check():
    """Health Check Endpoint for Render / UptimeRobot"""
    return {"status": "healthy", "service": "AI Travel Planner Backend"}

# Helpers to run async functions in a synchronous context (for FastAPI endpoints)
def _run_crawler_in_thread(query: str, max_results: int):
    """ Run the async search_and_crawl function in a separate thread with its own event loop."""
    # 1. Create a new Event Loop
    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
    else:
        loop = asyncio.new_event_loop()
    
    # 2. Run the async function in the new event loop
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(search_and_crawl(query, max_results))
    finally:
        loop.close()

# FastAPI Endpoint for Chat Queries
@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """
    Core Chat Endpoint:
    1. If auto_scrape is enabled, perform Search-driven Scraping to fetch relevant documents and ingest them into Pinecone.
    2. Query the RAG system to generate an answer based on the user's query and any newly ingested documents.
    Returns a JSON response containing the original query, the generated answer, and the count of newly scraped documents.
    """
    scraped_count = 0
    try:
        # 1. Search-Driven Scraping
        if request.auto_scrape:
            print(f"[FastAPI] Auto-scraping enabled. Searching and crawling for query: '{request.query}'...")
            documents = await asyncio.to_thread(_run_crawler_in_thread, request.query, 2)
            scraped_count = len(documents)
            
            if documents:
                await asyncio.to_thread(ingest_documents, documents)
                
        # 2. Query the RAG system
        print(f"[FastAPI] Querying the RAG system...")
        answer = await query_rag(request.query)

        return QueryResponse(
            query=request.query,
            answer=answer,
            scraped_documents_count=scraped_count
        )

    except Exception as e:
        print(f"❌ [FastAPI] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Local Test
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)


# uv run uvicorn backend.main:app --reload --loop asyncio
# http://127.0.0.1:8000/docs
# {
#   "query": "What are the recommended attractions in Shibuya?",
#   "auto_scrape": true
# }