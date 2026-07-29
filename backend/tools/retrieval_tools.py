import asyncio
from langchain_core.tools import tool
from services.crawler import search_and_crawl
from services.rag_engine import ingest_documents, query_rag

@tool
async def search_and_crawl_tool(query: str) -> str:
    """Useful when you need to search live web data for recent events, newly opened spots, or fresh travel info."""
    docs = await search_and_crawl(query, max_results=3)
    if docs:
        await asyncio.to_thread(ingest_documents, docs)
        return f"Successfully crawled and ingested {len(docs)} documents into Pinecone vector store."
    return "No relevant web pages found."

@tool
async def rag_retrieval_tool(query: str) -> str:
    """Useful for searching travel knowledge, recommendations, and context stored in Pinecone."""
    return await query_rag(query)

# Notes:
# We have implimented a CRAG in the graph
# Hence this tools are not being used in the graph.