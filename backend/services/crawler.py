import os
import re
from datetime import datetime
from typing import List
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from llama_index.core import Document

# Load environment variables from .env file
load_dotenv()

# Regular expressions for cleaning up text
RE_PURE_LINKS = re.compile(r'^\s*(?:[*+-]\s*)?(?:\[.*?\]\(https?://.*?\)\s*)+$', re.MULTILINE)
RE_EMPTY_LINKS = re.compile(r'\[\s*\]\(https?://.*?\)')
RE_MULTI_NEWLINES = re.compile(r'\n\s*\n')

# Tavily API Client Singleton
_tavily_client = None

def get_tavily_client() -> AsyncTavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("TAVILY_API_KEY is not set in the .env file.")
        _tavily_client = AsyncTavilyClient(api_key=api_key)
    return _tavily_client

def clean_markdown_noise(text: str) -> str:
    """Remove noise from the raw text, such as pure links and excessive blank lines."""
    if not text:
        return ""
    text = RE_PURE_LINKS.sub('', text)
    text = RE_EMPTY_LINKS.sub('', text)
    text = RE_MULTI_NEWLINES.sub('\n\n', text)
    return text.strip()

async def search_and_crawl(query: str, max_results: int = 3) -> List[Document]:
    """
    Fetch search results and full webpage contents directly via Tavily API.
    Replaces Crawl4AI/Chromium to save memory on Render Free Tier.
    """
    client = get_tavily_client()
    
    print(f"[Search Engine] Searching & Extracting content for: '{query}'...")
    
    # Advanced search with raw content fetching
    search_response = await client.search(
        query=query, 
        max_results=max_results,
        search_depth="advanced",
        include_raw_content=False  # Set to True if you want raw HTML content, but may increase payload size
    )
    
    results = search_response.get('results', [])
    if not results:
        print("[Search Engine] No results found for the query.")
        return []

    documents = []
    for result in results:
        raw_text = result.get('raw_content') or result.get('content') or ""
        clean_text = clean_markdown_noise(raw_text)
        
        if len(clean_text) > 100:
            doc = Document(
                text=clean_text,
                metadata={
                    "source_url": result.get('url', ''),
                    "title": result.get('title', ''),
                    "query": query,
                    "valid_year": datetime.now().year
                }
            )
            documents.append(doc)

    print(f"[Search Engine] Successfully retrieved {len(documents)} documents via Tavily.")
    return documents


# Local Test
if __name__ == "__main__":
    import asyncio
    test_query = "Tokyo's recommended street food and local cuisine"
    docs = asyncio.run(search_and_crawl(test_query, max_results=2))
    print(f"\n{len(docs)} documents retrieved for query: {test_query}")
    if docs:
        print(f"\n[Sample Content Preview]:\n{docs[0].text[:500]}...")

# uv run python backend/services/crawler.py
# uv run python -m backend.services.crawler