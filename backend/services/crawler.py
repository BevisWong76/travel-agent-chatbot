import os
import re
from typing import List
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from llama_index.core import Document

load_dotenv()

# Clean Markdown Noise
def clean_markdown_noise(text: str) -> str:
    """Remove noise from the raw text, such as pure links and excessive blank lines."""
    if not text:
        return ""
    
    # Remove pure link lists
    text = re.sub(r'^\s*(?:[*+-]\s*)?(?:\[.*?\]\(https?://.*?\)\s*)+$', '', text, flags=re.MULTILINE)
    
    # Remove inline empty links
    text = re.sub(r'\[\s*\]\(https?://.*?\)', '', text)
    
    # Remove excessive blank lines
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


async def search_and_crawl(query: str, max_results: int = 3) -> List[Document]:
    """
    Fetch search results and full webpage contents directly via Tavily API.
    Replaces Crawl4AI/Chromium to save memory on Render Free Tier.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY is not set in the .env file.")

    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
    
    print(f"[Search Engine] Searching & Extracting content for: '{query}'...")
    
    # Use Tavily's search API to get results and their raw content
    search_response = await tavily_client.search(
        query=query, 
        max_results=max_results,
        search_depth="advanced",       # Advanced search depth for better results
        include_raw_content=True       # Fetch full webpage content directly
    )
    
    results = search_response.get('results', [])
    if not results:
        print("[Search Engine] No results found for the query.")
        return []

    documents = []
    for result in results:
        # Prefer 'raw_content' if available, else fallback to 'content'
        raw_text = result.get('raw_content') or result.get('content') or ""
        clean_text = clean_markdown_noise(raw_text)
        
        if len(clean_text) > 100:
            doc = Document(
                text=clean_text,
                metadata={
                    "source_url": result.get('url', ''),
                    "title": result.get('title', ''),
                    "query": query
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