import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from llama_index.core import Document

load_dotenv()
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

# --- 1. Low-level Helpers ---

async def crawl_single_url(url: str) -> Optional[str]:
    """ Crawl a single URL and return the Clean Markdown"""
    # BrowserConfig: headless=True (no GUI), verbose=False (no logs)
    browser_config = BrowserConfig(headless=True, verbose=False, user_agent=user_agent)
    # CrawlerRunConfig: cache_mode=BYPASS (always fetch fresh), word_count_threshold=10 (ignore very short pages)
    run_config     = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, word_count_threshold=10)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        return result.markdown if result.success else None

async def crawl_urls_to_documents(urls: List[str]) -> List[Document]:
    """ Crawl multiple URLs concurrently and return a list of Document objects with Clean Markdown and metadata"""
    if not urls:
        return []
    browser_config = BrowserConfig(headless=True, verbose=False, user_agent=user_agent)
    run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    documents = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Use arun_many for concurrent crawling
        results = await crawler.arun_many(urls=urls, config=run_config)
        for result in results:
            if result.success and result.markdown:
                doc = Document(
                    text=result.markdown,
                    metadata={"source_url": result.url}
                )
                documents.append(doc)
    return documents

# --- 2. High-level Search-Driven Pipeline ---

async def search_and_crawl(query: str, max_results: int = 3) -> List[Document]:
    """ Integrate Tavily Search API with Crawl4AI to fetch URLs based on a query and crawl them into Document objects """
    # Load Tavily API key from environment variables
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY is not set in the .env file.")

    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
    
    # Perform the search
    print(f"[Search Engine] Searching: '{query}'...")
    search_response = await tavily_client.search(
        query=query, 
        max_results=max_results,
        search_depth="basic"
    )
    urls = [result['url'] for result in search_response.get('results', [])]
    
    if not urls:
        print("[Search Engine] No URLs found for the query.")
        return []

    # Crawl the retrieved URLs concurrently and convert them into Document objects
    print(f"[Search Engine] Found {len(urls)} URLs, starting concurrent crawling...")
    documents = await crawl_urls_to_documents(urls)
    
    # Add the original query to each Document's metadata for traceability
    for doc in documents:
        doc.metadata["query"] = query

    return documents

# Local Test
if __name__ == "__main__":
    test_query = "Tokyo's recommended street food and local cuisine"
    docs = asyncio.run(search_and_crawl(test_query, max_results=2))
    print(f"\n{len(docs)} documents retrieved for query: '{test_query}")