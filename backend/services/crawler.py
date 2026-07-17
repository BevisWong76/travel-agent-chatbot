import asyncio
import os
import re
from typing import List, Optional
from dotenv import load_dotenv
from tavily import AsyncTavilyClient
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from llama_index.core import Document

load_dotenv()
user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

# Pruning Content Filter
pruning_filter = PruningContentFilter(
    threshold=0.4,           # Pruning threshold for filtering out low-quality content blocks
    threshold_type="dynamic",# Use dynamic thresholding based on content length and quality
    min_word_threshold=5     # Ignore isolated link blocks with fewer than 5 words
)
md_generator = DefaultMarkdownGenerator(content_filter=pruning_filter)

# Clean Markdown Noise
def clean_markdown_noise(text: str) -> str:
    """Remove noise from the raw Markdown text, such as pure links in lists and excessive blank lines."""
    if not text:
        return ""
    
    # Remove pure link lists (e.g., "- [Link](https://example.com)")
    text = re.sub(r'^\s*(?:[*+-]\s*)?(?:\[.*?\]\(https?://.*?\)\s*)+$', '', text, flags=re.MULTILINE)
    
    # Remove inline links (e.g., "[Link](https://example.com)") but keep the text
    text = re.sub(r'\[\s*\]\(https?://.*?\)', '', text)
    
    # Remove excessive blank lines (more than 2 consecutive newlines)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


# --- 1. Low-level Helpers ---

async def crawl_single_url(url: str) -> Optional[str]:
    """ Crawl a single URL and return the Clean Markdown """
    browser_config = BrowserConfig(
        headless=True,              
        verbose=False, 
        user_agent=user_agent
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, 
        word_count_threshold=10,
        markdown_generator=md_generator
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        return result.markdown.fit_markdown if result.success else None

async def crawl_urls_to_documents(urls: List[str]) -> List[Document]:
    """ Crawl multiple URLs concurrently and return a list of Document objects with Clean Markdown and metadata """
    if not urls:
        return []
    
    urls = list(set(urls))  # Deduplicate URLs
    
    browser_config = BrowserConfig(
        headless=True, 
        verbose=False, 
        user_agent=user_agent,
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
        excluded_tags=["nav", "footer", "header", "aside"],
    )

    documents = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(urls=urls, config=run_config)
        for result in results:
            if result.success and result.markdown:
                # Clean Markdown Text
                # Use fit_markdown if available, otherwise fallback to raw_markdown
                clean_text = result.markdown.fit_markdown or result.markdown.raw_markdown
                
                clean_text = clean_markdown_noise(clean_text)
                
                # Create Document object with metadata
                if len(clean_text) > 100:
                    doc = Document(
                        text=clean_text,
                        metadata={"source_url": result.url}
                    )
                documents.append(doc)
    return documents

# --- 2. High-level Search-Driven Pipeline ---

async def search_and_crawl(query: str, max_results: int = 3) -> List[Document]:
    """ Integrate Tavily Search API with Crawl4AI to fetch URLs based on a query and crawl them into Document objects """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY is not set in the .env file.")

    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
    
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

    print(f"[Search Engine] Found {len(urls)} URLs, starting concurrent crawling...")
    documents = await crawl_urls_to_documents(urls)
    
    for doc in documents:
        doc.metadata["query"] = query

    return documents

# Local Test
if __name__ == "__main__":
    test_query = "Tokyo's recommended street food and local cuisine"
    docs = asyncio.run(search_and_crawl(test_query, max_results=2))
    print(f"\n{len(docs)} documents retrieved for query: {test_query}")
    if docs:
        print(f"\n[Sample Cleaned Output Preview]:\n{docs[0].text[:500]}...")

# uv run python backend/services/crawler.py
# uv run python -m backend.services.crawler