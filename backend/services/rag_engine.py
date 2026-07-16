import os
from typing import List
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# API Keys
gemini_api_key = os.getenv("GEMINI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set.")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set.")

# --- 1. LLM & Embedding Model Settings ---
Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=gemini_api_key)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# --- --- 2. Pinecone Index ---
pc = Pinecone(api_key=pinecone_api_key)
INDEX_NAME = "travel-planner"

# If the index does not exist, create it with the correct dimension and metric
existing_indexes = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"[Pinecone] Creaing new index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Dimension of BGE-Small
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index  = pc.Index(INDEX_NAME)                                      # Pinecone Index Object


# --- 3. LlamaIndex Vector Store & Storage Context ---
vector_store    = PineconeVectorStore(pinecone_index=pinecone_index)        # LlamaIndex Vector Store Wrapper for Pinecone
storage_context = StorageContext.from_defaults(vector_store=vector_store)   # Storage Container for LlamaIndex


# --- 4. RAG Engine Functions ---

# Ingest documents into the RAG system
# Not async because VectorStoreIndex.from_documents is synchronous
def ingest_documents(documents: List[Document]):
    """ Vectorize the documents from Crawl4AI and ingest them into Pinecone for RAG retrieval. """
    if not documents:
        print("[RAG Engine] No documents to ingest.")
        return
    
    for doc in documents:
        print(f"Scraped Doc URL: {doc.metadata.get('source_url')} | Length: {len(doc.text)} chars")
    
    print(f"[RAG Engine] Ingesting {len(documents)} documents into Pinecone...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("[RAG Engine] Successfully ingested documents into Pinecone!")

# Query the RAG system with a user question
Settings.text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
async def query_rag(user_query: str) -> str:
    """ Query the RAG system with a user question and return the generated answer. """
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    query_engine = index.as_query_engine(similarity_top_k=10)
    
    response = await query_engine.aquery(user_query)
    return str(response)


# Local Test
if __name__ == "__main__":
    import asyncio
    from backend.services.crawler import search_and_crawl

    print(f"\n [Test Step 1] Crawling web pages...")
    test_query = "What are the recommended street foods in Tokyo?"
    docs = asyncio.run(search_and_crawl(test_query, max_results=3))
    
    print(f"\n [Test Step 2] Ingesting into Pinecone...")
    ingest_documents(docs)
    
    print(f"\n [Test Step 3] RAG Engine Query...")
    answer = asyncio.run(query_rag("Tell me 3 specific street foods in Tokyo based on the retrieved context."))
    print(f"\n [Test Step 3 Answer]\n{answer}")
    
# uv run python -m backend.services.rag_engine