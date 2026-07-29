import os
from datetime import datetime
from typing import List
from dotenv import load_dotenv, find_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Settings, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core.postprocessor import SimilarityPostprocessor
from pinecone import Pinecone, ServerlessSpec

load_dotenv(find_dotenv())

# API Keys
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY is not set.")
if not gemini_api_key:
    print(f"DEBUG: Current Working Directory = {os.getcwd()}")
    print(f"DEBUG: Found .env file at = {find_dotenv()}")
    raise ValueError("GEMINI_API_KEY is not set.")

# --- 1. Embedding Model Settings ---
Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2",
    api_key=gemini_api_key
)
Settings.text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

# --- 2. Pinecone Index ---
pc = Pinecone(api_key=pinecone_api_key)
INDEX_NAME = "travel-planner"

existing_indexes = [i.name for i in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"[Pinecone] Creating new index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,     # for gemini-embedding-2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

pinecone_index = pc.Index(INDEX_NAME)

# --- 3. LlamaIndex Vector Store & Storage Context ---
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# --- 4. RAG Engine Functions ---
def ingest_documents(documents: List[Document]):
    """ Vectorize the documents via Gemini API and ingest them into Pinecone for RAG retrieval. """
    if not documents:
        print("[RAG Engine] No documents to ingest.")
        return
    
    current_year = datetime.now().year
    
    for doc in documents:
        url = doc.metadata.get("source_url", "")
        doc.id_ = url if url else doc.id_
        
        if "valid_year" not in doc.metadata:
            doc.metadata["valid_year"] = current_year
        
        print(f"Scraped Doc URL: {doc.metadata.get('source_url')} | Length: {len(doc.text)} chars")
    
    print(f"[RAG Engine] Ingesting {len(documents)} documents into Pinecone...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )
    print("[RAG Engine] Successfully ingested documents into Pinecone!")


async def query_rag(user_query: str, min_year: int = datetime.now().year) -> str:
    """ Search Pinecone for relevant documents and return concatenated context chunks. """
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    
    # 1. Filter nodes by valid_year and apply similarity postprocessor
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="valid_year", value=min_year)]
    )
    processor = SimilarityPostprocessor(similarity_cutoff=0.7)
    
    # 2. Create Retriever
    retriever = index.as_retriever(
        similarity_top_k=5,
        node_postprocessors=[processor],
        filters=filters
    )
    
    # 3. Retrieve relevant nodes from Pinecone
    nodes = await retriever.aretrieve(user_query)
    
    if not nodes:
        return "No relevant internal guides found in database."
    
    # 4. Return concatenated context
    context_list = []
    for i, node in enumerate(nodes, 1):
        source = node.metadata.get("source_url", "Unknown")
        context_list.append(f"[Source {i}]: {source}\nContent: {node.get_content().strip()}")
        
    return "\n\n---\n\n".join(context_list)


# Local Test
if __name__ == "__main__":
    import asyncio
    from .crawler import search_and_crawl

    print(f"\n [Test Step 1] Crawling web pages...")
    test_query = "What are the recommended street foods in Hong Kong?"
    docs = asyncio.run(search_and_crawl(test_query, max_results=2))
    
    print(f"\n [Test Step 2] Ingesting into Pinecone...")
    ingest_documents(docs)
    
    print(f"\n [Test Step 3] RAG Pure Retrieval Test...")
    context = asyncio.run(query_rag("Tell me 3 specific street foods in Hong Kong based on the retrieved context."))
    print(f"\n [Retrieved Context for Agent]\n{context}")
    
# uv run python backend/services/rag_engine.py
# uv run python -m backend.services.rag_engine