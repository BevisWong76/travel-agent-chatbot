import os
import chromadb
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load environment variables from .env file
load_dotenv()

# Check if GEMINI_API_KEY is set in the environment variables
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

# 1. LLM uses Gemini
Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=api_key)

# 2. Embeddings run locally using HuggingFace BGE Small (Fast & High Accuracy)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 3. Initialize ChromaDB
PERSIST_DIR = "./backend/chroma_db"
DATA_DIR = "./backend/data"

chroma_client = chromadb.PersistentClient(path=PERSIST_DIR)
chroma_collection = chroma_client.get_or_create_collection("travel_knowledge")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def build_or_load_index():
    if chroma_collection.count() > 0:
        print("⚡ [RAG Engine] Direct loading existing Index from ChromaDB...")
        index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
        )
    else:
        print("🔄 [RAG Engine] Building new Index from backend/data...")
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        print("✅ [RAG Engine] Index built and saved successfully!")
        
    return index

index = build_or_load_index()
query_engine = index.as_query_engine(similarity_top_k=2)

def query_rag(user_query: str) -> str:
    response = query_engine.query(user_query)
    return str(response)

if __name__ == "__main__":
    test_query = "What is special about the Night Paws cafe in Shibuya?"
    print(f"\n❓ Query: {test_query}")
    answer = query_rag(test_query)
    print(f"\n💡 Answer:\n{answer}")