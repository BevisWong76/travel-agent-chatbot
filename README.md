# AI Travel Planner Agent

## Introduction

An intelligent, full-stack travel assistant powered by LangGraph, FastAPI, and Streamlit. The agent leverages a stateful multi-step workflow to converse with users, look up destination data, and dynamically draft customized travel itineraries in real-time. 

Designed for reliability and seamless collaboration, it features robust session management, allowing users to share specific itinerary links via unique thread IDs, refresh without losing state, and persist all conversation history and drafts securely.

> ℹ️ **Hosting & Infrastructure Note**
> This application is deployed on **free-tier serverless infrastructure** (Free Tier Gemini API, Streamlit Community Cloud for frontend hosting, Render for backend hosting, and Neon PostgreSQL for state persistence).
> * **Rate Limits:** The free-tier Gemini API imposes strict request and token per minute (RPM/TPM) limits. Consequently, reasoning and embedding calls may occasionally hit rate-limit errors during high-concurrency usage.
> * **Cold Starts:** If left idle for a while, all components—including the frontend, backend server, and database—will spin down to sleep to conserve resources. The initial request or app load may take around **60 seconds** while the instances reboot and database SSL connections re-establish.
> * **Resilience Design:** To gracefully handle connection drops and cold-start latency, the Streamlit UI features custom polling status indicators, while the FastAPI gateway uses retry logic with exponential backoff for transient database and API errors.

---

## Demo

<p align="center">
  <img src="assets/demo.gif" alt="AI Travel Planner Demo" width="100%">
</p>

**Try the Live Interactive Web App:** [Travel Agent Chatbot on Streamlit Cloud](https://travel-agent-chatbot-hqrr6h54f7bfmgptpnmyrc.streamlit.app)

---

## Architecture

The system operates on an asynchronous, event-driven architecture designed for low-latency streaming and high reliability:

### System Architecture
<p align="center">
  <img src="assets/System_Architecture.png" alt="AI Travel Planner System Architecture" width="100%">
</p>

### Graph Diagram
<p align="center">
  <img src="assets/Graph_Diagram.png" alt="AI Travel Planner Graph Diagram" width="100%">
</p>

### Details

1. **Client Layer:** The Streamlit frontend captures user inputs and establishes a Server-Sent Events (SSE) connection with the backend. It actively parses the incoming stream to separate conversational text from structural `<itinerary>` XML tags.
2. **API Gateway:** A FastAPI server orchestrates requests, managing database lifecycles via an `@asynccontextmanager` and gracefully handling serverless cold starts through custom retry loops on the `/history/{thread_id}` endpoint.
3. **Graph Orchestrator:** LangGraph drives the core decision engine. It injects a structured `TravelAgentState` (demographics, preferences, constraints) and a trimmed context window into the LLM. 
4. **Information Retrieval:** A conditional `retrieve_or_crawl_node` queries a Pinecone vector database using `gemini-embedding-2` and strict chronological filters (`min_year`, `similarity_cutoff=0.7`). If insufficient data is found, it seamlessly delegates to an asynchronous Tavily web crawler.
5. **Tool Execution:** The LLM leverages parallel tool calling to hit external APIs (Maps, Weather) using a shared HTTPX client with connection pooling to maximize throughput and prevent socket exhaustion.
6. **State Persistence:** Every node transition and state mutation is snapshotted and persisted to a Neon PostgreSQL checkpointer, ensuring seamless session recovery.

---

## Key Features

* **Advanced Session Routing & Memory:** Persistent, shareable planning sessions using a multi-tiered thread ID resolution strategy (URL Parameters → LocalStorage → New Session), allowing users to refresh or share links without losing context.
* **Intelligent Time-Aware Weather Routing:** A bespoke weather tool automatically caches geocoding coordinates and routes API calls based on the trip date: forecasts for the next 14 days, exact historical data for past dates, and **'same period last year'** historical averages for distant future trips.
* **Hybrid RAG & Async Web Crawling:** Merges LlamaIndex vector search with dynamic web crawling. Ingested documents are strictly filtered by year and relevance, while raw crawler outputs are instantly sanitized (stripping empty lines and external links) to conserve token context.
* **Real-Time Dual-Pane UI:** Employs an asynchronous event generator to stream LLM tokens to the chat interface while simultaneously updating a dedicated itinerary dashboard at the moment the agent encapsulates draft updates in `<itinerary>` tags.
* **Resilient Serverless Design:** Built to withstand free-tier infrastructure limitations via automated FastAPI database retry loops and resilient Streamlit polling UX during backend wake-ups.

---

## Tech Stack

* **AI & Orchestration:** LangGraph, LangChain (`langchain-google-genai`), LlamaIndex, Google Gemini (`gemini-3.5-flash-lite`, `gemini-embedding-2`), Official Google GenAI SDK (`google-genai`)
* **Backend:** FastAPI, Uvicorn, Python `asyncio`, HTTPX (Connection Pooling & Async Requests), `async_lru` (In-Memory Caching)
* **Frontend:** Streamlit, `streamlit-local-storage`
* **Databases & Persistence:** Neon Serverless PostgreSQL (`psycopg3` + `psycopg-pool`, LangGraph Postgres Checkpointer), Pinecone (`pinecone`, `llama-index-vector-stores-pinecone`)
* **External APIs & Tools:** Tavily Async Search API (`tavily-python`), Open-Meteo API (via HTTPX), Google Maps Directions API (`googlemaps`)
* **Observability & Infrastructure:** LangSmith, `python-dotenv`, `nest-asyncio`

---

## Environment Variables

To run this application locally or deploy it to cloud infrastructure, create a `.env` file in the root directory and configure the following variables:

```env
# 1. External API Keys
GEMINI_API_KEY=your_gemini_api_key
TAVILY_API_KEY=your_tavily_api_key
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# 2. Infrastructure & Database
DATABASE_URL="postgresql://user:pass@ep-cool-db.neon.tech/neondb?sslmode=require"
BACKEND_URL="https://your-backend-service.onrender.com"
FRONTEND_URL="https://your-app.streamlit.app"

# 3. Observability & Tracing (LangSmith)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT="travel-agent-chatbot"
```

---

## Local Setup & Installation

Follow these steps to set up and run the application locally:

### 1. Prerequisites
* Python 3.11 or higher
* Git

### 2. Clone the Repository
```bash
git clone https://github.com/BevisWong76/travel-agent-chatbot.git
cd travel-agent-chatbot
```

### 3. Create a Virtual Environment & Install Dependencies
This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable Python package management.

1. **Install `uv`** (if not already installed):
   ```bash
   pip install uv
   ```

2. **Create a virtual environment:**
   ```bash
   uv venv
   ```

3. **Activate the virtual environment:**
   * **Windows (Command Prompt):** `.venv\Scripts\activate.bat`
   * **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
   * **macOS / Linux:** `source .venv/bin/activate`

4. **Install backend and frontend dependencies:**
   ```bash
   uv pip install -r ./backend/requirements.txt
   uv pip install -r ./frontend/requirements.txt
   ```

### 4. Configure Environment Variables
Ensure you have created a `.env` file in the root directory following the [Environment Variables](#environment-variables) template.

### 5. Run the Application
Launch the backend server and Streamlit frontend in separate terminal instances:

* **Terminal 1 (Backend):**
  ```bash
  uv run python ./backend/main.py
  ```
  *Verify the API endpoints via Swagger UI at `http://127.0.0.1:8000/docs`.*

* **Terminal 2 (Frontend):**
  ```bash
  streamlit run ./frontend/app.py
  ```
  *Access the user interface locally at `http://localhost:8501`.*

