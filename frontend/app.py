import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Page Configuration
st.set_page_config(page_title="AI Travel Planner", page_icon="✈️", layout="centered")
st.title("✈️ AI Travel Planner")
st.caption("Powered by Gemini API & Streamlit")

# Custom CSS
st.markdown("""
<style> 
    .stChatInput textarea {             
        font-size: 18px !important;
        line-height: 1.5 !important;
    }
    .stChatInput > div {
        min-height: 60px !important;
        padding-top: 5px !important;
        padding-bottom: 5px !important;
    }
    .stChatMessage p {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Gemini Client with caching to avoid re-initialization on every rerun
@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ GEMINI_API_KEY is not set in the environment variables.")
        st.stop()
    return genai.Client(api_key=api_key)

client = get_gemini_client()

# Define the system prompt for the AI model
SYSTEM_PROMPT = """
You are a professional and enthusiastic global travel planning expert.
Your task is to assist users in planning itineraries, recommending attractions, local cuisine, and transportation.

Rules for responding:
1. Use a friendly, professional, and organized tone.
2. When recommending itineraries, present daily plans using Markdown tables or bullet lists.
3. Proactively remind users of travel considerations (e.g., seasonal weather, visa requirements, transportation tickets, etc.).
"""

# Fallback model chain: If the current model hits quota limits, automatically switch to the next one
MODEL_CHAIN = [
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

# Model Index
if "model_index" not in st.session_state:
    st.session_state.model_index = 0

current_model = MODEL_CHAIN[st.session_state.model_index]

# Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Session
if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(
        model=current_model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.7,
        )
    )

# Sidebar for System Status and Controls
with st.sidebar:
    st.subheader("⚙️ System Status")
    st.info(f"**Active Model:** `{MODEL_CHAIN[st.session_state.model_index]}` (Tier {st.session_state.model_index + 1})")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.model_index = 0  # 重置回第一個模型
        st.session_state.chat = client.chats.create(
            model=MODEL_CHAIN[0],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
            )
        )
        st.rerun()

# Render chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_input := st.chat_input("Ask me anything about your travel plans..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Handle assistant response with model switching logic
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        success = False

        # Loop through the model chain until a successful response is received or all models are exhausted
        while st.session_state.model_index < len(MODEL_CHAIN) and not success:
            active_model = MODEL_CHAIN[st.session_state.model_index]
            
            try:    
                # Try to get a response from the current model
                response_stream = st.session_state.chat.send_message_stream(user_input)
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
                # If successful, append the assistant's response to the session state and mark success
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                success = True

            except Exception as e:
                error_str = str(e).lower()
                # Check if it's a 429 Rate Limit / Quota Exceeded error
                if "429" in error_str or "quota" in error_str or "resource_exhausted" in error_str:
                    st.warning(f"⚠️ Model `{active_model}` quota is exhausted, trying the next available model...")
                    st.session_state.model_index += 1
                    
                    # If there are more models in the chain, switch to the next one
                    if st.session_state.model_index < len(MODEL_CHAIN):
                        next_model = MODEL_CHAIN[st.session_state.model_index]
                        st.session_state.chat = client.chats.create(
                            model=next_model,
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                temperature=0.7,
                            )
                        )
                    else:
                        st.error("❌ All models in the chain have exhausted their quota. Please try again later.")
                else:
                    st.error(f"Error calling API: {str(e)}")
                    break
        
        # If a successful response was received, rerun the app to update the chat history
        if success:
            st.rerun()