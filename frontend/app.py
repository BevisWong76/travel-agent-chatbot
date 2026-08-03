import os
import re
import json
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Robust Secrets / Env Handling
if "BACKEND_URL" in st.secrets:
    BACKEND_URL = st.secrets["BACKEND_URL"]
else:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

BACKEND_URL = BACKEND_URL.rstrip("/")

st.set_page_config(page_title="AI Travel Planner", layout="wide")
st.title("✈️ AI Travel Planner Agent")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "draft_itinerary" not in st.session_state:
    st.session_state.draft_itinerary = None
if "is_drafting" not in st.session_state:
    st.session_state.is_drafting = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Fetch Historical Messages & Draft Itinerary from Backend
if "messages" not in st.session_state or not st.session_state.messages:
    res = requests.get(f"{BACKEND_URL}/history/{st.session_state.thread_id}")
    if res.status_code == 200:
        data = res.json()
        st.session_state.messages = data.get("messages", [])
        st.session_state.draft_itinerary = data.get("draft_itinerary", None)

# Layout: 1:1 Split
col1, col2 = st.columns([1, 1])

# ==================== Left Window: Chat Interface ====================
with col1:
    st.subheader("💬 Chat with Agent")
    
    # 1. Scrollable Chat Container (Fixed Height)
    chat_container = st.container(height=550)
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 2. Fixed Input at Bottom
    if prompt := st.chat_input("Ask anything about your travel plan..."):
        # Display user message and append to session state
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Prepare assistant response streaming
        with chat_container:
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                status_placeholder = st.empty()
                
                accumulated_text = ""
                in_itinerary = False
                draft_itinerary_data = None
                
                status_placeholder.info("Model is thinking...")
                
                payload = {
                    "query": prompt,
                    "thread_id": st.session_state.thread_id
                }
                
                try:
                    res = requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True, timeout=60)
                    
                    if res.status_code == 200:
                        # 1. Stream the response line by line
                        for line in res.iter_lines():
                            if line:
                                decoded_line = line.decode('utf-8')
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:].strip()
                                    if data_str == "[DONE]":
                                        break
                                    
                                    try:
                                        event_data = json.loads(data_str)
                                        event_type = event_data.get("type")


                                        if event_type == "status":
                                            status_msg = event_data.get("message", "Model is processing...")
                                            status_placeholder.info(f"⚙️ {status_msg}")
                                        
                                        # Process Streaming Token
                                        elif event_data.get("type") == "content":
                                            delta = event_data.get("delta", "")
                                            accumulated_text += delta
                                            
                                            # Check if entering <itinerary> tag
                                            if "<itinerary>" in accumulated_text:
                                                if not in_itinerary:
                                                    in_itinerary = True
                                                    status_placeholder.info("📝 Drafting itinerary...")
                                                
                                                # Inside tag: only display conversational intro text on the left chat window
                                                intro_text = accumulated_text.split("<itinerary>")[0].strip()
                                                if intro_text:
                                                    message_placeholder.markdown(intro_text)
                                                else:
                                                    message_placeholder.markdown("*(Generating your itinerary...)*")
                                            else:
                                                # Prior to tag: display live typing stream with cursor
                                                message_placeholder.markdown(accumulated_text + "▌")
                                        
                                        # Final processing of Metadata
                                        elif event_data.get("type") == "metadata":
                                            if event_data.get("draft_itinerary"):
                                                draft_itinerary_data = event_data.get("draft_itinerary")
                                                
                                    except json.JSONDecodeError:
                                        continue
                        
                        # 2. Clean up and extract dialogue after streaming completes
                        status_placeholder.empty()
                        
                        # Extract dialogue before and after <itinerary> tags
                        dialogue_before = accumulated_text.split("<itinerary>")[0].strip()
                        dialogue_after = ""
                        if "</itinerary>" in accumulated_text:
                            dialogue_after = accumulated_text.split("</itinerary>")[-1].strip()
                            
                        final_chat_text = f"{dialogue_before}\n\n{dialogue_after}".strip()
                        
                        # Provide default prompt if no conversational text was generated
                        if not final_chat_text:
                            final_chat_text = "I've updated your itinerary based on your details! Check out the panel on the right. ➡️"
                            
                        # Finalize rendering and update message history
                        message_placeholder.markdown(final_chat_text)
                        st.session_state.messages.append({"role": "assistant", "content": final_chat_text})
                        
                        # 3. Save draft itinerary data and trigger rerun to refresh right panel
                        if draft_itinerary_data:
                            st.session_state.draft_itinerary = draft_itinerary_data

                        st.rerun()

                    else:
                        st.error(f"Error from API ({res.status_code}): {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {str(e)}")

# ==================== Right Window: Draft Itinerary ====================
with col2:
    st.subheader("🗺️ Draft Itinerary")
    
    # Scrollable container matching left chat window height
    itinerary_container = st.container(height=550)
    
    with itinerary_container:
        if st.session_state.draft_itinerary:
            st.markdown(st.session_state.draft_itinerary)
        else:
            st.info("Your generated itinerary will appear here!")