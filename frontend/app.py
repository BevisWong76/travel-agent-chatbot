import os
import re
import json
import uuid
import requests
import streamlit as st
from streamlit_local_storage import LocalStorage
from dotenv import load_dotenv

# =====================================================================
# 1. Environment Variables & Backend/Frontend URL Setup
# =====================================================================
load_dotenv()

if "BACKEND_URL" in st.secrets:
    BACKEND_URL = st.secrets["BACKEND_URL"]
else:
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_URL = BACKEND_URL.rstrip("/")

if "FRONTEND_URL" in st.secrets:
    FRONTEND_URL = st.secrets["FRONTEND_URL"]
else:
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")

# =====================================================================
# 2. Streamlit Page Configuration and Helper Functions
# =====================================================================
st.set_page_config(page_title="AI Travel Planner", layout="wide")
st.title("✈️ AI Travel Planner Agent")


# Helper function to show a dialog for copying the itinerary link
@st.dialog("Share Itinerary Link")
def show_copy_dialog(url):
    st.write("Click the button on the right side of the box below to copy:")
    st.code(url, language=None)  # Native Streamlit copy button built-in!


# =====================================================================
# 3. Process thread_id (URL & LocalStorage Priority Handling)
# =====================================================================
if "local_s_instance" not in st.session_state:
    st.session_state.local_s_instance = LocalStorage(key="travel_local_storage_singleton")

local_s = st.session_state.local_s_instance
query_params = st.query_params

# Try to retrieve saved thread_id from LocalStorage
saved_thread_id = local_s.getItem("travel_thread_id")

# Priority 1: If the URL has a ?thread_id=xxx parameter (including F5 refresh or clicking a shared link)
if "thread_id" in query_params:
    target_thread_id = query_params["thread_id"]
    if st.session_state.get("thread_id") != target_thread_id:
        st.session_state.thread_id = target_thread_id
        st.session_state.pop("messages", None)
    local_s.setItem("travel_thread_id", target_thread_id)

# Priority 2: If no URL parameter, restore the last thread_id from LocalStorage
elif saved_thread_id:
    if st.session_state.get("thread_id") != saved_thread_id:
        st.session_state.thread_id = saved_thread_id
        st.session_state.pop("messages", None)
    st.query_params["thread_id"] = saved_thread_id

# Priority 3: If neither URL parameter nor LocalStorage has a value (or still initializing)
else:
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    st.query_params["thread_id"] = st.session_state.thread_id
    local_s.setItem("travel_thread_id", st.session_state.thread_id)

# Initialize session state variables if they don't exist
if "draft_itinerary" not in st.session_state:
    st.session_state.draft_itinerary = None
if "is_drafting" not in st.session_state:
    st.session_state.is_drafting = False

# =====================================================================
# 4. Top Action Bar: Link Sharing & New Plan Controls
# =====================================================================
top_col1, top_col2, top_col3 = st.columns([3, 1, 1])

with top_col1:
    # Display the current itinerary link for sharing
    current_url = f"{FRONTEND_URL}/?thread_id={st.session_state.thread_id}"
    st.text_input(
        "🔗 Your Itinerary Link:",
        value=current_url,
        disabled=True,
        label_visibility="collapsed",
    )

with top_col2:
    if st.button("📋 Copy Link", use_container_width=True):
        show_copy_dialog(current_url)

with top_col3:
    if st.button("➕ New Plan", use_container_width=True):
        new_id = str(uuid.uuid4())

        # 1. Update URL query parameter to reflect the new thread_id
        st.query_params["thread_id"] = new_id

        # 2. Update session state
        st.session_state.thread_id = new_id
        st.session_state.messages = []
        st.session_state.draft_itinerary = None

        st.rerun()

st.divider()

# =====================================================================
# 5. Fetch historical messages and drafted itinerary
# =====================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Fetch history from backend
    try:
        with st.spinner("Waking up backend server, please wait a moment..."):
            res = requests.get(f"{BACKEND_URL}/history/{st.session_state.thread_id}", timeout=60)
        if res.status_code == 200:
            data = res.json()
            raw_messages = data.get("messages", [])

            # Clean <itinerary> tags from assistant messages and provide default text if empty
            cleaned_messages = []
            for msg in raw_messages:
                content_str = str(msg.get("content", ""))

                if msg["role"] == "assistant" and "<itinerary>" in content_str:
                    clean_content = re.sub(
                        r"<itinerary>.*?</itinerary>", "", content_str, flags=re.DOTALL
                    ).strip()
                    if not clean_content:
                        clean_content = "I've updated your itinerary based on your details! Check out the panel on the right. ➡️"
                    msg["content"] = clean_content
                else:
                    msg["content"] = content_str

                cleaned_messages.append(msg)

            st.session_state.messages = cleaned_messages
            st.session_state.draft_itinerary = data.get("draft_itinerary", None)

    except Exception as e:
        st.error(f"Failed to fetch history: {str(e)}")

# =====================================================================
# 6. Main Layout: Left Chat Window and Right Draft Itinerary Window
# =====================================================================

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

                payload = {"query": prompt, "thread_id": st.session_state.thread_id}

                try:
                    res = requests.post(
                        f"{BACKEND_URL}/chat",
                        json=payload,
                        stream=True,
                        timeout=(20, 300),
                    )

                    if res.status_code == 200:
                        # 1. Stream the response line by line
                        for line in res.iter_lines():
                            if line:
                                decoded_line = line.decode("utf-8")
                                if decoded_line.startswith("data: "):
                                    data_str = decoded_line[6:].strip()
                                    if data_str == "[DONE]":
                                        break

                                    try:
                                        event_data = json.loads(data_str)
                                        event_type = event_data.get("type")

                                        if event_type == "status":
                                            status_msg = event_data.get(
                                                "message", "Model is processing..."
                                            )
                                            status_placeholder.info(f"⚙️ {status_msg}")

                                        # Process Streaming Token
                                        elif event_data.get("type") == "content":
                                            delta = event_data.get("delta", "")
                                            accumulated_text += delta

                                            # Check if entering <itinerary> tag
                                            if "<itinerary>" in accumulated_text:
                                                if not in_itinerary:
                                                    in_itinerary = True
                                                    status_placeholder.info(
                                                        "📝 Drafting itinerary..."
                                                    )

                                                # Inside tag: only display conversational intro text on the left chat window
                                                intro_text = accumulated_text.split(
                                                    "<itinerary>"
                                                )[0].strip()
                                                if intro_text:
                                                    message_placeholder.markdown(
                                                        intro_text
                                                    )
                                                else:
                                                    message_placeholder.markdown(
                                                        "*(Generating your itinerary...)*"
                                                    )
                                            else:
                                                # Prior to tag: display live typing stream with cursor
                                                message_placeholder.markdown(
                                                    accumulated_text + "▌"
                                                )

                                        # Final processing of Metadata
                                        elif event_data.get("type") == "metadata":
                                            if event_data.get("draft_itinerary"):
                                                draft_itinerary_data = event_data.get(
                                                    "draft_itinerary"
                                                )

                                    except json.JSONDecodeError:
                                        continue

                        # 2. Clean up and extract dialogue after streaming completes
                        status_placeholder.empty()

                        # Extract dialogue before and after <itinerary> tags
                        dialogue_before = accumulated_text.split("<itinerary>")[0].strip()
                        dialogue_after = ""
                        if "</itinerary>" in accumulated_text:
                            dialogue_after = accumulated_text.split("</itinerary>")[
                                -1
                            ].strip()

                        final_chat_text = (
                            f"{dialogue_before}\n\n{dialogue_after}".strip()
                        )

                        # Provide default prompt if no conversational text was generated
                        if not final_chat_text:
                            final_chat_text = "I've updated your itinerary based on your details! Check out the panel on the right. ➡️"

                        # Finalize rendering and update message history
                        message_placeholder.markdown(final_chat_text)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": final_chat_text}
                        )

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
    col2_title, col2_btn = st.columns([3, 1])
    with col2_title:
        st.subheader("🗺️ Draft Itinerary")

    with col2_btn:
        if st.session_state.draft_itinerary:
            st.download_button(
                label="📥 Download",
                data=st.session_state.draft_itinerary,
                file_name="my_itinerary.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # Scrollable container matching left chat window height
    itinerary_container = st.container(height=550)

    with itinerary_container:
        if st.session_state.draft_itinerary:
            st.markdown(st.session_state.draft_itinerary)
        else:
            st.info("Your generated itinerary will appear here!")
