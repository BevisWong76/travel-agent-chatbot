from typing import Annotated, TypedDict, List, Optional, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class TravelAgentState(TypedDict):
    # 1. Conversation History (Memory)
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 2. User Profile / Demographics
    user_age: Optional[int]
    user_gender: Optional[str]
    user_language: Optional[str]
    user_nationality: Optional[str]
    
    # 3. Travel Details
    origin: Optional[str]
    destination: Optional[str]
    
    start_date: Optional[str]
    end_date: Optional[str]
    
    travelers_count: Optional[int]          
    companion_type: Optional[str]   
    
    # 4. Preferences
    budget: Optional[str]
    interests: Optional[List[str]]
    accommodation_preferences: Optional[List[str]]
    transportation_preferences: Optional[List[str]]
    food_preferences: Optional[List[str]]
    activities_preferences: Optional[List[str]]
    
    # 5. Draft Itinerary
    draft_itinerary: Optional[str]
    
    # 6. Human in the Loop (HITL) & Workflow Control
    # current_step: Optional[str]             
    # require_human_feedback: Optional[bool]
    # human_feedback: Optional[str]
    
    # Notes: 
    # Human in the Loop (HITL) is not currently implemented, but the state structure allows for future integration of human feedback and workflow control if needed.
    # HITL allows for human intervention in the agent's decision-making process, enabling more accurate and context-aware responses when necessary.
    # For now, the agent operates autonomously, but the state structure is designed to accommodate future enhancements for human oversight and feedback.


def _format_list_field(val: Any) -> str:
    """Helper to safely format list fields into comma-separated strings."""
    if isinstance(val, list):
        return ", ".join(str(item) for item in val)
    return str(val)


# Helper Function: Convert TravelAgentState to a structured prompt context string for LLM
def build_agent_context(state: TravelAgentState) -> str:
    """
    Constructs a structured context string from the current TravelAgentState.
    This context is used to inform the LLM about the user's profile, travel details, and preferences.
    """
    # 1. User Profile
    user_profile = []
    if state.get("user_age"): user_profile.append(f"Age: {state['user_age']}")
    if state.get("user_gender"): user_profile.append(f"Gender: {state['user_gender']}")
    if state.get("user_language"): user_profile.append(f"Language: {state['user_language']}")
    if state.get("user_nationality"): user_profile.append(f"Nationality: {state['user_nationality']}")
    profile_str = ", ".join(user_profile) if user_profile else "Not specified"

    # 2. Travel Details
    travel_details = []
    if state.get("origin"): travel_details.append(f"Origin: {state['origin']}")
    if state.get("destination"): travel_details.append(f"Destination: {state['destination']}")
    
    start_date = state.get("start_date")
    end_date   = state.get("end_date")
    if start_date or end_date:
        travel_details.append(f"Dates: {start_date or 'TBD'} to {end_date or 'TBD'}")

    if state.get("travelers_count"): travel_details.append(f"Travelers: {state['travelers_count']}")
    if state.get("companion_type"): travel_details.append(f"Companion: {state['companion_type']}")
    details_str = "\n  • ".join(travel_details) if travel_details else "Not specified"

    # 3. Preferences
    prefs = []
    if state.get("budget"): 
        prefs.append(f"Budget: {state['budget']}")
    if val := state.get("interests"): 
        prefs.append(f"Interests: {_format_list_field(val)}")
    if val := state.get("food_preferences"): 
        prefs.append(f"Food: {_format_list_field(val)}")
    if val := state.get("accommodation_preferences"): 
        prefs.append(f"Accommodation: {_format_list_field(val)}")
    if val := state.get("transportation_preferences"): 
        prefs.append(f"Transportation: {_format_list_field(val)}")
    if val := state.get("activities_preferences"): 
        prefs.append(f"Activities: {_format_list_field(val)}")
    prefs_str = "\n  • ".join(prefs) if prefs else "General"

    # 4. Draft Itinerary
    draft_str = state.get("draft_itinerary") or "None"

    # Return a structured context string
    return (
        "\n================ [CURRENT USER STATE & CONTEXT] ================\n"
        f"USER PROFILE: {profile_str}\n"
        f"TRIP DETAILS:\n  • {details_str}\n"
        f"PREFERENCES:\n  • {prefs_str}\n"
        f"DRAFT ITINERARY IN STATE:\n{draft_str}\n"
        "=================================================================\n"
    )
