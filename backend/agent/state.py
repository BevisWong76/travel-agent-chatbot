from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class TravelAgentState(TypedDict):
    # 1. Conversation History (Memory)
    # Use Annotated to add metadata for LangGraph to recognize this field as a message list
    messages: Annotated[List[BaseMessage], add_messages]
    
    # 2. User Profile / Demographics
    user_age: Optional[int]
    user_gender: Optional[str]
    user_nationality: Optional[str]
    user_language: Optional[str]
    
    # 3. Travel Details
    origin: Optional[str]
    destination: Optional[str]
    
    start_date: Optional[str]
    end_date: Optional[str]
    duration_days: Optional[int]
    
    travelers_count: Optional[int]          
    companion_type: Optional[str]   
    
    # 4. Preferences
    budget: Optional[str]
    interests: Optional[List[str]]
    accommodation_preferences: Optional[List[str]]
    food_preferences: Optional[List[str]]
    activities_preferences: Optional[List[str]]
    
    # 5. Human in the Loop (HITL) & Workflow Control
    current_step: Optional[str]             
    require_human_feedback: Optional[bool]
    human_feedback: Optional[str]
    draft_itinerary: Optional[str]