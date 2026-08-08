from typing import Any, Dict
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from langchain_core.tools import InjectedToolCallId
from typing_extensions import Annotated

@tool
def update_travel_state_tool(
    updates: Dict[str, Any],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """
    Use this tool to update the user's travel state whenever new details are provided.
    Pass ONLY the key-value pairs that need to be updated.
    
    ALLOWED KEYS IN `updates` DICTIONARY:
    - user_age (int): User's age.
    - user_gender (str): e.g., 'Male', 'Female', etc.
    - user_language (str): Preferred language.
    - user_nationality (str): User's country of origin.
    - origin (str): Departure city/airport.
    - destination (str): Destination city/country.
    - start_date (str): YYYY-MM-DD.
    - end_date (str): YYYY-MM-DD.
    - travelers_count (int): Number of people travelling.
    - companion_type (str): e.g., 'Solo', 'Couple', 'Family with kids', 'Friends'.
    - budget (str): e.g., 'Low', 'Moderate', 'Luxury'.
    - interests (List[str]): e.g., ['Anime', 'History', 'Nature'].
    - food_preferences (List[str]): e.g., ['Vegetarian', 'Seafood', 'Ramen'].
    - accommodation_preferences (List[str]): e.g., ['Hotel', 'Airbnb', 'Resort'].
    - transportation_preferences (List[str]): e.g., ['Public Transit', 'Rental Car'].
    - activities_preferences (List[str]): e.g., ['Museums', 'Hiking', 'Shopping'].
    - weather_forecast (str): Weather forecast for the trip.
    """

    # Extract the keys that were updated for logging purposes
    print(f"[Tool] Updating travel state with keys: {list(updates.keys())}")
    updated_keys = list(updates.keys()) if isinstance(updates, dict) else []
    msg = "Successfully updated state fields: " + ", ".join(updated_keys)

    return Command(
        update={
            **updates,  
            "messages": [ToolMessage(content=msg, tool_call_id=tool_call_id)]
        }
    )