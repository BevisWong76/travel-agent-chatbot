# backend/tools/external_api_tools.py
from langchain_core.tools import tool

@tool
def maps_tool(origin: str, destination: str, mode: str = "transit") -> str:
    """Calculates route distance and travel time between two locations."""
    # TODO: Maps API
    # Google Maps / DuckDuckGo / OpenStreetMap / Nominatim
    
    # Return the response in a string or JSON to be used by the agent
    return f"Estimated transit time from {origin} to {destination} is 20 mins. (Mock response)"

@tool
def weather_tool(city: str, date: str) -> str:
    """Gets weather forecast for a given city and travel date."""
    # TODO: Weather API
    # OpenWeather Open-Meteo API
    return f"Weather forecast for {city} on {date}: Sunny, 22°C. (Mock response)"

@tool
def flights_tool(origin: str, destination: str, date: str) -> str:
    """Searches flight options and prices between cities."""
    # TODO: Flight API
    # Skyscanner / Kiwi / Amadeus / Google Flights
    return f"Lowest flight fare from {origin} to {destination} on {date} is ~$350. (Mock response)"