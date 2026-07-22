import os
from datetime import datetime
import httpx
import googlemaps
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

# --- 1. Google Maps Directions Tool ---

# Initialize Google Maps Client
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
gmaps = googlemaps.Client(key=API_KEY) if API_KEY else None

def parse_departure_time(departure_time_str: str):
    """
    Parses an ISO format string or 'now' into a format accepted by Google Maps API.
    Google Directions API requires an integer Unix timestamp for specific schedule queries.
    """
    if not departure_time_str or departure_time_str.lower() == "now":
        return "now"

    try:
        # Handles ISO strings like '2026-09-09T10:00:00' or '2026-09-09T10:00:00+09:00'
        dt = datetime.fromisoformat(departure_time_str)
        return int(dt.timestamp())
    except Exception as e:
        print(f"[DEBUG] Failed to parse departure_time '{departure_time_str}': {e}. Defaulting to 'now'.")
        return "now"

def format_directions_response(directions: list, mode: str) -> str:
    """
    Formats the raw Google Maps Directions API response into a clear, 
    noise-free summary for the LLM Agent.
    """
    route = directions[0]
    leg = route["legs"][0]

    summary = [
        f"Route Summary ({mode.upper()}):",
        f"• From: {leg['start_address']}",
        f"• To: {leg['end_address']}",
        f"• Distance: {leg['distance']['text']}",
        f"• Estimated Duration: {leg['duration']['text']}",
        "\nDetailed Steps:"
    ]

    for idx, step in enumerate(leg["steps"], 1):
        travel_mode = step.get("travel_mode", "").upper()
        
        # Handle Transit Steps with additional details
        if travel_mode == "TRANSIT":
            transit_details = step.get("transit_details", {})
            line = transit_details.get("line", {})
            vehicle = line.get("vehicle", {}).get("name", "Transit")
            line_name = line.get("short_name") or line.get("name", "Unknown Line")
            departure_stop = transit_details.get("departure_stop", {}).get("name", "")
            arrival_stop = transit_details.get("arrival_stop", {}).get("name", "")
            num_stops = transit_details.get("num_stops", 0)

            summary.append(
                f"  {idx}. [{vehicle}] Take {line_name} from '{departure_stop}' "
                f"to '{arrival_stop}' ({num_stops} stops, {step['duration']['text']})"
            )
        else:
            # Driving, Walking, or Bicycling instruction
            instructions = step.get("html_instructions", "")
            # Clean basic HTML tags from instructions
            clean_instructions = (
                instructions.replace("<b>", "")
                .replace("</b>", "")
                .replace('<div style="font-size:0.9em">', " (")
                .replace("</div>", ")")
                .replace("<wbr/>", "")
            )
            summary.append(f"  {idx}. [{travel_mode}] {clean_instructions} ({step['distance']['text']})")

    return "\n".join(summary)

@tool
async def maps_tool(origin: str, destination: str, mode: str = "transit", departure_time: str = "now") -> str:
    """
    Queries Google Maps Directions API for routing details between origin and destination.
    
    Args:
        origin: Departure location (e.g., 'Shibuya Station, Tokyo' or 'Taipei Main Station')
        destination: Arrival location (e.g., 'Tokyo Tower' or 'Taipei 101')
        mode: Transport mode ('transit', 'driving', 'walking', 'bicycling')
        departure_time: ISO timestamp (e.g., '2026-09-09T10:00:00+09:00') or 'now'
    """
    if not gmaps:
        return "ERROR: GOOGLE_MAPS_API_KEY is missing or invalid."

    parsed_time = parse_departure_time(departure_time)

    try:
        # Call Google Maps Directions API
        directions = gmaps.directions(
            origin=origin,
            destination=destination,
            mode=mode,
            departure_time=parsed_time,
        )

        # Handle ZERO_RESULTS transparently without leaking fake driving/walking data
        if not directions:
            return (
                f"STATUS: ZERO_RESULTS\n"
                f"No {mode} routes were found from '{origin}' to '{destination}' "
                f"for departure time '{departure_time}'.\n"
                f"Note: Transit options may be closed at this hour or schedule data is unavailable for this location. "
                f"Consider querying with a different mode (e.g., 'driving' or 'walking') or adjusting the departure time."
            )

        return format_directions_response(directions, mode)

    except Exception as e:
        return f"STATUS: API_ERROR\nFailed to fetch directions: {str(e)}"


# --- 2. Weather Forecast Tool ---
@tool
async def weather_tool(location: str, date: str = "today") -> str:
    """
    Gets the weather forecast for a given location (city or neighborhood) and travel date.
    
    Args:
        location: City or district name (e.g., 'Tokyo', 'Shibuya', 'Osaka', 'Paris').
        date: Target date or description (e.g., 'today', 'next week', '2026-04-01').
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Step 1: Geocoding - Get latitude and longitude for the location
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_res = await client.get(geo_url)
            geo_data = geo_res.json()

            if not geo_data.get("results"):
                return (
                    f"Location '{location}' could not be found. "
                    "Please ask the user to clarify or provide a nearby major city."
                )

            result = geo_data["results"][0]
            lat = result["latitude"]
            lon = result["longitude"]
            city_name = result.get("name", location)
            country = result.get("country", "")

            # Step 2: Fetch Weather Data
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}&"
                f"current=temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m&"
                f"daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,precipitation_probability_max,uv_index_max&"
                f"timezone=auto"
            )
            weather_res = await client.get(weather_url)
            weather_data = weather_res.json()

            current = weather_data.get("current", {})
            daily = weather_data.get("daily", {})

            # Step 3: WMO Weather Code Mapping
            weather_code_map = {
                0: "Clear sky ☀️",
                1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
                45: "Foggy 🌫️", 48: "Depositing rime fog 🌫️",
                51: "Light drizzle 🌧️", 61: "Slight rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
                80: "Slight rain showers 🌦️", 95: "Thunderstorm 🌩️"
            }

            curr_temp = current.get("temperature_2m", "N/A")
            curr_apparent_temp = current.get("apparent_temperature", "N/A")
            curr_code = current.get("weather_code", 0)
            curr_condition = weather_code_map.get(curr_code, "Varied conditions")

            # Step 4: Prepare 3 days Forecast Summary
            forecast_summary = []
            if daily and "time" in daily:
                for i in range(min(3, len(daily["time"]))):
                    d_time = daily["time"][i]
                    max_t = daily["temperature_2m_max"][i]
                    min_t = daily["temperature_2m_min"][i]
                    feels_max = daily["apparent_temperature_max"][i]
                    feels_min = daily["apparent_temperature_min"][i]
                    precip = daily["precipitation_probability_max"][i]
                    uv_max = daily["uv_index_max"][i] if "uv_index_max" in daily else "N/A"
                    code = daily["weather_code"][i]
                    cond = weather_code_map.get(code, "Clear/Cloudy")

                    forecast_summary.append(
                        f"• {d_time}: {cond}, {min_t}°C - {max_t}°C (Feels like: {feels_min}°C - {feels_max}°C) | Rain prob: {precip}% | UV: {uv_max}"
                    )

            forecast_text = "\n".join(forecast_summary)

            return (
                f"Weather report for {city_name}, {country}:\n"
                f"Current Temp: {curr_temp}°C, Apparent Temp: {curr_apparent_temp}°C, Condition: {curr_condition}\n"
                f"Upcoming Forecast:\n{forecast_text}"
            )

    except Exception as e:
        return f"Error fetching weather data for {location}: {str(e)}"
    

if __name__ == "__main__":
    import asyncio

    async def main_test():
        print("=== Testing Maps Tool ===")
        res_maps = await maps_tool.ainvoke({
            "origin": "Taipei Main Station",
            "destination": "Taipei 101",
            "mode": "walking",
            "departure_time": "2026-09-15T10:00:00+08:00"
        })
        print(res_maps)

        print("\n=== Testing Weather Tool (Shibuya) ===")
        res_weather1 = await weather_tool.ainvoke({"location": "Shibuya"})
        print(res_weather1)

        print("\n=== Testing Weather Tool (Paris) ===")
        res_weather2 = await weather_tool.ainvoke({"location": "Paris"})
        print(res_weather2)

    asyncio.run(main_test())
    

# uv run python backend/tools/external_api_tools.py
# uv run python -m backend.tools.external_api_tools