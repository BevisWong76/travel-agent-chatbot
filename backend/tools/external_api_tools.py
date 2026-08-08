import os
from datetime import datetime, date, timedelta
from typing import Optional
import httpx
import googlemaps
from dotenv import load_dotenv
from langchain_core.tools import tool
from async_lru import alru_cache
from langchain_core.tools import tool

load_dotenv()

# =====================================================================
# 1. Google Maps Directions Tool
# =====================================================================

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
        print(f"[Maps Tools - DEBUG] Failed to parse departure_time '{departure_time_str}': {e}. Defaulting to 'now'.")
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

    print(f"[Maps Tools] Directions Summary: {len(summary)} steps formatted.")
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
        print(f"[Maps Tools] Fetching directions from '{origin}' to '{destination}' via {mode} at '{departure_time}'")
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
        print(f"[Maps Tools] Directions fetched successfully. Processing {len(directions[0]['legs'][0]['steps'])} steps...")

        return format_directions_response(directions, mode)

    except Exception as e:
        return f"STATUS: API_ERROR\nFailed to fetch directions: {str(e)}"


# =====================================================================
# 2. Weather Forecast Tool
# =====================================================================

# Default shared HTTPX client for async requests with connection pooling
shared_httpx_client = httpx.AsyncClient(
    timeout=5.0,
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
)

# WMO Weather Code Mapping
WEATHER_CODE_MAP = {
    0: "Clear sky ☀️",
    1: "Mainly clear 🌤️", 2: "Partly cloudy ⛅", 3: "Overcast ☁️",
    45: "Foggy 🌫️", 48: "Depositing rime fog 🌫️",
    51: "Light drizzle 🌧️", 61: "Slight rain 🌧️", 63: "Moderate rain 🌧️", 65: "Heavy rain 🌧️",
    80: "Slight rain showers 🌦️", 95: "Thunderstorm 🌩️"
}

# Cache ONLY the geocoding coordinates (Location -> Lat/Lon never changes)
# This achieves a 100% Cache Hit Rate for repeated cities with zero memory side-effects.
@alru_cache(maxsize=500)
async def _get_coordinates(location_clean: str):
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_clean}&count=1&language=en&format=json"
    geo_res = await shared_httpx_client.get(geo_url)
    geo_data = geo_res.json()

    if not geo_data.get("results"):
        return None

    result = geo_data["results"][0]
    return {
        "lat": result["latitude"],
        "lon": result["longitude"],
        "city_name": result.get("name", location_clean),
        "country": result.get("country", "")
    }

async def _fetch_weather_data(location: str, start_date: Optional[str] = None, end_date: Optional[str] = None ) -> str:
    location_clean = location.strip().lower()
    
    # 1. Get coordinates
    geo_info = await _get_coordinates(location_clean)
    if not geo_info:
        return f"Location '{location}' could not be found. Please ask the user to clarify or provide a nearby major city."

    lat, lon = geo_info["lat"], geo_info["lon"]
    city_name, country = geo_info["city_name"], geo_info["country"]

    # 2. Parse start_date and end_date, defaulting to 3 days forcast if not provided
    today = date.today()
    s_date = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else today
    e_date = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else s_date + timedelta(days=2)
    
    # Use the Open-Meteo API to fetch weather data based on the date range
    days_from_today = (s_date - today).days
    if -365 <= days_from_today < 0:
        # A: Past date within the last year -> Call Archive API for historical data
        weather_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date={s_date}&end_date={e_date}&"
            f"daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum&"
            f"timezone=auto"
        )
        data_type = "Historical Data"
        
    elif 0 <= days_from_today <= 14:
        # B: Future date within 14 days -> Call Forecast API (accurate forecast)
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date={s_date}&end_date={e_date}&"
            f"daily=weather_code,temperature_2m_max,temperature_2m_min,apparent_temperature_max,apparent_temperature_min,precipitation_probability_max,uv_index_max&"
            f"timezone=auto"
        )
        data_type = "Live Forecast"
        
    else:
        # C: Future date beyond 14 days (e.g., half a year later) -> Automatically fetch "same period last year's historical data" for climate reference
        last_year_s_date = s_date.replace(year=s_date.year - 1)
        last_year_e_date = e_date.replace(year=e_date.year - 1)
        
        weather_url = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}&"
            f"start_date={last_year_s_date}&end_date={last_year_e_date}&"
            f"daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum&"
            f"timezone=auto"
        )
        data_type = f"Historical Climate Estimate (based on same period last year: {last_year_s_date} to {last_year_e_date})"

    # 4. Fetch weather data from the appropriate API endpoint
    weather_res = await shared_httpx_client.get(weather_url, timeout=5.0)
    weather_data = weather_res.json()
    daily = weather_data.get("daily", {})

    times = daily.get("time", [])
    if not times:
        return f"No weather data available for {city_name} for the requested dates ({s_date} to {e_date})."

    # 5. Parse and format the results
    forecast_summary = []
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip_list = daily.get("precipitation_probability_max") or daily.get("precipitation_sum") or []
    uv_list = daily.get("uv_index_max", [])
    codes = daily.get("weather_code", [])

    for i in range(len(times)):
        d_time = times[i]
        max_t = max_temps[i] if i < len(max_temps) else "N/A"
        min_t = min_temps[i] if i < len(min_temps) else "N/A"
        precip = precip_list[i] if i < len(precip_list) else "N/A"
        uv_max = uv_list[i] if i < len(uv_list) else "N/A"
        code = codes[i] if i < len(codes) else 0
        cond = WEATHER_CODE_MAP.get(code, "Clear/Cloudy")

        forecast_summary.append(
            f"• {d_time}: {cond}, {min_t}°C - {max_t}°C | Rain/Precip: {precip}% | UV: {uv_max}"
        )

    return (
        f"Weather report for {city_name}, {country} [{data_type}]:\n"
        + "\n".join(forecast_summary)
    )


@tool
async def weather_tool(location: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> str:
    """Gets the weather forecast or historical weather for a given location and optional date range.
    
    Args:
        location: City or district name (e.g., 'Tokyo', 'Paris').
        start_date: Optional start date in 'YYYY-MM-DD' format (e.g., '2026-10-15').
        end_date: Optional end date in 'YYYY-MM-DD' format (e.g., '2026-10-18').
    """
    try:
        print(f"[Weather Tools] Fetching weather for {location} from {start_date} to {end_date}")
        res = await _fetch_weather_data(location, start_date, end_date)
        print(f"[Weather Tools] Weather fetch result: {res[:100]}...")
        return res
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