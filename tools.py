from tavily import TavilyClient
import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

def web_search(query: str):
    """
    Search for up-to-date information on the web.
    This tool returns a comprehensive summary of search results including titles, snippets, and source URLs.

    Args:
        query (str): The search query to look up on the internet.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY missing."
    
    try:
        client = TavilyClient(api_key=api_key)
        # Using search_depth="advanced" and including raw content/snippets for better context
        response = client.search(
            query=query, 
            search_depth="advanced", 
            max_results=3
        )
        
        results = response.get('results', [])
        if not results:
            return "No relevant search results found."
            
        # Combining snippets for the LLM to process
        search_context = []
        for result_item in results:
            title = result_item.get('title', 'No Title')
            content = result_item.get('content', '')
            url = result_item.get('url', '')
            search_context.append(f"Source: {title} ({url})\nContent: {content}")
            
        return "\n\n".join(search_context)
    except Exception as e:
        return f"Web search error: {str(e)}"

def get_current_weather(location: str):
    """
    Get the current weather information for a specified location.
    The tool first resolves the location to coordinates and then fetches real-time weather data.

    Args:
        location (str): The name of the city or location to get weather for.
    """
    try:
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote(location)}&count=1"
        geo_data = requests.get(geocode_url).json()
        
        if not geo_data.get('results'):
            return f"Error: Location '{location}' not found."
            
        latitude = geo_data['results'][0]['latitude']
        longitude = geo_data['results'][0]['longitude']
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current_weather=true"
        weather_data = requests.get(weather_url).json()
        
        if 'current_weather' in weather_data:
            current_weather = weather_data['current_weather']
            return f"Weather in {location}: {current_weather.get('temperature')}°C, Wind: {current_weather.get('windspeed')} km/h"
        return "Error: Weather data not available."
    except Exception as e:
        return f"Error: {str(e)}"

def calculator(expression: str):
    """
    Evaluate a mathematical expression safely.
    Supports basic arithmetic operations: addition (+), subtraction (-), multiplication (*), division (/), and parentheses.

    Args:
        expression (str): The mathematical expression to evaluate (e.g., "2 * (3 + 4)").
    """
    try:
        valid_chars = "0123456789+-*/(). "
        if all(c in valid_chars for c in expression):
            return eval(expression)
        return "Error: Invalid characters."
    except Exception as e:
        return f"Error: {str(e)}"
