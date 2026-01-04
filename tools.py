"""Tool implementations for AI agents: web search, weather, and math operations."""

import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()


def perform_search(query: str) -> str:
    """
    Search the web for up-to-date information.
    
    Returns a comprehensive summary including titles, snippets, and source URLs.
    
    Args:
        query: The search query to look up on the internet.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY missing."
    
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="advanced", max_results=3)
        
        results = response.get('results', [])
        if not results:
            return "No relevant search results found."
        
        search_context = []
        for item in results:
            title = item.get('title', 'No Title')
            content = item.get('content', '')
            url = item.get('url', '')
            search_context.append(f"Source: {title} ({url})\nContent: {content}")
            
        return "\n\n".join(search_context)
    except Exception as e:
        return f"Web search error: {str(e)}"


def fetch_weather(location: str) -> str:
    """
    Get current weather information for a location.
    
    Resolves location to coordinates and fetches real-time weather data.
    
    Args:
        location: The city or location name to get weather for.
    """
    try:
        geocode_url = f"https://geocoding-api.open-meteo.com/v1/search?name={quote(location)}&count=1"
        geo_data = requests.get(geocode_url).json()
        
        if not geo_data.get('results'):
            return f"Error: Location '{location}' not found."
        
        lat = geo_data['results'][0]['latitude']
        lon = geo_data['results'][0]['longitude']
        
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_data = requests.get(weather_url).json()
        
        if 'current_weather' in weather_data:
            current = weather_data['current_weather']
            return f"Weather in {location}: {current.get('temperature')}°C, Wind: {current.get('windspeed')} km/h"
        return "Error: Weather data not available."
    except Exception as e:
        return f"Error: {str(e)}"


def execute_math(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    
    Supports basic arithmetic: +, -, *, /, and parentheses.
    
    Args:
        expression: The mathematical expression to evaluate (e.g., "2 * (3 + 4)").
    """
    try:
        valid_chars = "0123456789+-*/(). "
        if all(c in valid_chars for c in expression):
            return str(eval(expression))
        return "Error: Invalid characters in expression."
    except Exception as e:
        return f"Error: {str(e)}"


# Aliases for backward compatibility
web_search = perform_search
get_current_weather = fetch_weather
calculator = execute_math
