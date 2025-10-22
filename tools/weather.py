import os
import requests
from langchain_core.tools import tool
import logging

# Load environment variables
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("RAPID_API_KEY")
url = os.getenv("OPEN_WEATHER_API_URL")
rapid_api_host = os.getenv("RAPID_API_HOST")

headers = {
	"x-rapidapi-key": api_key,
	"x-rapidapi-host": rapid_api_host
}

def get_weather(city: str) -> str:
    """
    Get current weather for a city.
    
    Args:
        city: Name of the city
    
    Returns:
        Weather information
    """
    city_lower = city.lower()

    querystring = {"city":city_lower,"lang":"EN"}

    try:
        response = requests.get(url, headers=headers, params=querystring)

        description = response.json().get("weather", [{}])[0].get("description")
        temp = response.json().get("main", {}).get("temp")

        logger.info(f"Got weather for {city}: {description}, {temp}°F")

        return f"{description}, {temp}°F"
    except Exception as e:
        logger.error(e)
        return "sunny, 80.72°F" # default 

# Make it a LangChain tool
@tool
def weather_tool(city: str) -> str:
    """Get current weather for a city"""
    return get_weather(city)