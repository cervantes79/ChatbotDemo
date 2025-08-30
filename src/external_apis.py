import os
import requests
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class WeatherAPI:
    def __init__(self):
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def get_weather(self, city: str) -> Optional[Dict[str, Any]]:
        try:
            if not self.api_key:
                logger.error("OPENWEATHER_API_KEY not found in environment variables")
                return None
            
            if not city or not city.strip():
                logger.error("City name is required")
                return None
            
            params = {
                "q": city.strip(),
                "appid": self.api_key,
                "units": "metric"
            }
            
            logger.info(f"Fetching weather data for: {city}")
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                weather_info = {
                    "city": data.get("name", "Unknown"),
                    "country": data.get("sys", {}).get("country", "Unknown"),
                    "temperature": data.get("main", {}).get("temp", 0),
                    "feels_like": data.get("main", {}).get("feels_like", 0),
                    "humidity": data.get("main", {}).get("humidity", 0),
                    "pressure": data.get("main", {}).get("pressure", 0),
                    "description": data.get("weather", [{}])[0].get("description", "Unknown"),
                    "wind_speed": data.get("wind", {}).get("speed", 0),
                    "visibility": data.get("visibility", 0) / 1000 if data.get("visibility") else 0
                }
                
                logger.info(f"Successfully fetched weather for {weather_info['city']}")
                return weather_info
                
            elif response.status_code == 404:
                logger.error(f"City not found: {city}")
                return None
            elif response.status_code == 401:
                logger.error("Invalid API key")
                return None
            else:
                logger.error(f"API request failed with status code: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error("Weather API request timed out")
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to weather API")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather API request failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in weather API: {str(e)}")
            return None

    def format_weather_response(self, weather_data: Dict[str, Any]) -> str:
        try:
            if not weather_data:
                return "Weather information is currently unavailable."
            
            response = f"""Weather in {weather_data['city']}, {weather_data['country']}:
- Temperature: {weather_data['temperature']}°C (feels like {weather_data['feels_like']}°C)
- Condition: {weather_data['description'].title()}
- Humidity: {weather_data['humidity']}%
- Wind Speed: {weather_data['wind_speed']} m/s
- Pressure: {weather_data['pressure']} hPa"""
            
            if weather_data['visibility'] > 0:
                response += f"\n- Visibility: {weather_data['visibility']} km"
                
            return response
            
        except Exception as e:
            logger.error(f"Error formatting weather response: {str(e)}")
            return "Error formatting weather information."