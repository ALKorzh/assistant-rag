import logging
from typing import Any

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_weather(city: str) -> str:
    """Return current weather for a city using OpenWeather."""
    settings = get_settings()

    if not settings.openweather_api_key:
        logger.warning("OpenWeather API key is missing")
        return "Ошибка: API ключ OpenWeather не настроен в .env файле."

    params = {
        "q": city,
        "appid": settings.openweather_api_key,
        "units": "metric",
        "lang": "ru",
    }

    logger.info("Fetching weather for city=%s", city)
    try:
        response = requests.get(
            settings.openweather_url,
            params=params,
            timeout=settings.openweather_timeout_seconds,
        )
        data: dict[str, Any] = response.json()

        if response.status_code == 404:
            logger.info("Weather city not found: %s", city)
            return f"Город '{city}' не найден."

        if response.status_code != 200:
            logger.warning("OpenWeather API error: status=%s message=%s", response.status_code, data.get("message"))
            return f"Ошибка API: {data.get('message', 'Неизвестная ошибка')}"

        logger.info("Weather response received for city=%s", city)
        return _format_weather_response(data)
    except requests.RequestException:
        logger.exception("Weather request failed for city=%s", city)
        return "Ошибка подключения к погодному сервису."
    except (KeyError, IndexError, TypeError):
        logger.exception("Weather response parsing failed for city=%s", city)
        return "Ошибка разбора ответа OpenWeather."


def _format_weather_response(data: dict[str, Any]) -> str:
    main = data["main"]
    weather = data["weather"][0]
    wind = data.get("wind", {})

    return (
        f"Погода в {data['name']}: {weather['description'].capitalize()}. "
        f"Температура: {main['temp']}°C (ощущается как {main['feels_like']}°C). "
        f"Влажность: {main['humidity']}%, ветер: {wind.get('speed', 'N/A')} м/с."
    )
