from typing import Any

import requests

from app.core.config import get_settings


def get_weather(city: str) -> str:
    """Return current weather for a city using OpenWeather."""
    settings = get_settings()

    if not settings.openweather_api_key:
        return "Ошибка: API ключ OpenWeather не настроен в .env файле."

    params = {
        "q": city,
        "appid": settings.openweather_api_key,
        "units": "metric",
        "lang": "ru",
    }

    try:
        response = requests.get(
            settings.openweather_url,
            params=params,
            timeout=settings.openweather_timeout_seconds,
        )
        data: dict[str, Any] = response.json()

        if response.status_code == 404:
            return f"Город '{city}' не найден."

        if response.status_code != 200:
            return f"Ошибка API: {data.get('message', 'Неизвестная ошибка')}"

        return _format_weather_response(data)
    except requests.RequestException as exc:
        return f"Ошибка подключения: {exc}"
    except (KeyError, IndexError, TypeError) as exc:
        return f"Ошибка разбора ответа OpenWeather: {exc}"


def _format_weather_response(data: dict[str, Any]) -> str:
    main = data["main"]
    weather = data["weather"][0]
    wind = data.get("wind", {})

    return (
        f"Погода в {data['name']}: {weather['description'].capitalize()}. "
        f"Температура: {main['temp']}°C (ощущается как {main['feels_like']}°C). "
        f"Влажность: {main['humidity']}%, ветер: {wind.get('speed', 'N/A')} м/с."
    )
