import logging
from datetime import datetime
from typing import Any

import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def get_weather(city: str, target_date: str | None = None) -> str:
    """Return weather for a city and target date using OpenWeather APIs."""
    settings = get_settings()
    today = datetime.now().date()

    if target_date:
        try:
            requested_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            logger.warning("Invalid weather target_date='%s'", target_date)
            return "Некорректная дата. Используйте формат YYYY-MM-DD."
    else:
        requested_date = today

    if not settings.openweather_api_key:
        logger.warning("OpenWeather API key is missing")
        return "Ошибка: API ключ OpenWeather не настроен в .env файле."

    params = {
        "q": city,
        "appid": settings.openweather_api_key,
        "units": "metric",
        "lang": "ru",
    }

    if requested_date < today:
        logger.info("Historical weather requested but unsupported: city=%s date=%s", city, requested_date)
        return "Историческая погода не поддерживается. Укажите сегодня или будущую дату."

    if requested_date > today:
        return _get_forecast_weather(city=city, requested_date=requested_date, settings=settings, params=params)

    logger.info("Fetching current weather for city=%s date=%s", city, requested_date)
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


def _get_forecast_weather(
    city: str,
    requested_date,
    settings,
    params: dict[str, Any],
) -> str:
    logger.info("Fetching forecast for city=%s date=%s", city, requested_date)
    try:
        response = requests.get(
            settings.openweather_forecast_url,
            params=params,
            timeout=settings.openweather_timeout_seconds,
        )
        data: dict[str, Any] = response.json()

        if response.status_code == 404:
            logger.info("Forecast city not found: %s", city)
            return f"Город '{city}' не найден."

        if response.status_code != 200:
            logger.warning(
                "OpenWeather forecast API error: status=%s message=%s",
                response.status_code,
                data.get("message"),
            )
            return f"Ошибка API прогноза: {data.get('message', 'Неизвестная ошибка')}"

        forecasts = data.get("list", [])
        day_points = [
            item
            for item in forecasts
            if datetime.fromtimestamp(item.get("dt", 0)).date() == requested_date
        ]
        if not day_points:
            return (
                f"Нет прогноза на {requested_date.isoformat()} для '{city}'. "
                "OpenWeather обычно дает прогноз примерно на 5 дней."
            )

        midpoint = day_points[len(day_points) // 2]
        return _format_forecast_response(midpoint, data.get("city", {}).get("name", city), requested_date.isoformat())
    except requests.RequestException:
        logger.exception("Forecast request failed for city=%s date=%s", city, requested_date)
        return "Ошибка подключения к сервису прогноза."
    except (KeyError, IndexError, TypeError, ValueError):
        logger.exception("Forecast response parsing failed for city=%s date=%s", city, requested_date)
        return "Ошибка разбора ответа прогноза погоды."


def _format_forecast_response(item: dict[str, Any], city_name: str, target_date: str) -> str:
    main = item["main"]
    weather = item["weather"][0]
    wind = item.get("wind", {})
    dt_txt = item.get("dt_txt", target_date)
    return (
        f"Прогноз в {city_name} на {target_date} ({dt_txt}): {weather['description'].capitalize()}. "
        f"Температура: {main['temp']}°C (ощущается как {main['feels_like']}°C). "
        f"Влажность: {main['humidity']}%, ветер: {wind.get('speed', 'N/A')} м/с."
    )
