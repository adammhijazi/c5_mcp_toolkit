"""
Lebanon Outing Planner: a FastMCP server with exactly three tools.

    Data tool          get_weather_forecast   reads a forecast, changes nothing
    Action tool        save_outing            writes a planned outing to disk
    Orchestration tool plan_outing            calls the Data step, picks the best day, then calls the Action step

Plus two read-only resources:
    outings://saved        the outings saved so far
    outings://activities   the weather rules used to judge each activity

Weather data comes from the free Open-Meteo API (no API key needed).
Run it:  python server.py      (stdio transport, which the agent in agent.py launches automatically)
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import httpx
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field

mcp = FastMCP(
    "lebanon-outing-planner",
    instructions="Weather-aware outing planner for Lebanon: forecasts, saving outings, and planning the best day.",
)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OUTINGS_FILE = Path(os.getenv("OUTINGS_FILE", Path(__file__).parent / "data" / "outings.json"))
HTTP_TIMEOUT = 10.0

# WMO weather codes used by Open-Meteo -> short text
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "rime fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers", 81: "heavy rain showers",
    82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail", 99: "severe thunderstorm",
}

# What "good weather" means for each activity (used only by the orchestration tool)
ACTIVITY_RULES = {
    "hiking":  {"min_temp": 10, "max_temp": 28, "max_rain_chance": 30, "max_wind": 35},
    "beach":   {"min_temp": 24, "max_temp": 38, "max_rain_chance": 20, "max_wind": 30},
    "picnic":  {"min_temp": 16, "max_temp": 32, "max_rain_chance": 20, "max_wind": 25},
    "cycling": {"min_temp": 10, "max_temp": 30, "max_rain_chance": 25, "max_wind": 25},
    "skiing":  {"min_temp": -15, "max_temp": 8, "max_rain_chance": 100, "max_wind": 45},
}


# ----------------------------------------------------------------------------- typed outputs
class DayForecast(BaseModel):
    date: str = Field(description="YYYY-MM-DD")
    condition: str = Field(description="Short weather description, e.g. 'partly cloudy'")
    temp_min_c: float
    temp_max_c: float
    rain_chance_pct: int = Field(description="Maximum precipitation probability during the day (0-100)")
    wind_max_kmh: float


class Forecast(BaseModel):
    city: str
    country: str
    latitude: float
    longitude: float
    days: list[DayForecast]


class SavedOuting(BaseModel):
    id: int
    city: str
    date: str
    activity: str
    note: str
    saved_at: str


class OutingPlan(BaseModel):
    status: str = Field(description="'saved' if a good day was found and saved, 'no_suitable_day' otherwise")
    city: str
    activity: str
    best_day: DayForecast | None = None
    reason: str
    saved_outing: SavedOuting | None = None
    days_considered: list[DayForecast]


# ----------------------------------------------------------------------------- core logic (plain functions)
OFFLINE = os.getenv("WEATHER_OFFLINE") == "1"      # use built-in sample data instead of the internet (tests / demos)

SAMPLE_PLACES = {
    "beirut": {"name": "Beirut", "country": "Lebanon", "country_code": "LB", "latitude": 33.89, "longitude": 35.49},
    "byblos": {"name": "Byblos", "country": "Lebanon", "country_code": "LB", "latitude": 34.12, "longitude": 35.65},
    "faraya": {"name": "Faraya", "country": "Lebanon", "country_code": "LB", "latitude": 34.01, "longitude": 35.83},
}
#                     code  max   min  rain wind   (one row per day, starting today)
SAMPLE_DAYS = [(63, 14.0, 8.0, 85, 30.0), (3, 17.0, 9.0, 40, 22.0), (1, 19.0, 10.0, 5, 12.0),
               (0, 21.0, 11.0, 0, 18.0), (61, 16.0, 9.0, 60, 28.0), (2, 22.0, 12.0, 10, 15.0), (95, 15.0, 8.0, 90, 40.0)]


def _sample_response(url: str, params: dict) -> dict:
    if url == GEOCODE_URL:
        place = SAMPLE_PLACES.get(params["name"].lower())
        return {"results": [place]} if place else {}
    n = params["forecast_days"]
    today = dt.date.today()
    rows = SAMPLE_DAYS[:n]
    return {"daily": {"time": [(today + dt.timedelta(days=i)).isoformat() for i in range(n)],
                      "weather_code": [r[0] for r in rows], "temperature_2m_max": [r[1] for r in rows],
                      "temperature_2m_min": [r[2] for r in rows], "precipitation_probability_max": [r[3] for r in rows],
                      "wind_speed_10m_max": [r[4] for r in rows]}}


def _http_get(url: str, params: dict) -> dict:
    """Single place for network access, so tests can replace it."""
    if OFFLINE:
        return _sample_response(url, params)
    try:
        response = httpx.get(url, params=params, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError) as e:
        raise ToolError(f"Weather service is unavailable right now ({type(e).__name__}). Please try again later.") from e


def fetch_forecast(city: str, days: int = 3) -> Forecast:
    if not city or not city.strip():
        raise ToolError("Please give a city name, for example 'Beirut' or 'Byblos'.")
    if not 1 <= days <= 7:
        raise ToolError(f"days must be between 1 and 7 (got {days}).")

    places = _http_get(GEOCODE_URL, {"name": city.strip(), "count": 5, "language": "en", "format": "json"}).get("results") or []
    if not places:
        raise ToolError(f"City '{city}' was not found. Check the spelling or try a nearby larger town.")
    place = next((p for p in places if p.get("country_code") == "LB"), places[0])   # prefer Lebanon

    data = _http_get(FORECAST_URL, {
        "latitude": place["latitude"], "longitude": place["longitude"], "timezone": "auto", "forecast_days": days,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
    })
    d = data.get("daily") or {}
    if not d.get("time"):
        raise ToolError("The weather service returned no forecast data. Please try again later.")
    out = [DayForecast(date=d["time"][i],
                       condition=WEATHER_CODES.get(d["weather_code"][i], f"code {d['weather_code'][i]}"),
                       temp_min_c=d["temperature_2m_min"][i], temp_max_c=d["temperature_2m_max"][i],
                       rain_chance_pct=int(d["precipitation_probability_max"][i] or 0),
                       wind_max_kmh=d["wind_speed_10m_max"][i])
           for i in range(len(d["time"]))]
    return Forecast(city=place["name"], country=place.get("country", ""), latitude=place["latitude"],
                    longitude=place["longitude"], days=out)


def _load_outings() -> list[dict]:
    if not OUTINGS_FILE.exists():
        return []
    try:
        return json.loads(OUTINGS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        raise ToolError(f"The outings file {OUTINGS_FILE.name} is corrupted. Fix or delete it and try again.")


def store_outing(city: str, date: str, activity: str, note: str = "") -> SavedOuting:
    try:
        day = dt.date.fromisoformat(date)
    except ValueError:
        raise ToolError(f"Invalid date '{date}'. Use the format YYYY-MM-DD, e.g. 2026-12-05.")
    if day < dt.date.today():
        raise ToolError(f"{date} is in the past. Choose today or a future date.")
    if not city.strip() or not activity.strip():
        raise ToolError("city and activity must not be empty.")

    outings = _load_outings()
    if any(o["city"].lower() == city.lower() and o["date"] == date and o["activity"].lower() == activity.lower()
           for o in outings):
        raise ToolError(f"An outing for {activity} in {city} on {date} is already saved.")
    outing = SavedOuting(id=max((o["id"] for o in outings), default=0) + 1, city=city.strip(), date=date,
                         activity=activity.strip().lower(), note=note.strip(),
                         saved_at=dt.datetime.now().isoformat(timespec="seconds"))
    OUTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTINGS_FILE.write_text(json.dumps(outings + [outing.model_dump()], indent=2, ensure_ascii=False), encoding="utf-8")
    return outing


def score_day(day: DayForecast, rules: dict) -> tuple[bool, float]:
    """(is the day acceptable?, score where lower is better)."""
    ok = (rules["min_temp"] <= day.temp_max_c <= rules["max_temp"] and day.rain_chance_pct <= rules["max_rain_chance"]
          and day.wind_max_kmh <= rules["max_wind"])
    ideal = (rules["min_temp"] + rules["max_temp"]) / 2
    return ok, day.rain_chance_pct + abs(day.temp_max_c - ideal) * 2 + day.wind_max_kmh * 0.5


# ----------------------------------------------------------------------------- MCP tools
@mcp.tool
def get_weather_forecast(city: str, days: int = 3) -> Forecast:
    """DATA TOOL. Get the daily weather forecast for a city (Lebanon is preferred when names are ambiguous).

    Args:
        city: City or town name, e.g. "Beirut", "Byblos", "Faraya".
        days: Number of days to forecast, from 1 to 7 (default 3).

    Returns: the resolved place and, for each day, the condition, min/max temperature (°C),
    maximum rain chance (%) and maximum wind speed (km/h). Read-only: it never changes anything.
    Errors: unknown city, days outside 1-7, or the weather service being unavailable.
    """
    return fetch_forecast(city, days)


@mcp.tool
def save_outing(city: str, date: str, activity: str, note: str = "") -> SavedOuting:
    """ACTION TOOL. Save a planned outing to the user's outing list.

    Args:
        city: Where the outing takes place.
        date: Day of the outing in YYYY-MM-DD format (today or later).
        activity: What the user will do, e.g. "hiking", "beach", "picnic".
        note: Optional free-text note (e.g. the expected weather).

    Returns: the saved outing with its id. Changes state: appends to the outings file.
    Errors: badly formatted or past date, empty fields, or the same outing already saved.
    """
    return store_outing(city, date, activity, note)


@mcp.tool
def plan_outing(city: str, activity: str, days: int = 5) -> OutingPlan:
    """ORCHESTRATION TOOL. Plan an outing: get the forecast (Data tool), pick the best day for the
    activity, then save that day (Action tool).

    Args:
        city: Where the user wants to go.
        activity: One of: hiking, beach, picnic, cycling, skiing.
        days: How many upcoming days to consider, from 1 to 7 (default 5).

    Returns: status 'saved' with the best day, the reason and the saved outing, or status
    'no_suitable_day' (nothing is saved) if no day meets the activity's weather rules.
    Errors: unknown activity, plus any error from the Data or Action step.
    """
    key = activity.strip().lower()
    if key not in ACTIVITY_RULES:
        raise ToolError(f"Unknown activity '{activity}'. Choose one of: {', '.join(ACTIVITY_RULES)}.")
    rules = ACTIVITY_RULES[key]

    forecast = fetch_forecast(city, days)                                     # step 1: Data
    scored = [(day, *score_day(day, rules)) for day in forecast.days]
    good = sorted([s for s in scored if s[1]], key=lambda s: s[2])
    if not good:
        return OutingPlan(status="no_suitable_day", city=forecast.city, activity=key, days_considered=forecast.days,
                          reason=(f"None of the next {len(forecast.days)} days fits {key}: it needs "
                                  f"{rules['min_temp']}-{rules['max_temp']}°C, at most {rules['max_rain_chance']}% rain "
                                  f"chance and wind under {rules['max_wind']} km/h. Nothing was saved."))
    best = good[0][0]
    reason = (f"{best.date} has the best {key} weather: {best.condition}, {best.temp_min_c:.0f}-{best.temp_max_c:.0f}°C, "
              f"{best.rain_chance_pct}% rain chance, wind up to {best.wind_max_kmh:.0f} km/h.")
    saved = store_outing(forecast.city, best.date, key, note=reason)          # step 2: Action
    return OutingPlan(status="saved", city=forecast.city, activity=key, best_day=best, reason=reason,
                      saved_outing=saved, days_considered=forecast.days)


# ----------------------------------------------------------------------------- MCP resources (pure data lookups)
@mcp.resource("outings://saved", mime_type="application/json")
def saved_outings() -> str:
    """All outings saved so far, as a JSON list."""
    return json.dumps(_load_outings(), indent=2, ensure_ascii=False)


@mcp.resource("outings://activities", mime_type="application/json")
def activity_rules() -> str:
    """The weather rules plan_outing uses for each activity."""
    return json.dumps(ACTIVITY_RULES, indent=2)


if __name__ == "__main__":
    mcp.run(show_banner=False)   # stdio transport
