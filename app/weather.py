"""
app/weather.py — Live Weather for SiteWatch
============================================

Two-layer weather pipeline:
  1. wttr.in (JSON) → current conditions RIGHT NOW  (primary, fast, no key)
  2. Open-Meteo geocode + forecast fallback          (used when wttr.in fails)

Both are free, no API key required.

Location sanity check:
  If wttr.in returns a city name that doesn't match what was asked
  (e.g. 'Jagoe' when 'Denton' was asked), retry with ', US' appended,
  then fall back to Open-Meteo which uses explicit geocoding.

Public API:
  get_current_conditions(location) → dict with keys:
      success, location, temp_c, feels_like_c, wind_kmh, gust_kmh,
      humidity_pct, precip_mm, description, wind_dir, summary_text, risk_flags
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import requests

# ── Constants ─────────────────────────────────────────────────────────────────

_WTTR_URL     = "https://wttr.in/{location}?format=j1"
_GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_TIMEOUT      = 10   # seconds


# ── Location sanity check ────────────────────────────────────────────────────

def _location_matches(returned_city: str, asked: str) -> bool:
    """
    True if the returned city name shares at least one significant word
    with the location the user asked for.
    'Denton' vs 'Jagoe'          → False  → trigger retry
    'Austin' vs 'Austin, US'     → True
    """
    stop_words = {"the", "of", "and", "city", "town", "village"}
    asked_words = {
        w.lower() for w in re.split(r"[,\s]+", asked)
        if len(w) > 2 and w.lower() not in stop_words
    }
    city_words = {
        w.lower() for w in re.split(r"[,\s]+", returned_city)
        if len(w) > 2 and w.lower() not in stop_words
    }
    return bool(asked_words & city_words)


# ── wttr.in fetcher ──────────────────────────────────────────────────────────

def _wttr_fetch_raw(query: str) -> Optional[Dict[str, Any]]:
    """Single wttr.in request; returns normalised dict or None."""
    try:
        url = _WTTR_URL.format(location=requests.utils.quote(query))
        r   = requests.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        cur  = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0]
        city    = area.get("areaName",  [{}])[0].get("value", query)
        country = area.get("country",   [{}])[0].get("value", "")
        display = f"{city}, {country}" if country else city

        temp_c    = float(cur.get("temp_C",         0))
        feels_c   = float(cur.get("FeelsLikeC",     temp_c))
        wind_kmh  = float(cur.get("windspeedKmph",  0))
        humidity  = float(cur.get("humidity",        0))
        precip_mm = float(cur.get("precipMM",        0))
        desc      = cur.get("weatherDesc", [{}])[0].get("value", "")
        wind_dir  = cur.get("winddir16Point", "")
        # wttr.in field name varies — try several spellings
        gust_kmh = float(
            cur.get("WindGustKmph") or
            cur.get("windGustKmph") or
            cur.get("Windgust_kmph") or
            wind_kmh        # fallback: treat gusts == sustained
        )

        return {
            "success":      True,
            "location":     display,
            "_city_raw":    city,   # used for sanity check only
            "temp_c":       temp_c,
            "feels_like_c": feels_c,
            "wind_kmh":     wind_kmh,
            "gust_kmh":     gust_kmh,
            "humidity_pct": humidity,
            "precip_mm":    precip_mm,
            "description":  desc,
            "wind_dir":     wind_dir,
        }
    except Exception:
        return None


def _wttr_fetch(location: str) -> Optional[Dict[str, Any]]:
    """
    Fetch from wttr.in with location sanity check.
    If the returned city name doesn't match what was asked, returns None
    so the caller falls through to Open-Meteo's explicit geocoding.
    """
    res = _wttr_fetch_raw(location)
    if res:
        if _location_matches(res.get("_city_raw", ""), location):
            res.pop("_city_raw", None)
            return res
        # Sanity check failed — wrong location returned (e.g. 'Jagoe' for 'Denton')
        # Don't return garbage; fall through to Open-Meteo
        res.pop("_city_raw", None)
    return None


# ── Open-Meteo fallback ──────────────────────────────────────────────────────

def _strip_to_city(location: str) -> str:
    """
    Strip state/country suffix so the Open-Meteo geocoder works.
    'Austin, TX'        → 'Austin'
    'Denton, Texas'     → 'Denton'
    'Sydney, Australia' → 'Sydney'
    """
    return re.split(r"[,\s]+(?:[A-Z][a-z]+|[A-Z]{2,})$", location.strip())[0].strip()


def _geocode(location: str) -> Optional[Dict[str, Any]]:
    """
    Geocode via Open-Meteo. Retries with city-only if the full string fails.
    """
    for query in [location, _strip_to_city(location)]:
        try:
            r = requests.get(
                _GEOCODE_URL,
                params={"name": query, "count": 1, "language": "en", "format": "json"},
                timeout=_TIMEOUT,
            )
            data = r.json()
            if data.get("results"):
                return data["results"][0]
        except Exception:
            pass
    return None


def _openmeteo_current(location: str) -> Optional[Dict[str, Any]]:
    """Geocode + most-recent-hour forecast from Open-Meteo."""
    geo = _geocode(location)
    if not geo:
        return None

    lat   = geo["latitude"]
    lon   = geo["longitude"]
    name  = geo.get("name", location)
    cntry = geo.get("country", "")
    display = f"{name}, {cntry}" if cntry else name

    try:
        r = requests.get(
            _FORECAST_URL,
            params={
                "latitude":      lat,
                "longitude":     lon,
                "hourly":        (
                    "temperature_2m,apparent_temperature,precipitation,"
                    "windspeed_10m,windgusts_10m,relativehumidity_2m"
                ),
                "forecast_days": 1,
                "timezone":      "auto",
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    try:
        h = data["hourly"]
        return {
            "success":      True,
            "location":     display,
            "temp_c":       round(h["temperature_2m"][-1],     1),
            "feels_like_c": round(h["apparent_temperature"][-1], 1),
            "wind_kmh":     round(h["windspeed_10m"][-1],       1),
            "gust_kmh":     round(h["windgusts_10m"][-1],       1),
            "humidity_pct": round(h["relativehumidity_2m"][-1], 1),
            "precip_mm":    round(h["precipitation"][-1],       2),
            "description":  "Open-Meteo forecast",
            "wind_dir":     "",
        }
    except Exception:
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def get_current_conditions(location: str) -> Dict[str, Any]:
    """
    Fetch current weather for any location the user mentions.

    Strategy:
      1. wttr.in  — real-time conditions, no API key, with location validation
      2. Open-Meteo — explicit geocoding fallback (handles 'Austin, TX' format)
      3. Failure  — returns success=False with a descriptive error

    location: whatever city/place the user typed, e.g. 'Denton', 'Austin, TX',
              'Sydney, Australia', 'Mumbai', etc. No hard-coding anywhere.
    """
    # 1. wttr.in (with sanity check + retry)
    result = _wttr_fetch(location)

    # 2. Open-Meteo fallback
    if not result:
        result = _openmeteo_current(location)

    # 3. Complete failure
    if not result:
        return {
            "success":  False,
            "location": location,
            "error":    f"Could not fetch weather for '{location}' from any source.",
        }

    result["summary_text"] = _build_summary(result)
    result["risk_flags"]   = _build_risk_flags(result)
    return result


# ── Formatting helpers ────────────────────────────────────────────────────────

def _build_summary(d: Dict[str, Any]) -> str:
    """Concise bullet-list for the LLM synthesis prompt."""
    lines = [
        f"• Location       : {d['location']}",
        f"• Temperature    : {d['temp_c']}°C (feels like {d['feels_like_c']}°C)",
        (
            f"• Wind           : {d['wind_kmh']} km/h sustained, "
            f"{d['gust_kmh']} km/h gusts"
            + (f" from {d['wind_dir']}" if d.get("wind_dir") else "")
        ),
        f"• Precipitation  : {d['precip_mm']} mm",
        f"• Humidity       : {d['humidity_pct']}%",
        f"• Conditions     : {d.get('description', 'N/A')}",
    ]
    flags = _build_risk_flags(d)
    if flags:
        lines.append(f"• ⚠ Risk flags   : {' | '.join(flags)}")
    else:
        lines.append("• ✓ No active wind/precip risk flags")
    return "\n".join(lines)


def _build_risk_flags(d: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    w = d.get("wind_kmh", 0)
    g = d.get("gust_kmh", 0)
    p = d.get("precip_mm", 0)
    t = d.get("temp_c", 20)

    if w > 38 or g > 38:
        flags.append(f"🏗 Crane ops risk (wind {w} km/h > 38 km/h limit)")
    if w > 30 or g > 30:
        flags.append(f"🪟 Glazing risk (wind {w} km/h > 30 km/h limit)")
    if w > 25:
        flags.append(f"🔧 Scaffolding caution (wind {w} km/h > 25 km/h)")
    if p > 2:
        flags.append(f"🌧 Waterproofing risk ({p} mm/hr precip)")
    if t < 10:
        flags.append(f"🌡 Cold concrete pour risk ({t}°C)")
    if t > 35:
        flags.append(f"🌡 Heat pour risk ({t}°C)")
    return flags
