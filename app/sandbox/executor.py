"""
Safe Python Sandbox + Open-Meteo Time Series Analysis
======================================================
Execution model:
  • Code is written to a temp file and run via subprocess with:
      - Hard timeout (SANDBOX_TIMEOUT seconds, default 30 s)
      - No stdin
      - Captured stdout/stderr
      - Working dir = isolated temp directory
  • Only the Open-Meteo public API is used (no key required)

Open-Meteo endpoints used:
  • Geocoding: https://geocoding-api.open-meteo.com/v1/search
  • Archive:   https://archive-api.open-meteo.com/v1/archive  (historical)
  • Forecast:  https://api.open-meteo.com/v1/forecast          (fallback)

Analytics computed inside the sandbox subprocess:
  • Rolling averages   – 24h and 168h (7-day) windows
  • Volatility (σ)     – daily std-dev per variable
  • Missingness        – % null per variable
  • Anomaly detection  – |z-score| > 2 threshold, with timestamps
  • Operational risk thresholds for construction sites:
      Wind  > 38 km/h  → crane operations suspended
      Wind  > 30 km/h  → glazing / facade work suspended
      Wind  > 25 km/h  → scaffolding caution
      Temp  < 10 °C    → concrete pour risk (cold)
      Temp  > 35 °C    → concrete pour risk (heat)
      Precip > 2 mm/h  → waterproofing / finishing concern
  • Structured JSON summary block appended to stdout
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import date, timedelta
from typing import Any, Dict, Optional

import requests

from app.config import SANDBOX_TIMEOUT


# ── URL constants ────────────────────────────────────────────────────────────

GEOCODE_URL  = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"

# Variables we always request (hourly)
_HOURLY_VARS = (
    "temperature_2m,apparent_temperature,precipitation,"
    "windspeed_10m,windgusts_10m,relativehumidity_2m"
)

# Archive data lags ~5-7 days; requests touching the last 7 days → forecast
_ARCHIVE_LAG_DAYS = 7


# ── Geocoding helper ─────────────────────────────────────────────────────────

def geocode(location: str) -> Optional[Dict[str, Any]]:
    """Resolve a location name to lat/lon via Open-Meteo geocoding."""
    try:
        r = requests.get(
            GEOCODE_URL,
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        data = r.json()
        if data.get("results"):
            return data["results"][0]
    except Exception:
        pass
    return None


# ── Weather fetchers ─────────────────────────────────────────────────────────

def fetch_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Fetch historical weather from Open-Meteo archive.
    If end_date falls within the archive lag window, auto-falls back to forecast.
    """
    cutoff = str(date.today() - timedelta(days=_ARCHIVE_LAG_DAYS))

    # If the requested end is too recent for the archive, use forecast endpoint
    if end_date >= cutoff:
        days_ahead = (date.fromisoformat(end_date) - date.today()).days + 1
        forecast_days = max(7, min(days_ahead + 1, 16))
        return fetch_forecast(latitude, longitude, days=forecast_days)

    params = {
        "latitude":   latitude,
        "longitude":  longitude,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly":     _HOURLY_VARS,
        "timezone":   "auto",
    }
    r = requests.get(ARCHIVE_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_forecast(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> Dict[str, Any]:
    """Fetch forecast data (up to 16 days ahead)."""
    params = {
        "latitude":      latitude,
        "longitude":     longitude,
        "hourly":        _HOURLY_VARS,
        "forecast_days": min(days, 16),
        "timezone":      "auto",
    }
    r = requests.get(FORECAST_URL, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# ── Analysis script generator ────────────────────────────────────────────────

def _build_analysis_script(weather_json: Dict, location_name: str) -> str:
    """
    Generate the Python analytics script that runs inside the sandbox subprocess.
    Produces human-readable output + a structured JSON block at the end.
    """
    data_str = json.dumps(weather_json)

    script = textwrap.dedent(f"""\
import json, math, statistics

data = json.loads({repr(data_str)})
hourly  = data.get("hourly", {{}})
times   = hourly.get("time", [])
temps   = hourly.get("temperature_2m", [])
app_t   = hourly.get("apparent_temperature", [])
precips = hourly.get("precipitation", [])
winds   = hourly.get("windspeed_10m", [])
gusts   = hourly.get("windgusts_10m", [])
humid   = hourly.get("relativehumidity_2m", [])

LOCATION = {repr(location_name)}

# ── Helpers ──────────────────────────────────────────────────────────────────

def clean(series):
    return [x for x in series if x is not None]

def missingness(series):
    if not series:
        return 100.0
    return round(sum(1 for x in series if x is None) / len(series) * 100, 2)

def rolling_avg(series, window):
    result = []
    for i in range(len(series)):
        chunk = [x for x in series[max(0, i - window + 1):i + 1] if x is not None]
        result.append(round(sum(chunk) / len(chunk), 2) if chunk else None)
    return result

def daily_volatility(series, times):
    \"\"\"Return list of (date, stddev) pairs for each day.\"\"\"
    by_day = {{}}
    for t, v in zip(times, series):
        if v is not None:
            day = t[:10]
            by_day.setdefault(day, []).append(v)
    return [
        (d, round(statistics.stdev(vs), 2) if len(vs) > 1 else 0.0)
        for d, vs in sorted(by_day.items())
    ]

def detect_anomalies(series, times, threshold=2.0):
    c = clean(series)
    if len(c) < 3:
        return []
    mean = sum(c) / len(c)
    std  = statistics.stdev(c)
    if std == 0:
        return []
    return [
        (times[i] if i < len(times) else "?", round(v, 2), round((v - mean) / std, 2))
        for i, v in enumerate(series)
        if v is not None and abs(v - mean) > threshold * std
    ]

def pct_hours_above(series, threshold):
    c = clean(series)
    if not c:
        return 0.0
    return round(sum(1 for x in c if x > threshold) / len(c) * 100, 1)

def pct_hours_below(series, threshold):
    c = clean(series)
    if not c:
        return 0.0
    return round(sum(1 for x in c if x < threshold) / len(c) * 100, 1)

total_hours  = len(temps)
date_range   = f"{{times[0] if times else '?'}} → {{times[-1] if times else '?'}}"

# ── Temperature ──────────────────────────────────────────────────────────────
ct = clean(temps)
if ct:
    t_mean   = round(sum(ct) / len(ct), 2)
    t_max    = round(max(ct), 2)
    t_min    = round(min(ct), 2)
    t_std    = round(statistics.stdev(ct), 2) if len(ct) > 1 else 0.0
    t_miss   = missingness(temps)
    t_roll24 = rolling_avg(temps, 24)
    t_roll7d = rolling_avg(temps, 168)
    t_anom   = detect_anomalies(temps, times)
    t_dvol   = daily_volatility(temps, times)
    t_cold_pct = pct_hours_below(temps, 10.0)
    t_hot_pct  = pct_hours_above(temps, 35.0)
else:
    t_mean = t_max = t_min = t_std = 0.0
    t_miss = 100.0; t_anom = []; t_dvol = []; t_roll24 = []; t_roll7d = []
    t_cold_pct = t_hot_pct = 0.0

# ── Precipitation ────────────────────────────────────────────────────────────
cp = clean(precips)
if cp:
    p_total    = round(sum(cp), 2)
    p_max_hr   = round(max(cp), 2)
    p_miss     = missingness(precips)
    rainy_hrs  = sum(1 for x in cp if x > 0.1)
    heavy_hrs  = sum(1 for x in cp if x > 2.0)   # construction waterproofing risk
else:
    p_total = p_max_hr = 0.0; p_miss = 100.0; rainy_hrs = heavy_hrs = 0

# ── Wind ─────────────────────────────────────────────────────────────────────
cw = clean(winds)
cg = clean(gusts)
if cw:
    w_mean   = round(sum(cw) / len(cw), 2)
    w_max    = round(max(cw), 2)
    w_std    = round(statistics.stdev(cw), 2) if len(cw) > 1 else 0.0
    w_miss   = missingness(winds)
    w_roll24 = rolling_avg(winds, 24)
    w_anom   = detect_anomalies(winds, times)
    w_dvol   = daily_volatility(winds, times)
    # Construction risk hours
    crane_suspended   = sum(1 for x in cw if x > 38)
    glazing_suspended = sum(1 for x in cw if x > 30)
    scaffold_caution  = sum(1 for x in cw if x > 25)
else:
    w_mean = w_max = w_std = 0.0; w_miss = 100.0
    w_anom = []; w_dvol = []; w_roll24 = []
    crane_suspended = glazing_suspended = scaffold_caution = 0

g_max = round(max(cg), 2) if cg else None

# ── Humidity ─────────────────────────────────────────────────────────────────
ch = clean(humid)
h_mean = round(sum(ch) / len(ch), 2) if ch else None

# ── Print human-readable report ──────────────────────────────────────────────
print(f"=== SiteWatch Weather Analysis: {{LOCATION}} ===")
print(f"Period         : {{date_range}}")
print(f"Total records  : {{total_hours}} hourly observations\\n")

print("── TEMPERATURE (°C) ────────────────────────────────────")
print(f"  Mean           : {{t_mean}}")
print(f"  Max / Min      : {{t_max}} / {{t_min}}")
print(f"  Std Dev (vol.) : {{t_std}}")
print(f"  24h Rolling Avg: {{t_roll24[-1] if t_roll24 else 'N/A'}} (last window)")
print(f"  7-day Rolling  : {{t_roll7d[-1] if t_roll7d else 'N/A'}} (last window)")
print(f"  Anomalies      : {{len(t_anom)}} hours > 2σ")
print(f"  Missingness    : {{t_miss}}%")
if t_anom:
    print("  Top anomalies:")
    for ts, val, z in sorted(t_anom, key=lambda x: abs(x[2]), reverse=True)[:3]:
        print(f"    {{ts}}: {{val}}°C  (z={{z}})")

print("\\n── PRECIPITATION (mm/hr) ───────────────────────────────")
print(f"  Total          : {{p_total}} mm")
print(f"  Peak hour      : {{p_max_hr}} mm")
print(f"  Rainy hours    : {{rainy_hrs}}")
print(f"  Heavy hrs (>2mm): {{heavy_hrs}}")
print(f"  Missingness    : {{p_miss}}%")

print("\\n── WIND SPEED (km/h) ───────────────────────────────────")
print(f"  Mean           : {{w_mean}}")
print(f"  Max sustained  : {{w_max}}")
print(f"  Max gust       : {{g_max if g_max is not None else 'N/A'}}")
print(f"  Std Dev (vol.) : {{w_std}}")
print(f"  24h Rolling Avg: {{w_roll24[-1] if w_roll24 else 'N/A'}} (last window)")
print(f"  Anomalies      : {{len(w_anom)}} hours > 2σ")
print(f"  Missingness    : {{w_miss}}%")
if w_anom:
    print("  Top anomalies:")
    for ts, val, z in sorted(w_anom, key=lambda x: abs(x[2]), reverse=True)[:3]:
        print(f"    {{ts}}: {{val}} km/h  (z={{z}})")

print("\\n── DAILY VOLATILITY (temp σ per day) ──────────────────")
for day, vol in t_dvol[-5:]:   # last 5 days
    bar = "█" * int(vol)
    print(f"  {{day}}: σ={{vol}}  {{bar}}")

# ── OPERATIONAL RISK THRESHOLDS (SiteWatch Construction) ─────────────────────
print("\\n── OPERATIONAL RISK THRESHOLDS ─────────────────────────")
print(f"  🏗  Crane ops suspended (>38 km/h)  : {{crane_suspended}} hrs")
print(f"  🪟  Glazing suspended   (>30 km/h)  : {{glazing_suspended}} hrs")
print(f"  🔧  Scaffolding caution (>25 km/h)  : {{scaffold_caution}} hrs")
print(f"  🌡  Concrete pour risk  (<10°C)     : {{t_cold_pct}}% of hrs")
print(f"  🌡  Concrete pour risk  (>35°C)     : {{t_hot_pct}}% of hrs")
print(f"  🌧  Waterproofing risk  (>2mm/hr)   : {{heavy_hrs}} hrs")
if h_mean is not None:
    print(f"  💧  Avg Relative Humidity           : {{h_mean}}%")

# ── Risk summary ──────────────────────────────────────────────────────────────
risk_flags = []
if crane_suspended > 0:
    risk_flags.append(f"crane suspended {{crane_suspended}} hrs")
if glazing_suspended > 0:
    risk_flags.append(f"glazing suspended {{glazing_suspended}} hrs")
if scaffold_caution > 0:
    risk_flags.append(f"scaffolding caution {{scaffold_caution}} hrs")
if t_cold_pct > 5:
    risk_flags.append(f"cold concrete pour risk {{t_cold_pct}}% hrs")
if t_hot_pct > 5:
    risk_flags.append(f"heat concrete pour risk {{t_hot_pct}}% hrs")
if heavy_hrs > 0:
    risk_flags.append(f"waterproofing concern {{heavy_hrs}} hrs")

if risk_flags:
    print("\\n⚠ RISK SUMMARY: " + " | ".join(risk_flags))
else:
    print("\\n✓ No operational risk thresholds exceeded in this period.")

# ── Structured JSON output (for LLM consumption) ────────────────────────────
summary = {{
    "location": LOCATION,
    "date_range": date_range,
    "total_hours": total_hours,
    "temperature": {{
        "mean_c": t_mean, "max_c": t_max, "min_c": t_min,
        "std_dev": t_std, "missing_pct": t_miss,
        "anomaly_count": len(t_anom),
        "cold_pour_risk_pct": t_cold_pct,
        "heat_pour_risk_pct": t_hot_pct,
    }},
    "precipitation": {{
        "total_mm": p_total, "peak_mm_per_hr": p_max_hr,
        "rainy_hours": rainy_hrs, "heavy_hours_gt2mm": heavy_hrs,
        "missing_pct": p_miss,
    }},
    "wind": {{
        "mean_kmh": w_mean, "max_kmh": w_max, "max_gust_kmh": g_max,
        "std_dev": w_std, "missing_pct": w_miss,
        "anomaly_count": len(w_anom),
        "crane_suspended_hrs": crane_suspended,
        "glazing_suspended_hrs": glazing_suspended,
        "scaffold_caution_hrs": scaffold_caution,
    }},
    "humidity": {{"mean_pct": h_mean}},
    "risk_flags": risk_flags,
    "overall_risk": ("HIGH" if len(risk_flags) >= 3
                     else "MEDIUM" if len(risk_flags) >= 1
                     else "LOW"),
}}
print("\\n__JSON_SUMMARY__")
print(json.dumps(summary))
print("__END_JSON__")
print("\\n[Analysis complete]")
""")
    return script


# ── Safe subprocess executor ─────────────────────────────────────────────────

def run_code_safely(code: str) -> Dict[str, Any]:
    """
    Execute Python code in a restricted subprocess.
    Returns: {success, stdout, stderr, timed_out}
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "analysis.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=SANDBOX_TIMEOUT,
                cwd=tmpdir,
                env={
                    "PATH": os.environ.get("PATH", ""),
                    "PYTHONPATH": "",   # isolate from host packages
                },
            )
            return {
                "success":   result.returncode == 0,
                "stdout":    result.stdout,
                "stderr":    result.stderr,
                "timed_out": False,
            }
        except subprocess.TimeoutExpired:
            return {
                "success":   False,
                "stdout":    "",
                "stderr":    f"Execution timed out after {SANDBOX_TIMEOUT}s",
                "timed_out": True,
            }
        except Exception as e:
            return {
                "success":   False,
                "stdout":    "",
                "stderr":    str(e),
                "timed_out": False,
            }


def _parse_json_summary(stdout: str) -> Optional[Dict[str, Any]]:
    """Extract the structured JSON block from the analysis output."""
    try:
        start = stdout.index("__JSON_SUMMARY__") + len("__JSON_SUMMARY__")
        end   = stdout.index("__END_JSON__")
        return json.loads(stdout[start:end].strip())
    except (ValueError, json.JSONDecodeError):
        return None


# ── Public API ───────────────────────────────────────────────────────────────

def analyze_weather(
    location: str,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """
    Full pipeline:
      1. Geocode location name → lat/lon
      2. Fetch hourly time-series from Open-Meteo (archive or forecast fallback)
      3. Generate + execute analytics script in isolated subprocess
      4. Parse structured JSON summary
      5. Return rich result dict

    Returns dict with keys:
      success, location, latitude, longitude,
      start_date, end_date, output (human text),
      summary (structured analytics dict), risk_flags, overall_risk,
      timed_out, error (on failure)
    """
    # 1. Geocode
    geo = geocode(location)
    if not geo:
        return {
            "success":  False,
            "error":    f"Could not geocode location: '{location}'",
            "location": location,
        }

    lat      = geo["latitude"]
    lon      = geo["longitude"]
    name     = geo.get("name", location)
    country  = geo.get("country", "")
    full_name = f"{name}, {country}" if country else name

    # 2. Fetch weather
    try:
        weather_data = fetch_weather(lat, lon, start_date, end_date)
    except Exception as exc:
        return {
            "success":  False,
            "error":    f"Open-Meteo API error: {exc}",
            "location": full_name,
        }

    # 3. Build + execute analytics
    script = _build_analysis_script(weather_data, full_name)
    result = run_code_safely(script)

    stdout = result["stdout"] if result["success"] else ""

    # 4. Parse JSON summary
    structured = _parse_json_summary(stdout) if result["success"] else None

    # 5. Build human-readable output (strip the JSON block for display)
    display_output = stdout
    if "__JSON_SUMMARY__" in display_output:
        display_output = display_output[:display_output.index("__JSON_SUMMARY__")].strip()

    return {
        "success":    result["success"],
        "location":   full_name,
        "latitude":   lat,
        "longitude":  lon,
        "start_date": start_date,
        "end_date":   end_date,
        "output":     display_output if result["success"] else result["stderr"],
        "summary":    structured,
        "risk_flags": structured.get("risk_flags", []) if structured else [],
        "overall_risk": structured.get("overall_risk", "UNKNOWN") if structured else "UNKNOWN",
        "timed_out":  result["timed_out"],
    }
