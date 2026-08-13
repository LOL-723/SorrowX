"""Pure utility and HTTP tools owned by the Agent tool subsystem."""

from __future__ import annotations

import ast
import operator
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


DEFAULT_TIMEZONE = "Asia/Shanghai"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_EXPRESSION_LENGTH = 200
MAX_AST_NODES = 100
MAX_ABS_NUMBER = 1_000_000_000_000
MAX_ABS_EXPONENT = 100

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_TIMEZONES = {
    "中国": "Asia/Shanghai", "中国大陆": "Asia/Shanghai", "北京": "Asia/Shanghai", "北京时间": "Asia/Shanghai", "上海": "Asia/Shanghai", "香港": "Asia/Hong_Kong", "台湾": "Asia/Taipei", "台北": "Asia/Taipei",
    "日本": "Asia/Tokyo", "东京": "Asia/Tokyo", "韩国": "Asia/Seoul", "首尔": "Asia/Seoul", "新加坡": "Asia/Singapore",
    "英国": "Europe/London", "伦敦": "Europe/London", "法国": "Europe/Paris", "巴黎": "Europe/Paris", "德国": "Europe/Berlin", "柏林": "Europe/Berlin",
    "美国东部": "America/New_York", "纽约": "America/New_York", "华盛顿": "America/New_York", "美国中部": "America/Chicago", "芝加哥": "America/Chicago", "美国山区": "America/Denver", "丹佛": "America/Denver", "美国西部": "America/Los_Angeles", "洛杉矶": "America/Los_Angeles", "旧金山": "America/Los_Angeles",
    "澳大利亚": "Australia/Sydney", "悉尼": "Australia/Sydney", "俄罗斯": "Europe/Moscow", "莫斯科": "Europe/Moscow", "印度": "Asia/Kolkata", "新德里": "Asia/Kolkata", "迪拜": "Asia/Dubai", "utc": "UTC", "gmt": "UTC",
}
_LOCATIONS: dict[str, tuple[str, float, float, str]] = {
    "沈阳": ("沈阳", 41.8057, 123.4315, "Asia/Shanghai"), "北京": ("北京", 39.9042, 116.4074, "Asia/Shanghai"),
    "上海": ("上海", 31.2304, 121.4737, "Asia/Shanghai"), "香港": ("香港", 22.3193, 114.1694, "Asia/Hong_Kong"), "台北": ("台北", 25.0330, 121.5654, "Asia/Taipei"),
    "东京": ("东京", 35.6762, 139.6503, "Asia/Tokyo"), "首尔": ("首尔", 37.5665, 126.9780, "Asia/Seoul"),
    "新加坡": ("新加坡", 1.3521, 103.8198, "Asia/Singapore"), "伦敦": ("伦敦", 51.5072, -0.1276, "Europe/London"),
    "巴黎": ("巴黎", 48.8566, 2.3522, "Europe/Paris"), "柏林": ("柏林", 52.5200, 13.4050, "Europe/Berlin"), "纽约": ("纽约", 40.7128, -74.0060, "America/New_York"), "华盛顿": ("华盛顿", 38.9072, -77.0369, "America/New_York"), "芝加哥": ("芝加哥", 41.8781, -87.6298, "America/Chicago"), "丹佛": ("丹佛", 39.7392, -104.9903, "America/Denver"),
    "洛杉矶": ("洛杉矶", 34.0522, -118.2437, "America/Los_Angeles"), "旧金山": ("旧金山", 37.7749, -122.4194, "America/Los_Angeles"), "悉尼": ("悉尼", -33.8688, 151.2093, "Australia/Sydney"),
    "莫斯科": ("莫斯科", 55.7558, 37.6173, "Europe/Moscow"), "新德里": ("新德里", 28.6139, 77.2090, "Asia/Kolkata"), "迪拜": ("迪拜", 25.2048, 55.2708, "Asia/Dubai"),
}
_LOCATION_ALIASES = {"中国": "北京", "中国大陆": "北京", "北京时间": "北京", "台湾": "台北", "日本": "东京", "韩国": "首尔", "英国": "伦敦", "法国": "巴黎", "德国": "柏林", "美国东部": "纽约", "美国中部": "芝加哥", "美国山区": "丹佛", "美国西部": "洛杉矶", "澳大利亚": "悉尼", "俄罗斯": "莫斯科", "印度": "新德里", "utc": "伦敦", "gmt": "伦敦"}
_WEATHER = {0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴", 45: "雾", 48: "雾凇", 51: "小毛毛雨", 53: "中等毛毛雨", 55: "大毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨", 71: "小雪", 73: "中雪", 75: "大雪", 80: "小阵雨", 81: "中等阵雨", 82: "强阵雨", 95: "雷暴"}


def get_current_time(timezone_name: str | None = None, location: str | None = None, **_: Any) -> dict[str, str]:
    resolved = _resolve_timezone(timezone_name, location)
    now = datetime.now(_load_timezone(resolved))
    return {"location": location or resolved, "timezone": resolved, "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"), "iso_timestamp": now.isoformat(timespec="seconds"), "utc_offset": now.strftime("%z")}


def calculate_expression(expression: str, **_: Any) -> dict[str, Any]:
    try:
        normalized = _normalize_expression(expression)
        return {"expression": normalized, "result": _normalize_number(_safe_calculate(normalized))}
    except (ArithmeticError, ValueError) as exc:
        return {"expression": expression, "error": str(exc)}


def get_today_weather(location: str | None = None, timezone_name: str | None = None, **_: Any) -> dict[str, Any]:
    name, latitude, longitude, resolved_timezone = _resolve_weather_location(location, timezone_name)
    current = get_current_time(timezone_name=resolved_timezone, location=name)
    current_hour = datetime.fromisoformat(current["iso_timestamp"]).hour
    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params={"latitude": latitude, "longitude": longitude, "hourly": "temperature_2m,weather_code", "daily": "weather_code,temperature_2m_max,temperature_2m_min,sunrise,sunset", "forecast_days": 1, "timezone": resolved_timezone}, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        return {"location": name, "timezone": resolved_timezone, "error": f"weather request failed: {exc}"}
    hourly, daily = data.get("hourly", {}), data.get("daily", {})
    forecast = [_hour(index, value, hourly, current_hour) for index, value in enumerate((hourly.get("time") or [])[:24])]
    return {"location": name, "timezone": resolved_timezone, "date": _first(daily.get("time")), "current_time": current["timestamp"], "current_hour": current_hour, "current_weather": next((item for item in forecast if item["is_current_hour"]), None), "temperature_unit": data.get("hourly_units", {}).get("temperature_2m", "°C"), "min_temperature": _first(daily.get("temperature_2m_min")), "max_temperature": _first(daily.get("temperature_2m_max")), "weather_status": _weather_text(_first(daily.get("weather_code"))), "sunrise": _minute(_first(daily.get("sunrise"))), "sunset": _minute(_first(daily.get("sunset"))), "hourly": forecast}


def _resolve_timezone(timezone_name: str | None, location: str | None) -> str:
    for value in (timezone_name, location):
        if value and value.strip():
            cleaned = value.strip()
            if cleaned in _TIMEZONES: return _TIMEZONES[cleaned]
            if cleaned.lower() in _TIMEZONES: return _TIMEZONES[cleaned.lower()]
            if "/" in cleaned or cleaned.upper() == "UTC": return cleaned
    return DEFAULT_TIMEZONE


def _load_timezone(name: str):
    try: return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return timezone(timedelta(hours=8), "CST") if name == DEFAULT_TIMEZONE else UTC


def _normalize_expression(expression: str) -> str:
    if not expression.strip(): raise ValueError("expression cannot be empty")
    if len(expression) > MAX_EXPRESSION_LENGTH: raise ValueError("expression is too long")
    candidates = [expression.strip()]
    candidate = "".join(char if char in "0123456789+-*/%(). " else " " for char in expression).strip()
    if candidate: candidates.append(candidate)
    for value in sorted(candidates, key=len, reverse=True):
        try:
            _safe_calculate(value)
            return value
        except (SyntaxError, ValueError): pass
    raise ValueError("expression contains unsupported syntax")


def _safe_calculate(expression: str) -> int | float:
    parsed = ast.parse(expression, mode="eval")
    if sum(1 for _ in ast.walk(parsed)) > MAX_AST_NODES: raise ValueError("expression is too complex")
    return _calculate(parsed.body)


def _calculate(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and not isinstance(node.value, bool) and isinstance(node.value, (int, float)):
        if abs(node.value) > MAX_ABS_NUMBER: raise ValueError("number is too large")
        return node.value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY: return _UNARY[type(node.op)](_calculate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _calculate(node.left), _calculate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABS_EXPONENT: raise ValueError("exponent is too large")
        result = _BINARY[type(node.op)](left, right)
        if abs(result) > MAX_ABS_NUMBER: raise ValueError("result is too large")
        return result
    raise ValueError("expression contains unsupported syntax")


def _resolve_weather_location(location: str | None, timezone_name: str | None) -> tuple[str, float, float, str]:
    value = (location or timezone_name or "沈阳").strip()
    value = _LOCATION_ALIASES.get(value, value)
    if value in _LOCATIONS: return _LOCATIONS[value]
    timezone_value = _resolve_timezone(timezone_name, location)
    return next((item for item in _LOCATIONS.values() if item[3] == timezone_value), _LOCATIONS["沈阳"])


def _hour(index: int, timestamp: Any, hourly: dict[str, Any], current_hour: int) -> dict[str, Any]:
    try: hour = datetime.fromisoformat(str(timestamp)).hour
    except ValueError: hour = index
    return {"hour": f"{hour:02d}:00-{hour + 1:02d}:00" if hour < 23 else "23:00-24:00", "temperature": _at(hourly.get("temperature_2m"), index), "weather_status": _weather_text(_at(hourly.get("weather_code"), index)), "is_current_hour": hour == current_hour}


def _at(values: Any, index: int) -> Any: return values[index] if isinstance(values, list) and index < len(values) else None
def _first(values: Any) -> Any: return _at(values, 0)
def _weather_text(code: Any) -> str:
    try: return _WEATHER.get(int(code), "未知")
    except (TypeError, ValueError): return "未知"
def _minute(value: Any) -> str | None:
    try: return datetime.fromisoformat(str(value)).strftime("%H:%M")
    except ValueError: return None
def _normalize_number(value: int | float) -> int | float: return int(value) if isinstance(value, float) and value.is_integer() else value
