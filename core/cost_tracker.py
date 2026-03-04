"""Cost tracking for Digital Ketu API usage.

Tracks token usage and costs for all Claude API calls (replies, learning, analysis).
Persists to PostgreSQL via kv_store — survives deploys.

Pricing (per million tokens, as of March 2026):
- Claude Haiku 4.5: $0.80 input, $4.00 output
- Claude Sonnet 4:  $3.00 input, $15.00 output
- OpenAI Whisper:   $0.006 per minute
"""

import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Pricing per million tokens (USD)
MODEL_PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    # Fallback for unknown models
    "default": {"input": 1.00, "output": 5.00},
}

WHISPER_COST_PER_MINUTE = 0.006

# In-memory daily cost accumulator (DB-persisted)
_cost_state: dict = {}
_cost_state_loaded = False


def _load_cost_state():
    """Load cost state from DB on first access."""
    global _cost_state, _cost_state_loaded
    if _cost_state_loaded:
        return
    _cost_state_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if not is_db_available():
            return
        data = kv_get("cost_tracker")
        if data and isinstance(data, dict):
            _cost_state.update(data)
            logger.info(f"[Cost] Loaded from DB: {len(_cost_state.get('daily', {}))} days tracked")
    except Exception as e:
        logger.warning(f"[Cost] DB load failed: {e}")


def _save_cost_state():
    """Persist cost state to DB."""
    try:
        from core.database import is_db_available, kv_set
        if not is_db_available():
            return
        kv_set("cost_tracker", _cost_state)
    except Exception as e:
        logger.warning(f"[Cost] DB save failed: {e}")


def _get_model_pricing(model: str) -> dict:
    """Get pricing for a model."""
    return MODEL_PRICING.get(model, MODEL_PRICING["default"])


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD for a single API call."""
    pricing = _get_model_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def track_api_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    source: str = "unknown",
    customer_phone: str = "",
):
    """Track cost of a single Claude API call.

    Called after every API call to accumulate daily costs.
    """
    _load_cost_state()

    cost_usd = _calculate_cost(model, input_tokens, output_tokens)
    today = datetime.now(IST).strftime("%Y-%m-%d")

    # Initialize daily structure
    if "daily" not in _cost_state:
        _cost_state["daily"] = {}
    if today not in _cost_state["daily"]:
        _cost_state["daily"][today] = {
            "total_cost_usd": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_calls": 0,
            "by_source": {},
            "by_model": {},
        }

    day = _cost_state["daily"][today]
    day["total_cost_usd"] = round(day["total_cost_usd"] + cost_usd, 6)
    day["total_input_tokens"] += input_tokens
    day["total_output_tokens"] += output_tokens
    day["total_calls"] += 1

    # Track by source
    if source not in day["by_source"]:
        day["by_source"][source] = {"calls": 0, "cost_usd": 0, "input_tokens": 0, "output_tokens": 0}
    src = day["by_source"][source]
    src["calls"] += 1
    src["cost_usd"] = round(src["cost_usd"] + cost_usd, 6)
    src["input_tokens"] += input_tokens
    src["output_tokens"] += output_tokens

    # Track by model
    if model not in day["by_model"]:
        day["by_model"][model] = {"calls": 0, "cost_usd": 0}
    mdl = day["by_model"][model]
    mdl["calls"] += 1
    mdl["cost_usd"] = round(mdl["cost_usd"] + cost_usd, 6)

    # Keep only last 90 days of data
    dates = sorted(_cost_state["daily"].keys())
    if len(dates) > 90:
        for old_date in dates[:-90]:
            del _cost_state["daily"][old_date]

    _save_cost_state()


def track_whisper_cost(duration_seconds: float, source: str = "voice-note"):
    """Track Whisper transcription cost."""
    _load_cost_state()

    duration_minutes = duration_seconds / 60
    cost_usd = round(duration_minutes * WHISPER_COST_PER_MINUTE, 6)
    today = datetime.now(IST).strftime("%Y-%m-%d")

    if "daily" not in _cost_state:
        _cost_state["daily"] = {}
    if today not in _cost_state["daily"]:
        _cost_state["daily"][today] = {
            "total_cost_usd": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_calls": 0,
            "by_source": {},
            "by_model": {},
        }

    day = _cost_state["daily"][today]
    day["total_cost_usd"] = round(day["total_cost_usd"] + cost_usd, 6)
    day["total_calls"] += 1

    if source not in day["by_source"]:
        day["by_source"][source] = {"calls": 0, "cost_usd": 0, "input_tokens": 0, "output_tokens": 0}
    day["by_source"][source]["calls"] += 1
    day["by_source"][source]["cost_usd"] = round(day["by_source"][source]["cost_usd"] + cost_usd, 6)

    if "whisper-1" not in day["by_model"]:
        day["by_model"]["whisper-1"] = {"calls": 0, "cost_usd": 0}
    day["by_model"]["whisper-1"]["calls"] += 1
    day["by_model"]["whisper-1"]["cost_usd"] = round(day["by_model"]["whisper-1"]["cost_usd"] + cost_usd, 6)

    _save_cost_state()


def get_cost_summary() -> dict:
    """Get cost summary for dashboard."""
    _load_cost_state()

    today = datetime.now(IST).strftime("%Y-%m-%d")
    daily = _cost_state.get("daily", {})

    # Today's costs
    today_data = daily.get(today, {
        "total_cost_usd": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "total_calls": 0,
        "by_source": {},
        "by_model": {},
    })

    # Last 7 days
    now = datetime.now(IST)
    week_cost = 0
    week_calls = 0
    daily_breakdown = []
    for i in range(7):
        d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        day_data = daily.get(d, {})
        cost = day_data.get("total_cost_usd", 0)
        calls = day_data.get("total_calls", 0)
        week_cost += cost
        week_calls += calls
        daily_breakdown.append({
            "date": d,
            "cost_usd": round(cost, 4),
            "calls": calls,
        })

    # Last 30 days
    month_cost = 0
    month_calls = 0
    for i in range(30):
        d = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        day_data = daily.get(d, {})
        month_cost += day_data.get("total_cost_usd", 0)
        month_calls += day_data.get("total_calls", 0)

    # Cost per reply estimate
    reply_calls = today_data.get("by_source", {}).get("whatsapp-reply", {}).get("calls", 0)
    reply_cost = today_data.get("by_source", {}).get("whatsapp-reply", {}).get("cost_usd", 0)
    avg_cost_per_reply = round(reply_cost / reply_calls, 4) if reply_calls > 0 else 0

    # INR conversion (approximate)
    usd_to_inr = 83.5

    return {
        "today": {
            "cost_usd": round(today_data["total_cost_usd"], 4),
            "cost_inr": round(today_data["total_cost_usd"] * usd_to_inr, 2),
            "total_calls": today_data["total_calls"],
            "input_tokens": today_data["total_input_tokens"],
            "output_tokens": today_data["total_output_tokens"],
            "by_source": today_data.get("by_source", {}),
            "by_model": today_data.get("by_model", {}),
        },
        "week": {
            "cost_usd": round(week_cost, 4),
            "cost_inr": round(week_cost * usd_to_inr, 2),
            "total_calls": week_calls,
            "daily_breakdown": daily_breakdown,
        },
        "month": {
            "cost_usd": round(month_cost, 4),
            "cost_inr": round(month_cost * usd_to_inr, 2),
            "total_calls": month_calls,
        },
        "per_reply": {
            "avg_cost_usd": avg_cost_per_reply,
            "avg_cost_inr": round(avg_cost_per_reply * usd_to_inr, 4),
            "replies_today": reply_calls,
        },
        "pricing_info": {
            "haiku_input_per_mtok": "$0.80",
            "haiku_output_per_mtok": "$4.00",
            "sonnet_input_per_mtok": "$3.00",
            "sonnet_output_per_mtok": "$15.00",
            "whisper_per_minute": "$0.006",
            "usd_to_inr_rate": usd_to_inr,
        },
    }
