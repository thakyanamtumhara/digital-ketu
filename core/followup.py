"""Follow-up Intelligence for Digital Ketu.

Sends ONE gentle reminder to customers who showed buying interest
but didn't complete their order within 24 hours.

Rules:
- Only runs when auto_reply AND followup are both enabled
- Maximum ONE follow-up per customer per interaction cycle
- Gentle, helpful tone — not pushy ("Sir, order hua? Koi difficulty toh nahi?")
- No "sale" language, no discounts, no pressure
- Resets when customer messages again (new cycle)
- Checks within 24-hour window only
"""

import logging
from datetime import datetime, timezone, timedelta

from core.customer_memory import (
    get_interested_customers,
    mark_follow_up_sent,
    get_profile,
)
from core.config import settings

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# Follow-up message templates — genuine, helpful, not pushy
_FOLLOWUP_TEMPLATES = [
    "Ji {name_or_sir}, order hua? Koi question ho toh bata do, help kar deta hun.",
    "{name_or_sir}, website pe koi difficulty aa rahi hai kya? Bata do, solve kar dete hain.",
    "Ji {name_or_sir}, kuch aur puchna ho toh bata do. Happy to help!",
    "Hello {name_or_sir}, sab theek? Agar order mein koi help chahiye toh batao.",
]


def _get_followup_message(customer_name: str, interests: list) -> str:
    """Generate a natural follow-up message based on customer context."""
    name_or_sir = customer_name if customer_name else "sir"

    # Rotate templates based on simple hash
    import hashlib
    hash_val = int(hashlib.md5(name_or_sir.encode()).hexdigest()[:8], 16)
    template = _FOLLOWUP_TEMPLATES[hash_val % len(_FOLLOWUP_TEMPLATES)]

    return template.format(name_or_sir=name_or_sir)


def get_pending_followups() -> list[dict]:
    """Get list of customers who need a follow-up.

    Returns customers who:
    - Showed buying interest (stage: interested/negotiating/ready)
    - Last messaged within 24 hours
    - Haven't been followed up yet
    """
    if not settings.auto_reply_enabled:
        return []

    # Check followup_enabled (defaults to True when auto_reply is on)
    if not getattr(settings, 'followup_enabled', True):
        return []

    customers = get_interested_customers(hours=24)

    followups = []
    for customer in customers:
        phone = customer["phone"]
        name = customer.get("name", "")
        interests = customer.get("interests", [])

        msg = _get_followup_message(name, interests)
        followups.append({
            "phone": phone,
            "name": name,
            "stage": customer["stage"],
            "interests": interests,
            "message": msg,
            "last_seen": customer["last_seen"],
        })

    return followups


def execute_followup(phone: str) -> dict:
    """Execute a follow-up for a specific customer.

    Marks the customer as followed-up so they don't get another one.
    Returns the follow-up message to send.

    The actual sending happens via wwbun API (caller's responsibility).
    """
    profile = get_profile(phone)
    name = ""  # We'll get it from the profile's notable details or leave empty

    # Don't follow up if already done
    if profile.get("follow_up_sent"):
        return {"status": "already_sent", "phone": phone}

    interests = profile.get("interests", [])
    message = _get_followup_message(name, interests)

    # Mark as followed up
    mark_follow_up_sent(phone)

    logger.info(f"[Followup] Sent to {phone[-4:]}: {message[:50]}")
    return {
        "status": "sent",
        "phone": phone,
        "message": message,
    }


def get_followup_stats() -> dict:
    """Get follow-up intelligence stats for dashboard."""
    pending = get_pending_followups()
    return {
        "followup_enabled": getattr(settings, 'followup_enabled', True) and settings.auto_reply_enabled,
        "pending_followups": len(pending),
        "customers": pending[:10],  # Top 10
    }
