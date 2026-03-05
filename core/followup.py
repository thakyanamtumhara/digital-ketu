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
    get_repeat_customers,
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


# Follow-up templates for REPEAT buyers — lighter, friendly, no hand-holding
_REPEAT_FOLLOWUP_TEMPLATES = [
    "{name_or_sir}, phir se order karna ho toh website se kar lo. Koi issue ho toh bata dena!",
    "Hello {name_or_sir}, kuch chahiye toh website pe check karo — sale91.com. Koi problem aaye toh batao.",
    "{name_or_sir}, naya stock aa gaya hai. Zaroorat ho toh order kar do, koi dikkat ho toh bolo.",
    "Ji {name_or_sir}, kaise hain? Kuch chahiye toh batao, website se order kar sakte hain directly.",
]

# Templates for buyers returning after 30+ days — warm re-engagement
_RETURNING_BUYER_TEMPLATES = [
    "{name_or_sir}, bahut din ho gaye! Naya collection aa gaya hai. Dekho ek baar: sale91.com",
    "Hello {name_or_sir}, kaafi time ho gaya. Naye products aaye hain, zaroorat ho toh batao!",
    "{name_or_sir}, kaise hain? Kaafi din se baat nahi hui. Kuch chahiye toh batao, help kar deta hun.",
]


def _get_followup_message(customer_name: str, interests: list) -> str:
    """Generate a natural follow-up message based on customer context."""
    name_or_sir = customer_name if customer_name else "sir"

    # Rotate templates based on simple hash
    import hashlib
    hash_val = int(hashlib.md5(name_or_sir.encode()).hexdigest()[:8], 16)
    template = _FOLLOWUP_TEMPLATES[hash_val % len(_FOLLOWUP_TEMPLATES)]

    return template.format(name_or_sir=name_or_sir)


def _get_repeat_followup_message(customer_name: str, days_since: int) -> str:
    """Generate a lighter follow-up for repeat/returning buyers."""
    name_or_sir = customer_name if customer_name else "sir"

    # 30+ days → warm re-engagement, otherwise lighter regular follow-up
    templates = _RETURNING_BUYER_TEMPLATES if days_since >= 30 else _REPEAT_FOLLOWUP_TEMPLATES

    import hashlib
    hash_val = int(hashlib.md5(name_or_sir.encode()).hexdigest()[:8], 16)
    template = templates[hash_val % len(templates)]

    return template.format(name_or_sir=name_or_sir)


def get_pending_followups() -> list[dict]:
    """Get list of customers who need a follow-up.

    Returns:
    - Interested/negotiating/ready customers (24h window) → standard follow-up
    - Repeat/bought customers (30 day window) → lighter follow-up
    """
    if not settings.auto_reply_enabled:
        return []

    # Check followup_enabled (defaults to True when auto_reply is on)
    if not getattr(settings, 'followup_enabled', True):
        return []

    followups = []

    # Standard follow-ups for interested customers
    customers = get_interested_customers(hours=24)
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
            "followup_type": "standard",
        })

    # Lighter follow-ups for repeat/bought customers
    repeat_customers = get_repeat_customers(days=30)
    for customer in repeat_customers:
        phone = customer["phone"]
        name = customer.get("name", "")
        days_since = customer.get("days_since_last_purchase", 0)

        msg = _get_repeat_followup_message(name, days_since)
        followups.append({
            "phone": phone,
            "name": name,
            "stage": customer["stage"],
            "purchase_count": customer.get("purchase_count", 1),
            "days_since_last_purchase": days_since,
            "message": msg,
            "last_seen": customer["last_seen"],
            "followup_type": "repeat_buyer",
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
