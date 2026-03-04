"""Persistent Customer Memory for Digital Ketu.

Remembers every customer's interests, product preferences, interaction history,
and buying stage. This context is injected into every reply so Digital Ketu
can have personalized conversations — like a real shop owner who remembers
his regular customers.

Storage: PostgreSQL (primary) + in-memory cache (fast access).
"""

import json
import logging
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# In-memory cache of customer profiles
_profiles: dict[str, dict] = {}
_profiles_loaded = False

# Buying stages
STAGE_NEW = "new"               # First-time visitor
STAGE_INQUIRY = "inquiry"       # Asked about products/prices
STAGE_INTERESTED = "interested" # Showed buying intent (asked MOQ, bulk, delivery)
STAGE_NEGOTIATING = "negotiating"  # Discussing price/quantity
STAGE_READY = "ready"           # Ready to buy (asked payment/website link)
STAGE_BOUGHT = "bought"         # Ordered (mentioned order, payment done)
STAGE_REPEAT = "repeat"         # Has ordered before


def _init_customer_table():
    """Create customer_profiles table if it doesn't exist."""
    try:
        from core.database import is_db_available, _execute
        if not is_db_available():
            return
        _execute("""
            CREATE TABLE IF NOT EXISTS customer_profiles (
                phone TEXT PRIMARY KEY,
                name TEXT DEFAULT '',
                data JSONB NOT NULL DEFAULT '{}'::jsonb,
                first_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_seen TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_customer_last_seen
                ON customer_profiles(last_seen DESC);
        """)
        logger.info("[CustomerMemory] Table ready")
    except Exception as e:
        logger.warning(f"[CustomerMemory] Table creation failed: {e}")


def _load_profile_from_db(phone: str) -> dict | None:
    """Load a single customer profile from DB."""
    try:
        from core.database import is_db_available, _execute
        if not is_db_available():
            return None
        rows = _execute(
            "SELECT data FROM customer_profiles WHERE phone = %s",
            (phone,), fetch=True,
        )
        if rows and rows[0]:
            return rows[0]["data"]
    except Exception as e:
        logger.debug(f"[CustomerMemory] DB load failed for {phone}: {e}")
    return None


def _save_profile_to_db(phone: str, name: str, profile: dict):
    """Save customer profile to DB."""
    try:
        from core.database import is_db_available, _execute
        if not is_db_available():
            return
        now = datetime.now(IST)
        _execute(
            """INSERT INTO customer_profiles (phone, name, data, first_seen, last_seen, updated_at)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (phone) DO UPDATE
               SET name = COALESCE(NULLIF(%s, ''), customer_profiles.name),
                   data = %s, last_seen = %s, updated_at = %s""",
            (phone, name, json.dumps(profile, ensure_ascii=False),
             now, now, now,
             name, json.dumps(profile, ensure_ascii=False), now, now),
        )
    except Exception as e:
        logger.warning(f"[CustomerMemory] DB save failed for {phone}: {e}")


def _new_profile() -> dict:
    """Create a blank customer profile."""
    return {
        "stage": STAGE_NEW,
        "interests": [],          # Products they asked about
        "preferred_gsm": [],      # GSM weights they showed interest in
        "preferred_category": [], # oversized, round-neck, hoodie, etc.
        "price_sensitivity": "",  # "budget", "mid", "premium", or ""
        "language": "",           # Customer's preferred language
        "total_messages": 0,
        "last_topics": [],        # Last 5 conversation topics
        "notable_details": [],    # Key facts (e.g. "runs printing business", "based in Jaipur")
        "objections_raised": [],  # Price objections, quality concerns
        "follow_up_sent": False,  # Whether we already sent a follow-up
        "follow_up_at": "",       # When follow-up was sent
    }


def get_profile(phone: str) -> dict:
    """Get or create a customer profile."""
    if phone in _profiles:
        return _profiles[phone]

    # Try DB
    db_profile = _load_profile_from_db(phone)
    if db_profile:
        _profiles[phone] = db_profile
        return db_profile

    # New customer
    profile = _new_profile()
    _profiles[phone] = profile
    return profile


def update_profile(
    phone: str,
    name: str = "",
    message: str = "",
    ai_reply: str = "",
):
    """Update customer profile based on latest interaction.

    Analyzes the message to detect:
    - Product interests (which products they're asking about)
    - Buying stage progression
    - Price sensitivity
    - Language preference
    - Notable details
    """
    profile = get_profile(phone)
    profile["total_messages"] = profile.get("total_messages", 0) + 1

    if not message:
        _profiles[phone] = profile
        _save_profile_to_db(phone, name, profile)
        return profile

    msg_lower = message.lower()

    # Detect product interests
    product_keywords = {
        "oversized": ["oversized", "oversize", "os", "drop shoulder"],
        "round_neck": ["round neck", "roundneck", "rn", "round-neck"],
        "polo": ["polo"],
        "hoodie": ["hoodie", "hoodies"],
        "sweatshirt": ["sweatshirt", "sweat shirt"],
        "varsity_jacket": ["varsity", "jacket"],
        "shorts": ["shorts", "short"],
        "kids": ["kids", "children", "bachche"],
        "zip_hoodie": ["zip hoodie", "zipper hoodie", "zip-hoodie"],
    }
    for product, keywords in product_keywords.items():
        if any(kw in msg_lower for kw in keywords):
            if product not in profile.get("interests", []):
                profile.setdefault("interests", []).append(product)

    # Detect GSM preference
    import re
    gsm_matches = re.findall(r'(\d{3})\s*(?:gsm|GSM)', message)
    for gsm in gsm_matches:
        if gsm not in profile.get("preferred_gsm", []):
            profile.setdefault("preferred_gsm", []).append(gsm)

    # Detect buying stage progression
    current_stage = profile.get("stage", STAGE_NEW)

    buying_intent_words = {
        "order", "khareed", "kharidna", "buy", "purchase", "lena hai",
        "chahiye", "bhej do", "send", "ship", "payment", "pay",
        "upi", "bank transfer", "gpay", "paytm",
    }
    bulk_words = {"bulk", "wholesale", "500", "1000", "quantity", "moq", "lot"}
    price_words = {"price", "rate", "cost", "kitna", "kitne", "kya rate", "kya price", "mehnga", "sasta"}
    ready_words = {"website", "sale91", "order karta", "payment kar", "link", "checkout"}

    if any(w in msg_lower for w in ready_words):
        profile["stage"] = STAGE_READY
    elif any(w in msg_lower for w in buying_intent_words):
        if current_stage in (STAGE_NEW, STAGE_INQUIRY):
            profile["stage"] = STAGE_INTERESTED
    elif any(w in msg_lower for w in bulk_words):
        if current_stage in (STAGE_NEW, STAGE_INQUIRY):
            profile["stage"] = STAGE_INTERESTED
    elif any(w in msg_lower for w in price_words):
        if current_stage == STAGE_NEW:
            profile["stage"] = STAGE_INQUIRY

    # Detect price sensitivity
    if any(w in msg_lower for w in ["sasta", "cheap", "budget", "kam price", "lowest", "sabse sasta"]):
        profile["price_sensitivity"] = "budget"
    elif any(w in msg_lower for w in ["premium", "best quality", "heavy", "430", "240"]):
        profile["price_sensitivity"] = "premium"

    # Detect language preference
    hindi_chars = sum(1 for c in message if '\u0900' <= c <= '\u097F')
    tamil_chars = sum(1 for c in message if '\u0B80' <= c <= '\u0BFF')
    telugu_chars = sum(1 for c in message if '\u0C00' <= c <= '\u0C7F')

    if tamil_chars > 2:
        profile["language"] = "tamil"
    elif telugu_chars > 2:
        profile["language"] = "telugu"
    elif hindi_chars > 5:
        profile["language"] = "hindi"
    elif all(ord(c) < 128 or c in ' \n\t' for c in message):
        profile["language"] = "english"
    else:
        profile["language"] = profile.get("language", "") or "hinglish"

    # Track last topics (keep last 5)
    topics = profile.get("last_topics", [])
    if len(message) > 10:
        topics.append(message[:80])
        profile["last_topics"] = topics[-5:]

    # Save
    _profiles[phone] = profile
    _save_profile_to_db(phone, name, profile)
    return profile


def format_customer_context(phone: str) -> str:
    """Format customer profile as context string for system prompt.

    Returns empty string for new customers with no history.
    """
    profile = get_profile(phone)
    if profile.get("total_messages", 0) < 2 and profile.get("stage") == STAGE_NEW:
        return ""

    parts = []
    parts.append("## CUSTOMER MEMORY (you remember this customer):")

    stage = profile.get("stage", STAGE_NEW)
    stage_labels = {
        STAGE_NEW: "Naya customer hai",
        STAGE_INQUIRY: "Pehle puchtaach ki hai",
        STAGE_INTERESTED: "Kharidne mein interested hai",
        STAGE_NEGOTIATING: "Price negotiate kar raha hai",
        STAGE_READY: "Order karne ko ready hai",
        STAGE_BOUGHT: "Pehle order kar chuka hai",
        STAGE_REPEAT: "Repeat customer hai — VIP treatment",
    }
    parts.append(f"- Status: {stage_labels.get(stage, stage)}")
    parts.append(f"- Total messages: {profile.get('total_messages', 0)}")

    interests = profile.get("interests", [])
    if interests:
        parts.append(f"- Products mein interest: {', '.join(interests)}")

    gsm_prefs = profile.get("preferred_gsm", [])
    if gsm_prefs:
        parts.append(f"- GSM preference: {', '.join(gsm_prefs)} GSM")

    sensitivity = profile.get("price_sensitivity", "")
    if sensitivity == "budget":
        parts.append("- Budget-conscious hai — sasta option pehle bata")
    elif sensitivity == "premium":
        parts.append("- Premium quality chahiye — heavy GSM recommend kar")

    language = profile.get("language", "")
    if language and language not in ("hinglish", ""):
        parts.append(f"- Language preference: {language} (isi mein reply kar)")

    topics = profile.get("last_topics", [])
    if topics:
        parts.append(f"- Last baat: {topics[-1]}")

    notable = profile.get("notable_details", [])
    if notable:
        for detail in notable[-3:]:
            parts.append(f"- Note: {detail}")

    return "\n".join(parts)


def get_interested_customers(hours: int = 24) -> list[dict]:
    """Get customers who showed buying interest in last N hours but haven't been followed up.

    Used by follow-up intelligence to send gentle reminders.
    """
    try:
        from core.database import is_db_available, _execute
        if not is_db_available():
            return []

        cutoff = datetime.now(IST) - timedelta(hours=hours)
        rows = _execute(
            """SELECT phone, name, data, last_seen
               FROM customer_profiles
               WHERE last_seen >= %s
               ORDER BY last_seen DESC""",
            (cutoff,), fetch=True,
        )
        if not rows:
            return []

        interested = []
        for row in rows:
            data = row["data"] if isinstance(row["data"], dict) else {}
            stage = data.get("stage", STAGE_NEW)
            already_followed = data.get("follow_up_sent", False)

            # Only interested/negotiating/ready customers who haven't been followed up
            if stage in (STAGE_INTERESTED, STAGE_NEGOTIATING, STAGE_READY) and not already_followed:
                interested.append({
                    "phone": row["phone"],
                    "name": row["name"] or "",
                    "stage": stage,
                    "interests": data.get("interests", []),
                    "last_seen": row["last_seen"].isoformat() if hasattr(row["last_seen"], "isoformat") else str(row["last_seen"]),
                    "total_messages": data.get("total_messages", 0),
                })
        return interested
    except Exception as e:
        logger.warning(f"[CustomerMemory] Failed to get interested customers: {e}")
        return []


def mark_follow_up_sent(phone: str):
    """Mark that a follow-up has been sent to this customer."""
    profile = get_profile(phone)
    profile["follow_up_sent"] = True
    profile["follow_up_at"] = datetime.now(IST).isoformat()
    _profiles[phone] = profile
    _save_profile_to_db(phone, "", profile)


def reset_follow_up_flag(phone: str):
    """Reset follow-up flag when customer messages again (new interaction cycle)."""
    profile = get_profile(phone)
    if profile.get("follow_up_sent"):
        profile["follow_up_sent"] = False
        profile["follow_up_at"] = ""
        _profiles[phone] = profile
        _save_profile_to_db(phone, "", profile)


def get_all_profiles_summary() -> dict:
    """Get summary of all customer profiles for dashboard."""
    try:
        from core.database import is_db_available, _execute
        if not is_db_available():
            return {"total": len(_profiles), "by_stage": {}}

        rows = _execute(
            "SELECT data FROM customer_profiles", fetch=True,
        )
        if not rows:
            return {"total": 0, "by_stage": {}}

        by_stage: dict[str, int] = {}
        for row in rows:
            data = row["data"] if isinstance(row["data"], dict) else {}
            stage = data.get("stage", STAGE_NEW)
            by_stage[stage] = by_stage.get(stage, 0) + 1

        return {
            "total": len(rows),
            "by_stage": by_stage,
        }
    except Exception:
        return {"total": len(_profiles), "by_stage": {}}
