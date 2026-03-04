import json
import time
import logging

from anthropic import Anthropic
import httpx

from core.config import settings, KNOWLEDGE_DIR
from core.knowledge import format_context

logger = logging.getLogger(__name__)

PROMPT_FILE = KNOWLEDGE_DIR / "prompt.json"

# In-memory conversation history per customer (phone -> messages)
_conversations: dict[str, list] = {}
_conversation_timestamps: dict[str, float] = {}
CONVERSATION_TTL = 3600  # 1 hour

# Customer insights tracking (in-memory, resets on deploy)
_customer_message_counts: dict[str, int] = {}  # phone_last4 -> count
_customer_names: dict[str, str] = {}  # phone_last4 -> name
_hourly_message_counts: dict[int, int] = {}  # hour (0-23) -> count

# FAQ hit rate tracking
_faq_hit_counts: dict[str, int] = {}  # question_prefix -> hit count


def _load_prompt_config() -> dict:
    """Load the evolving prompt configuration — DB first, then JSON file fallback.

    DB is the source of truth for version and evolved data (survives deploys).
    JSON file is the fallback when DB is unavailable.
    """
    # Try DB first (has the latest evolved version)
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            db_prompt = load_knowledge_from_db("prompt")
            if db_prompt:
                return db_prompt
    except Exception as e:
        logger.warning(f"DB prompt load failed, falling back to file: {e}")

    # Fallback to JSON file
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load prompt.json: {e}")
        return {}


def _build_system_prompt() -> str:
    """Build system prompt dynamically from prompt.json + knowledge context.

    This prompt evolves over time as Digital Ketu learns from Ketu's real messages.
    """
    config = _load_prompt_config()
    knowledge_context = format_context()

    identity = config.get("identity", {})
    name = identity.get("name", "Digital Ketu")
    role = identity.get("role", "AI twin of Ketu")
    core = identity.get("core_instruction", "You ARE Ketu.")

    # Base personality traits (original + evolved)
    base_traits = config.get("personality_traits", [])
    evolved_traits = config.get("evolved_traits", [])
    all_traits = base_traits + evolved_traits

    # Reply rules (original + evolved)
    base_rules = config.get("reply_rules", [])
    evolved_rules = config.get("evolved_rules", [])
    all_rules = base_rules + evolved_rules

    # Signature phrases (original + evolved)
    base_phrases = config.get("signature_phrases", [])
    evolved_phrases = config.get("evolved_phrases", [])
    all_phrases = base_phrases + evolved_phrases

    # Build prompt
    sections = []

    sections.append(f"You are {name} — {role}.\n\n{core}")

    # Personality
    if all_traits:
        trait_lines = "\n".join(f"- {t}" for t in all_traits)
        sections.append(f"## HOW KETU TALKS:\n{trait_lines}")

    # Rules
    if all_rules:
        rule_lines = "\n".join(f"{i+1}. {r}" for i, r in enumerate(all_rules))
        sections.append(f"## REPLY RULES:\n{rule_lines}")

    # Signature phrases — these are Ketu's real words, use them naturally
    if all_phrases:
        phrase_str = ", ".join(f'"{p}"' for p in all_phrases)
        sections.append(f"## KETU'S SIGNATURE PHRASES (use naturally):\n{phrase_str}")

    # Dynamic knowledge context (products, FAQs, style, etc.)
    sections.append(knowledge_context)

    sections.append("You ARE Ketu. Natural, confident, short, helpful. Jaise Ketu bolte hain waise bol.")

    return "\n\n".join(sections)


def _cleanup_old_conversations():
    now = time.time()
    expired = [
        phone for phone, ts in _conversation_timestamps.items()
        if now - ts > CONVERSATION_TTL
    ]
    for phone in expired:
        _conversations.pop(phone, None)
        _conversation_timestamps.pop(phone, None)


def get_conversation_history(phone: str) -> list:
    _cleanup_old_conversations()
    return _conversations.get(phone, [])


def generate_reply(
    message: str,
    customer_phone: str = "",
    customer_name: str = "",
    conversation_history: list | None = None,
) -> str:
    client = Anthropic(api_key=settings.anthropic_api_key)

    # Use provided history or fetch from in-memory store
    if conversation_history:
        # Map roles from wwbun format to Claude API format
        # wwbun sends "customer"/"owner" but Claude needs "user"/"assistant"
        messages = []
        for msg in conversation_history:
            role = msg.get("role", "user")
            if role in ("customer", "user"):
                role = "user"
            elif role in ("owner", "assistant", "ai"):
                role = "assistant"
            else:
                role = "user"
            messages.append({"role": role, "content": msg.get("content", "")})
    elif customer_phone:
        messages = get_conversation_history(customer_phone)
    else:
        messages = []

    # Add current message
    messages = messages + [{"role": "user", "content": message}]

    # Track customer insights
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist)
    hour = now_ist.hour
    _hourly_message_counts[hour] = _hourly_message_counts.get(hour, 0) + 1
    if customer_phone:
        key = customer_phone[-4:] if len(customer_phone) >= 4 else customer_phone
        _customer_message_counts[key] = _customer_message_counts.get(key, 0) + 1
        if customer_name:
            _customer_names[key] = customer_name

    # Add customer context if available
    system = _build_system_prompt()
    if customer_name:
        system += f"\n\nCustomer name: {customer_name}"

    # Retry with exponential backoff (max 3 attempts)
    last_error = None
    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                system=system,
                messages=messages,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )

            reply = response.content[0].text

            # Store conversation history
            if customer_phone:
                _conversations[customer_phone] = messages + [
                    {"role": "assistant", "content": reply}
                ]
                # Keep only last 20 messages
                if len(_conversations[customer_phone]) > 20:
                    _conversations[customer_phone] = _conversations[customer_phone][-20:]
                _conversation_timestamps[customer_phone] = time.time()

            return reply

        except Exception as e:
            last_error = e
            logger.warning(f"Claude API error (attempt {attempt + 1}/3): {e}")
            # Log to error tracker
            from core.error_tracker import track_error
            track_error("claude-api", str(e), {"attempt": attempt + 1})
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1s, 2s backoff

    logger.error(f"Claude API failed after 3 attempts: {last_error}")
    return "Ji sir, ek chhota sa technical issue aa gaya. Thodi der mein reply karta hun. Aap sale91.com pe check kar sakte hain."


def track_faq_hit(question: str):
    """Track that a FAQ was used in a reply."""
    key = question[:60]
    _faq_hit_counts[key] = _faq_hit_counts.get(key, 0) + 1


def get_customer_insights() -> dict:
    """Get customer message insights."""
    # Top 10 customers by message count
    sorted_customers = sorted(
        _customer_message_counts.items(), key=lambda x: x[1], reverse=True
    )[:10]
    top_customers = [
        {"phone_last4": phone, "name": _customer_names.get(phone, "Unknown"), "messages": count}
        for phone, count in sorted_customers
    ]

    # Peak hours
    peak_hours = sorted(
        _hourly_message_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]

    total_messages = sum(_customer_message_counts.values())
    unique_customers = len(_customer_message_counts)

    return {
        "total_messages": total_messages,
        "unique_customers": unique_customers,
        "top_customers": top_customers,
        "peak_hours": [{"hour": h, "count": c} for h, c in peak_hours],
        "hourly_distribution": dict(sorted(_hourly_message_counts.items())),
    }


def get_faq_hit_rates() -> list[dict]:
    """Get FAQ hit rates, sorted by most used."""
    return sorted(
        [{"question": q, "hits": c} for q, c in _faq_hit_counts.items()],
        key=lambda x: x["hits"],
        reverse=True,
    )
