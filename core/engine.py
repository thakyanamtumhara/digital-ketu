import json
import time
import logging

from anthropic import Anthropic
import httpx

from core.config import settings, KNOWLEDGE_DIR
from core.knowledge import format_context
from core.cost_tracker import track_api_cost
from core.customer_memory import (
    get_profile, update_profile, format_customer_context, reset_follow_up_flag,
)
from core.escalation import detect_escalation, format_escalation_notice, LEVEL_ESCALATE

logger = logging.getLogger(__name__)

PROMPT_FILE = KNOWLEDGE_DIR / "prompt.json"

# In-memory conversation history per customer (phone -> messages)
_conversations: dict[str, list] = {}
_conversation_timestamps: dict[str, float] = {}
CONVERSATION_TTL = 3600  # 1 hour

# Customer insights tracking (DB-persisted, survives deploys)
_customer_message_counts: dict[str, int] = {}  # phone_last4 -> count
_customer_names: dict[str, str] = {}  # phone_last4 -> name
_hourly_message_counts: dict[int, int] = {}  # hour (0-23) -> count
_insights_loaded = False

# FAQ hit rate tracking
_faq_hit_counts: dict[str, int] = {}  # question_prefix -> hit count
_faq_hits_loaded = False

# Last escalation result (per customer, for main.py to check)
_last_escalation: dict[str, dict] = {}  # phone -> escalation result


def _load_customer_insights_from_db():
    """Load customer insights from DB on first access (survives deploys)."""
    global _customer_message_counts, _customer_names, _hourly_message_counts, _insights_loaded
    if _insights_loaded:
        return
    _insights_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if not is_db_available():
            return
        data = kv_get("customer_insights")
        if data:
            _customer_message_counts.update(data.get("message_counts", {}))
            _customer_names.update(data.get("names", {}))
            # DB stores hour keys as strings, convert back to int
            for h, c in data.get("hourly", {}).items():
                _hourly_message_counts[int(h)] = _hourly_message_counts.get(int(h), 0) + c
            logger.info(f"[Insights] Loaded from DB: {len(_customer_message_counts)} customers")
    except Exception as e:
        logger.warning(f"[Insights] DB load failed: {e}")


def _save_customer_insights_to_db():
    """Persist current customer insights to DB."""
    try:
        from core.database import is_db_available, kv_set
        if not is_db_available():
            return
        kv_set("customer_insights", {
            "message_counts": _customer_message_counts,
            "names": _customer_names,
            "hourly": _hourly_message_counts,
        })
    except Exception as e:
        logger.warning(f"[Insights] DB save failed: {e}")


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


def _build_system_prompt(
    customer_phone: str = "",
    escalation_modifier: str = "",
) -> str:
    """Build system prompt dynamically from prompt.json + knowledge context.

    This prompt evolves over time as Digital Ketu learns from Ketu's real messages.
    Includes customer memory context and escalation handling when applicable.
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

    # Smart Suggestions rule — never push, only suggest when asked
    sections.append(
        "## SUGGESTION RULE:\n"
        "- KABHI unsolicited product push mat kar. Customer jo maang raha hai wohi de.\n"
        "- Agar customer 180 GSM regular fit maang raha hai, toh oversize mat suggest kar.\n"
        "- Sirf TABHI suggest kar jab customer KHUD puche: 'kya recommend karoge?', 'best kaun sa hai?', 'suggest karo'\n"
        "- Customer ki choice respect kar — unko freedom de kuch bhi kharidne ki"
    )

    # Objection Handling rule
    sections.append(
        "## OBJECTION HANDLING (jab customer price ya quality challenge kare):\n"
        "- 'Mehnga hai' → 'Bhai, factory direct rate hai, koi middleman nahi. Plus biowash + no shrinkage guarantee — ye quality iss price mein aur kahi nahi milegi.'\n"
        "- 'Competitor sasta de raha' → 'Sir, quality compare karo — 100% cotton, biowash, ready stock, dispatch within minutes. Sasta mein ye service nahi milegi.'\n"
        "- 'Discount do' → Bulk rate bata, website pe Rs 2/pc extra off mention kar. Fake discount mat de.\n"
        "- 'Quality kaisi hai?' → Confident bol — 'No shrinkage, no color bleeding, guaranteed. Tiruppur factory se direct.'\n"
        "- KABHI defensive mat ho. Confident aur genuine reh. Facts bata, argue mat kar."
    )

    # Chat pattern categorization
    sections.append(
        "## CHAT PATTERNS (situation ke hisaab se reply style):\n"
        "- INQUIRY: Customer product/price puch raha hai → Direct info de, short, with rate\n"
        "- COMPARISON: Customer compare kar raha hai → Fact-based difference bata, push mat kar\n"
        "- OBJECTION: Customer price/quality challenge kar raha → USP bata, confident reh\n"
        "- COMPLAINT: Customer naraz hai/issue hai → Empathetic ho, Ketu sir ko connect kar\n"
        "- CLOSING: Customer ready hai order karne ko → Website link de, payment info de, smooth karo\n"
        "- ACKNOWLEDGMENT: Customer ne 'ok', 'thanks' bola → Reply mat kar (conversation ender)\n"
        "- GREETING: Customer ne 'hi' bola → Warm welcome + kaise help karu"
    )

    # Signature phrases — these are Ketu's real words, use them naturally
    if all_phrases:
        phrase_str = ", ".join(f'"{p}"' for p in all_phrases)
        sections.append(f"## KETU'S SIGNATURE PHRASES (use naturally):\n{phrase_str}")

    # Dynamic knowledge context (products, FAQs, style, etc.)
    sections.append(knowledge_context)

    # Customer memory context (if returning customer)
    if customer_phone:
        customer_context = format_customer_context(customer_phone)
        if customer_context:
            sections.append(customer_context)

    # Escalation modifier (if complaint/anger detected)
    if escalation_modifier:
        sections.append(escalation_modifier)

    sections.append("CRITICAL: Reply CHHOTA rakh — 1-3 lines MAX. Ketu WhatsApp pe lamba reply nahi deta. Seedha point pe aa, faaltu explanation mat de. You ARE Ketu. Natural, confident, short, helpful.\n\nIMPORTANT: KABHI fake promise mat karo. 'Ek min check karta hun', 'abhi dekhta hun' — ye mat bol kyunki tu actually check nahi kar sakta. Agar kuch nahi pata (stock, order status, custom quote, delivery date) toh honest bol: 'Bhai ye Ketu sir khud batayenge, thodi der mein reply aayega.' Robot mat ban, genuine human ban.\n\nSTRICT RULE: Customer already WhatsApp pe baat kar raha hai — 'WhatsApp karo' ya WhatsApp number KABHI mat de. Ye same number hai jis pe baat ho rahi hai, faaltu lagta hai. Website link (sale91.com) bhi har reply mein mat daal — ek conversation mein ek baar mention karna kaafi hai. Agar pehle de chuka hai toh dubara mat de. Natural baat kar, har message mein same line repeat karke robot mat ban.")

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


# Cached ender patterns (loaded once from knowledge base + hardcoded)
_ender_patterns: set | None = None
_non_ender_patterns: set | None = None


def _load_ender_patterns() -> tuple[set, set]:
    """Load conversation ender patterns from knowledge base (DB → file → hardcoded fallback).

    Returns (enders_set, non_enders_set).
    Caches result in memory — call invalidate_ender_cache() to reload.
    """
    global _ender_patterns, _non_ender_patterns
    if _ender_patterns is not None:
        return _ender_patterns, _non_ender_patterns

    # Hardcoded defaults (always present)
    hardcoded = {
        "ok", "okay", "okk", "okkk", "okayy", "k", "kk", "kkk",
        "thanks", "thank you", "thankyou", "thnx", "thnks", "ty",
        "got it", "noted", "sure", "fine", "alright", "right",
        "great", "good", "nice", "cool", "done", "yes", "yep", "ya",
        "theek hai", "thik hai", "theek", "thik", "teek hai",
        "accha", "acha", "achha", "ok ji", "okay ji", "ji",
        "shukriya", "dhanyawad", "dhanyavaad",
        "samajh gaya", "samajh gaye", "samjh gya", "samjha",
        "haan", "ha", "haa", "hmm", "hm", "hmmmm",
        "bilkul", "zaroor", "sahi hai", "sahi",
        "badhiya", "bohot accha", "bahut accha",
    }

    learned = set()
    non_enders = set()

    # Try loading from DB first, then file
    enders_data = None
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            enders_data = load_knowledge_from_db("conversation_enders")
    except Exception:
        pass

    if not enders_data:
        try:
            enders_file = KNOWLEDGE_DIR / "conversation_enders.json"
            with open(enders_file, "r", encoding="utf-8") as f:
                enders_data = json.load(f)
        except Exception:
            pass

    if enders_data:
        # Merge hardcoded from knowledge file
        for e in enders_data.get("hardcoded_enders", []):
            hardcoded.add(e.lower() if isinstance(e, str) else e)

        # Add learned enders
        for e in enders_data.get("learned_enders", []):
            pattern = e.get("pattern", e) if isinstance(e, dict) else e
            learned.add(pattern.lower())

        # Non-enders (false positives Ketu replied to — override hardcoded)
        for e in enders_data.get("learned_non_enders", []):
            pattern = e.get("pattern", e) if isinstance(e, dict) else e
            non_enders.add(pattern.lower())

    # Final set: hardcoded + learned - non_enders
    _ender_patterns = (hardcoded | learned) - non_enders
    _non_ender_patterns = non_enders

    logger.info(f"[Enders] Loaded {len(_ender_patterns)} patterns ({len(learned)} learned, {len(non_enders)} non-enders)")
    return _ender_patterns, _non_ender_patterns


def invalidate_ender_cache():
    """Clear cached ender patterns — call after learning new enders."""
    global _ender_patterns, _non_ender_patterns
    _ender_patterns = None
    _non_ender_patterns = None


def _is_conversation_ender(message: str, last_ai_message: str = "") -> bool:
    """Detect if customer is just acknowledging/ending the conversation.

    If the customer says "okay", "thanks", "theek hai" etc. after we gave them
    info (address, price, details), there's no need to reply. Ketu doesn't
    keep replying after the conversation is naturally done.

    Uses both hardcoded patterns and learned patterns from real chat behavior.
    Returns True if we should NOT reply.
    """
    msg = message.strip().lower()

    # Remove common punctuation
    msg_clean = msg.rstrip("!.,?").strip()

    # Load ender patterns (hardcoded + learned from knowledge base)
    enders, _ = _load_ender_patterns()

    if msg_clean in enders:
        return True

    # Short messages (1-3 words) that look like acknowledgements
    words = msg_clean.split()
    if len(words) <= 3:
        # "ok bhai", "thanks sir", "theek hai ji", "accha ok"
        if any(w in enders for w in words):
            # But NOT if they're asking something (contains question mark or question words)
            question_words = {"kya", "kab", "kaise", "kitna", "kitne", "kaha", "kahan",
                              "what", "when", "how", "which", "where", "why", "price",
                              "rate", "sample", "order", "send", "bhej", "batao", "bata"}
            if not any(w in question_words for w in words) and "?" not in msg:
                return True

    return False


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

    # Check if this is a conversation-ender (customer just acknowledged, no need to reply)
    last_ai_msg = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_ai_msg = m.get("content", "")
            break

    if last_ai_msg and _is_conversation_ender(message, last_ai_msg):
        logger.info(f"Conversation ender detected: '{message[:50]}' — skipping reply")
        # Still store the message in history but don't generate a reply
        if customer_phone:
            _conversations[customer_phone] = messages + [{"role": "user", "content": message}]
            _conversation_timestamps[customer_phone] = time.time()
            # Update customer profile even for enders
            update_profile(customer_phone, customer_name, message)
        return ""  # Empty = don't send

    # Add current message
    messages = messages + [{"role": "user", "content": message}]

    # Load insights from DB on first call (survives deploys)
    _load_customer_insights_from_db()

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

    # Persist to DB
    _save_customer_insights_to_db()

    # Update customer memory profile
    if customer_phone:
        update_profile(customer_phone, customer_name, message)
        # Reset follow-up flag since customer is messaging again
        reset_follow_up_flag(customer_phone)

    # Detect escalation (complaints, anger, frustration)
    escalation = detect_escalation(message, messages)
    if customer_phone:
        _last_escalation[customer_phone] = escalation
    escalation_modifier = escalation.get("prompt_modifier", "")

    # Log escalation for Ketu's attention
    if escalation["level"] == LEVEL_ESCALATE:
        notice = format_escalation_notice(
            customer_phone, customer_name, message, escalation["reason"]
        )
        logger.warning(f"[ESCALATION] {notice}")
        # Log to activity so it shows on dashboard
        from core.activity_log import log_activity
        log_activity(
            source="escalation",
            action="flagged",
            details={
                "customer_phone": customer_phone[-4:] if customer_phone else "unknown",
                "customer_name": customer_name,
                "reason": escalation["reason"],
                "level": escalation["level"],
                "message_preview": message[:100],
            },
            items_count=1,
        )

    # Build system prompt with customer context and escalation modifier
    system = _build_system_prompt(
        customer_phone=customer_phone,
        escalation_modifier=escalation_modifier,
    )
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

            # Track API cost
            track_api_cost(
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                source="whatsapp-reply",
                customer_phone=customer_phone[-4:] if customer_phone else "",
            )

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


def _load_faq_hits_from_db():
    """Load FAQ hit counts from DB on first access."""
    global _faq_hit_counts, _faq_hits_loaded
    if _faq_hits_loaded:
        return
    _faq_hits_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get("faq_hit_counts")
            if data and isinstance(data, dict):
                _faq_hit_counts.update(data)
    except Exception as e:
        logger.warning(f"[FAQ Hits] DB load failed: {e}")


def track_faq_hit(question: str):
    """Track that a FAQ was used in a reply."""
    _load_faq_hits_from_db()
    key = question[:60]
    _faq_hit_counts[key] = _faq_hit_counts.get(key, 0) + 1
    # Persist to DB
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("faq_hit_counts", _faq_hit_counts)
    except Exception:
        pass


def get_customer_insights() -> dict:
    """Get customer message insights."""
    _load_customer_insights_from_db()
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
    _load_faq_hits_from_db()
    return sorted(
        [{"question": q, "hits": c} for q, c in _faq_hit_counts.items()],
        key=lambda x: x["hits"],
        reverse=True,
    )


def get_last_escalation(phone: str) -> dict:
    """Get the last escalation result for a customer phone."""
    return _last_escalation.get(phone, {"level": "none", "reason": "", "prompt_modifier": ""})
