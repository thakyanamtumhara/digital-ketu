import json
import time
import logging

from anthropic import Anthropic
import httpx

from core.config import settings, KNOWLEDGE_DIR
from core.knowledge import format_context, load_knowledge
from core.context_selector import classify_message, format_smart_context
from core.cost_tracker import track_api_cost
from core.customer_memory import (
    get_profile, update_profile, format_customer_context, reset_follow_up_flag,
)
from core.escalation import detect_escalation, format_escalation_notice, LEVEL_ESCALATE
from core.ketu_only import detect_ketu_only, log_deferred_question

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

# "Shut up" cooldown — when conversation ends (ender detected) or Ketu manually replies,
# AI stays silent for this customer. Prevents unnecessary follow-up replies.
_shutup_until: dict[str, float] = {}  # phone -> timestamp until which AI stays silent
_shutup_reason: dict[str, str] = {}  # phone -> reason (ender, ketu_manual_reply, ketu_only_deferral)
SHUTUP_COOLDOWN = 300  # 5 minutes — AI won't reply to this customer for 5 min after ender

# Prompt cache — caches the knowledge portion of system prompt by intent combo
# Key: frozenset of (intents + product_ids), Value: (prompt_text, timestamp)
_prompt_cache: dict[str, tuple[str, float]] = {}
PROMPT_CACHE_TTL = 60  # seconds — same as knowledge cache TTL

# Simple intents that can use Haiku (10x cheaper) instead of Sonnet
HAIKU_INTENTS = {"greeting", "shipping_delivery", "payment", "location_visit",
                 "gst_invoice", "moq", "return_complaint", "dropshipping", "order_how"}
HAIKU_MODEL = "claude-haiku-4-5-20251001"
SONNET_MODEL = "claude-sonnet-4-20250514"

# Estimated full context token count (for savings tracking)
# Updated periodically from actual API responses using full context
_estimated_full_tokens: int = 5500  # conservative default


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


def _load_repeat_buyer_style() -> list[str]:
    """Load learned repeat buyer reply examples from knowledge base."""
    try:
        from core.database import is_db_available, load_knowledge_from_db
        data = None
        if is_db_available():
            data = load_knowledge_from_db("repeat_buyer_style")
        if not data:
            style_file = KNOWLEDGE_DIR / "repeat_buyer_style.json"
            with open(style_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        # Extract reply strings
        replies = []
        for r in data.get("repeat_buyer_replies", []) + data.get("returning_buyer_replies", []):
            reply = r.get("reply", r) if isinstance(r, dict) else r
            if reply and len(reply) > 5:
                replies.append(reply)
        return replies[-10:]  # Last 10 examples
    except Exception:
        return []


def _make_prompt_cache_key(classification: dict) -> str:
    """Create a cache key from classification results."""
    intents = tuple(sorted(classification.get("intents", [])))
    products = tuple(sorted(classification.get("product_ids", [])))
    return f"{intents}|{products}"


def _build_system_prompt(
    customer_phone: str = "",
    escalation_modifier: str = "",
    customer_message: str = "",
) -> tuple[str, dict | None]:
    """Build system prompt dynamically from prompt.json + knowledge context.

    This prompt evolves over time as Digital Ketu learns from Ketu's real messages.
    Includes customer memory context and escalation handling when applicable.

    Smart context: When customer_message is provided, classifies the question and
    includes only relevant knowledge sections (60-70% token savings).
    Falls back to full context if classification fails.

    Returns (system_prompt, classification_dict_or_None).
    """
    config = _load_prompt_config()

    # Smart context selection — classify message and pick relevant sections only
    if customer_message:
        classification = classify_message(customer_message)

        # Check prompt cache — same intent combo = same knowledge context
        cache_key = _make_prompt_cache_key(classification)
        cached = _prompt_cache.get(cache_key)
        if cached and (time.time() - cached[1]) < PROMPT_CACHE_TTL:
            knowledge_context = cached[0]
            logger.info(f"[SmartContext] Cache HIT for {cache_key[:40]}")
        else:
            knowledge = load_knowledge()
            knowledge_context = format_smart_context(knowledge, classification)
            _prompt_cache[cache_key] = (knowledge_context, time.time())
            logger.info(
                f"[SmartContext] intents={classification['intents']}, "
                f"products={len(classification['product_ids'])}, "
                f"complex={classification['is_complex']}"
            )
    else:
        classification = None
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

    # Compact rules — merged suggestion, objection handling, and chat patterns
    sections.append(
        "## KEY RULES:\n"
        "- Unsolicited product push KABHI mat kar. Customer jo maange wohi de.\n"
        "- Objection handling: 'Mehnga hai' → factory direct, no middleman, biowash+no shrinkage guarantee. 'Competitor sasta' → quality compare karo. 'Discount' → bulk rate bata. Confident reh, argue mat kar.\n"
        "- INQUIRY → rate de. COMPLAINT → empathy + Ketu sir connect. CLOSING → website link. ACKNOWLEDGMENT (ok/thanks) → reply mat kar. REPEAT BUYER → short friendly reply."
    )

    # Repeat buyer learned style (if available)
    if customer_phone:
        profile = get_profile(customer_phone)
        if profile.get("stage") in ("repeat", "bought") and profile.get("purchase_count", 0) >= 1:
            repeat_style = _load_repeat_buyer_style()
            if repeat_style:
                examples = repeat_style[:5]
                example_lines = "\n".join(f'- "{e}"' for e in examples)
                sections.append(
                    f"## REPEAT BUYER KO KAISE BAAT KARO (Ketu ke real replies se seekha):\n"
                    f"{example_lines}\n"
                    f"Ye customer {profile.get('purchase_count', 1)} baar order kar chuka hai. "
                    f"Process jaanta hai. Short, friendly reply de — jaise purane customer ko dete hain."
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

    sections.append("## SABSE ZAROORI RULES:\n"
        "1. Reply MAXIMUM 1-2 lines. Bas. 3 line se zyada KABHI nahi. Ketu WhatsApp pe chhota likhta hai — ek do line mein baat khatam. Jitna kam utna better. Cost bhi bachta hai.\n"
        "2. KABHI fake promise mat karo. Nahi pata toh bol: 'Ye Ketu sir batayenge, thodi der mein reply aayega.'\n"
        "3. STOCK/RESTOCK TIMELINE: Tu KABHI mat bol 'X din mein aa jayega', '7-10 days', '15-20 days', 'next week' etc. Tujhe NAHI pata stock kab aayega — sirf Ketu jaanta hai. Agar koi puche kab aayega, toh bol: 'Bhai ye Ketu sir khud batayenge, thodi der mein reply aayega.' KABHI timeline fabricate mat kar.\n"
        "4. 'WhatsApp karo' ya WhatsApp number KABHI mat de — customer ALREADY isi WhatsApp pe baat kar raha hai.\n"
        "5. Website link har reply mein mat daal — ek conversation mein ek baar kaafi hai.\n"
        "6. Same line baar baar repeat mat kar — robot lagta hai, natural baat kar.\n"
        "7. FIRST MESSAGE RULE: Agar customer PEHLI BAAR message kar raha hai (conversation mein sirf 1 user message hai), toh reply ke end mein catalogue link naturally add kar: 'Poora catalogue yahan dekho: sale91.com/catalog' — push mat kar, bas casually share kar taaki customer website pe browse kare. Baad ke messages mein link DUBARA mat de.")

    return "\n\n".join(sections), classification


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


def activate_shutup(customer_phone: str, reason: str = "ender", minutes: float = 0):
    """Put AI in "shut up" mode for a customer — don't reply for SHUTUP_COOLDOWN seconds.

    Called when:
    - Conversation ender detected (buyer said "ok", "thanks", etc.)
    - Ketu manually replied to a customer (AI should back off)
    - Ketu-only question deferred to real Ketu (AI should wait)
    """
    if not customer_phone:
        return
    duration = minutes * 60 if minutes > 0 else SHUTUP_COOLDOWN
    _shutup_until[customer_phone] = time.time() + duration
    _shutup_reason[customer_phone] = reason
    logger.info(f"[ShutUp] Activated for {customer_phone[-4:]} — reason: {reason}, duration: {duration}s")


def is_shutup_active(customer_phone: str) -> bool:
    """Check if AI should stay silent for this customer."""
    if not customer_phone:
        return False
    until = _shutup_until.get(customer_phone, 0)
    if time.time() < until:
        remaining = int(until - time.time())
        reason = _shutup_reason.get(customer_phone, "unknown")
        logger.info(f"[ShutUp] Active for {customer_phone[-4:]} — {remaining}s remaining, reason: {reason}")
        return True
    # Expired — clean up
    _shutup_until.pop(customer_phone, None)
    _shutup_reason.pop(customer_phone, None)
    return False


def ketu_manual_reply(customer_phone: str):
    """Called when Ketu manually replies to a customer.

    Activates shut-up mode so AI doesn't jump back into the conversation.
    wwbun should call this (via /api/ketu-replied) when it detects Ketu typing.
    """
    activate_shutup(customer_phone, reason="ketu_manual_reply", minutes=10)


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

    # --- TRACK CUSTOMER INSIGHTS (before any early returns) ---
    # Every incoming message should be counted, regardless of shutup/ender/ketu-only
    _load_customer_insights_from_db()
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
    _save_customer_insights_to_db()

    # --- SHUT UP CHECK ---
    # If AI is in cooldown for this customer (ender detected earlier or Ketu replied),
    # don't reply at all. This prevents the AI from jumping back into finished conversations.
    if customer_phone and is_shutup_active(customer_phone):
        reason = _shutup_reason.get(customer_phone, "")

        # If Ketu is handling this customer (manual reply or ketu-only deferral),
        # AI must NEVER jump back in — regardless of what customer says.
        # Ketu is the boss. Only Ketu can break this cooldown (by not replying,
        # letting the timer expire).
        if reason in ("ketu_manual_reply", "ketu_only_deferral"):
            logger.info(
                f"[ShutUp] Skipping reply to {customer_phone[-4:]} — "
                f"Ketu is handling (reason: {reason}), msg: '{message[:50]}'"
            )
            if customer_phone:
                _conversations[customer_phone] = messages + [{"role": "user", "content": message}]
                _conversation_timestamps[customer_phone] = time.time()
            return ""

        # For ender-based cooldown: if customer asks a NEW question, break the cooldown
        msg_lower = message.strip().lower().rstrip("!.,")
        has_question = "?" in message or any(
            w in msg_lower.split() for w in
            {"kya", "kab", "kaise", "kitna", "kitne", "kaha", "price", "rate",
             "sample", "order", "send", "bhej", "batao", "bata", "what", "when",
             "how", "which", "where", "why"}
        )
        if not has_question:
            logger.info(f"[ShutUp] Skipping reply to {customer_phone[-4:]} — cooldown active, msg: '{message[:50]}'")
            if customer_phone:
                _conversations[customer_phone] = messages + [{"role": "user", "content": message}]
                _conversation_timestamps[customer_phone] = time.time()
            return ""
        else:
            logger.info(f"[ShutUp] Customer {customer_phone[-4:]} asked new question — breaking cooldown: '{message[:50]}'")
            _shutup_until.pop(customer_phone, None)
            _shutup_reason.pop(customer_phone, None)

    # --- CONVERSATION ENDER CHECK ---
    # Check if customer is just acknowledging/ending the conversation
    last_ai_msg = ""
    for m in reversed(messages):
        if m.get("role") == "assistant":
            last_ai_msg = m.get("content", "")
            break

    # Fix: Even without conversation history, pure enders should still be detected.
    # "Ok" by itself is ALWAYS an ender — doesn't need last_ai_msg context.
    is_ender = False
    if _is_conversation_ender(message, last_ai_msg):
        if last_ai_msg:
            # Normal case: we have history, ender detected
            is_ender = True
        else:
            # No history (server restart, etc.) — still detect pure enders
            # Only skip if it's a clear standalone ender (not a question)
            msg_clean = message.strip().lower().rstrip("!.,?").strip()
            enders, _ = _load_ender_patterns()
            if msg_clean in enders:
                is_ender = True
                logger.info(f"[Ender] Pure ender detected without history: '{message[:50]}'")

    if is_ender:
        logger.info(f"Conversation ender detected: '{message[:50]}' — skipping reply")
        # Activate shut-up cooldown so subsequent messages also get skipped
        activate_shutup(customer_phone, reason="ender_detected")
        # Still store the message in history but don't generate a reply
        if customer_phone:
            _conversations[customer_phone] = messages + [{"role": "user", "content": message}]
            _conversation_timestamps[customer_phone] = time.time()
            # Update customer profile even for enders
            update_profile(customer_phone, customer_name, message)
        return ""  # Empty = don't send

    # Check if this is a "Ketu Only" question (stock timeline, order status, etc.)
    # AI must NOT fabricate answers — defer to real Ketu
    ketu_check = detect_ketu_only(message, customer_phone, customer_name)
    if ketu_check:
        logger.info(f"[KetuOnly] Deferring to Ketu: {ketu_check['category_name']} | {message[:50]}")
        log_deferred_question(
            customer_phone=customer_phone,
            customer_name=customer_name,
            message=message,
            category_id=ketu_check["category_id"],
            category_name=ketu_check["category_name"],
            reason=ketu_check["reason"],
            defer_reply=ketu_check["defer_reply"],
        )
        # Store in conversation history so context is maintained
        defer_reply = ketu_check["defer_reply"]
        if customer_phone:
            _conversations[customer_phone] = messages + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": defer_reply},
            ]
            _conversation_timestamps[customer_phone] = time.time()
            update_profile(customer_phone, customer_name, message)
            # SHUT UP after deferring to Ketu — AI told customer "Ketu sir reply karenge",
            # so AI must stay silent until Ketu actually replies. 10 min cooldown.
            activate_shutup(customer_phone, reason="ketu_only_deferral", minutes=10)
        return defer_reply

    # Trim conversation history to last 6 messages (3 exchanges) before adding new one
    # This prevents sending 10-20 old messages to Claude (saves ~5,000 tokens per call)
    if len(messages) > 5:
        messages = messages[-5:]

    # Add current message
    messages = messages + [{"role": "user", "content": message}]

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

    # Build system prompt with customer context, escalation modifier, and smart context
    system, classification = _build_system_prompt(
        customer_phone=customer_phone,
        escalation_modifier=escalation_modifier,
        customer_message=message,
    )
    if customer_name:
        system += f"\n\nCustomer name: {customer_name}"

    # Tell AI if this is the first message (for catalogue link rule)
    user_msg_count = sum(1 for m in messages if m.get("role") == "user")
    if user_msg_count == 1:
        system += "\n\n>> YE CUSTOMER KA PEHLA MESSAGE HAI. Catalogue link share kar end mein: sale91.com/catalog"

    # Model selection — use Haiku for simple questions (10x cheaper)
    use_smart = classification is not None and not classification.get("is_complex", True)
    intents = set(classification.get("intents", [])) if classification else set()

    if use_smart and intents and intents.issubset(HAIKU_INTENTS):
        model = HAIKU_MODEL
        logger.info(f"[ModelSelect] Using Haiku for simple intents: {intents}")
    else:
        model = SONNET_MODEL

    # Retry with exponential backoff (max 3 attempts)
    last_error = None
    for attempt in range(3):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=200,
                system=system,
                messages=messages,
                timeout=httpx.Timeout(30.0, connect=10.0),
            )

            reply = response.content[0].text

            # Track API cost with smart context savings
            track_api_cost(
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                source="whatsapp-reply",
                customer_phone=customer_phone[-4:] if customer_phone else "",
                smart_context_used=use_smart,
                estimated_full_tokens=_estimated_full_tokens,
            )

            # Store conversation history
            if customer_phone:
                _conversations[customer_phone] = messages + [
                    {"role": "assistant", "content": reply}
                ]
                # Keep only last 6 messages (3 exchanges) — saves ~5,000 tokens per call
                if len(_conversations[customer_phone]) > 6:
                    _conversations[customer_phone] = _conversations[customer_phone][-6:]
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
    # Stay silent — don't send confusing "technical issue" message to customer.
    # Silence is better than a nonsensical reply. Ketu can reply manually if needed.
    return ""


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
