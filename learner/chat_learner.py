import json
import logging
import re

from anthropic import Anthropic

from core.config import settings, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)
PROMPT_FILE = KNOWLEDGE_DIR / "prompt.json"


def _load_prompt_config() -> dict:
    """Load current prompt config — DB first (survives deploys), file fallback."""
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            data = load_knowledge_from_db("prompt")
            if data:
                return data
    except Exception:
        pass
    try:
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# --- Smart Message Filter ---
# Messages that are too short or match these patterns are noise, not knowledge.

# Exact matches (case-insensitive) — zero information value
_JUNK_EXACT = {
    "ok", "okay", "okk", "okkk", "k", "kk", "kkk",
    "yes", "no", "nahi", "nhi", "na", "ha", "haa", "haan", "hanji", "han",
    "ji", "ji sir", "ji bhai", "jii", "ji ha", "haan ji", "hanji sir",
    "hmm", "hm", "hmmmm", "achha", "accha", "acha", "oh", "ohh",
    "hi", "hello", "hey", "hii", "hiii",
    "thanks", "thank you", "thanku", "thnx", "ty", "dhanyawad", "shukriya",
    "good morning", "good night", "good evening", "gm", "gn",
    "bye", "bye bye", "ok bye", "chalo bye",
    "theek hai", "thik hai", "sahi hai", "done", "ho gaya",
    "aa jao", "aa raha hun", "aa raha hu", "aata hun", "aata hu",
    "ruk", "ruko", "wait", "ek min", "ek minute",
    "bolo", "batao", "haan batao", "han bolo",
    "seen", "delivered", "read",
    "location", "live location",
    "👍", "👍🏻", "🙏", "🙏🏻", "❤️", "😊", "😂", "🤣", "👌", "✅", "🔥",
}

# Patterns — WhatsApp system messages, media & automated welcome messages
_JUNK_PATTERNS = [
    r"^<media omitted>$",
    r"^<image omitted>$",
    r"^<video omitted>$",
    r"^<audio omitted>$",
    r"^<document omitted>$",
    r"^<sticker omitted>$",
    r"^<contact card omitted>$",
    r"^<location:.*>$",
    r"^this message was deleted$",
    r"^you deleted this message$",
    r"^missed voice call$",
    r"^missed video call$",
    r"^\d{10,13}$",  # Just a phone number
    r"^https?://maps\.google",  # Google Maps links (location sharing)
    r"^https?://wa\.me/",  # WhatsApp links
    # Media-only placeholders — no learning value for text-based AI
    r"^\[?image\]?$",
    r"^\[?video\]?$",
    r"^\[?audio\]?$",
    r"^\[?document\]?$",
    r"^\[?sticker\]?$",
    r"^\[?voice\s*note\]?$",
    r"^\[?gif\]?$",
    r"^\[?contact\s*card\]?$",
    r"^\[?location\]?$",
    r"^\[?image\s*sent\]?$",
    r"^\[?document\s*sent\]?$",
    r"^\[?video\s*sent\]?$",
    # Welcome / automated template messages — no learning value
    r"welcome\s*(to|sir|ji|bhai|!)",
    r"swagat\s*hai",
    r"thank\s*(you|u)\s*(for\s*)?(contact|reach|enquir|interest)",
    r"thanks?\s*(for\s*)?(your\s*)?(contact|reach|enquir|interest)",
    r"dhanyawad.*(?:sampark|enquiry|interest)",
    r"aapka\s*(?:swagat|welcome)",
    r"namaste.*(?:welcome|swagat|enquiry)",
    r"hello.*(?:welcome|thank.*contact|thank.*enquir)",
]
_JUNK_COMPILED = [re.compile(p, re.IGNORECASE) for p in _JUNK_PATTERNS]

# Minimum word count for a message to be "useful" (owner messages need substance)
_MIN_WORDS_OWNER = 3  # Ketu's messages must have 3+ words to be worth learning
_MIN_WORDS_CUSTOMER = 2  # Customer messages need 2+ words (e.g. "180 gsm", "cod available")


def is_junk_message(text: str) -> bool:
    """Check if a message is noise/junk that has no learning value."""
    cleaned = text.strip().lower()

    # Empty
    if not cleaned:
        return True

    # Exact junk match
    if cleaned in _JUNK_EXACT:
        return True

    # Pure emoji (1-3 emojis, no real text)
    text_only = re.sub(r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\ufe00-\ufe0f]|[\u200d]|[\u20e3]|[\ufe0f]', '', cleaned).strip()
    if not text_only:
        return True

    # Pattern match
    for pattern in _JUNK_COMPILED:
        if pattern.search(cleaned):
            return True

    return False


def is_media_only_message(text: str) -> bool:
    """Check if a message is media-only (image, document, video, etc.) with no real text content.

    These are not useful for text-based learning — an AI can't learn to reply with images.
    Also catches cases like '[Image sent — could be payment screenshot]' or '[Document]'.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return True

    # Direct media placeholders: [Image], [Document], [Video], etc.
    if re.match(r'^\[?\s*(?:image|video|audio|document|sticker|voice\s*note|gif|contact\s*card|location|photo|pdf|file)'
                r'(?:\s+sent)?(?:\s*[-—:].*)?\s*\]?\s*$', cleaned, re.IGNORECASE):
        return True

    # WhatsApp export format: <media omitted>, <image omitted>, etc.
    if re.match(r'^<\s*(?:media|image|video|audio|document|sticker|contact\s*card)\s*omitted\s*>$', cleaned, re.IGNORECASE):
        return True

    return False


def has_business_intent(text: str) -> bool:
    """Check if a customer message shows genuine business intent worth learning from.

    Returns True if the message looks like a real business inquiry, order-related
    question, complaint, product inquiry, or negotiation — the kind of messages
    where the owner's reply would teach the AI something useful.

    Returns False for random chit-chat, forwarded messages, or non-business content.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return False

    # Too short to have business intent (single word greetings handled by junk filter)
    words = cleaned.split()
    if len(words) < _MIN_WORDS_CUSTOMER:
        return False

    # Business intent signals — if ANY match, it's likely a business message
    _BUSINESS_SIGNALS = [
        # Price / cost inquiry
        r'(?:price|rate|cost|kitna|kitne|kya\s*rate|kya\s*price|amount|charges?|shipping\s*charge)',
        # Product inquiry
        r'(?:available|stock|hai\s*kya|milega|mil\s*jayega|in\s*stock|out\s*of\s*stock)',
        r'(?:size|colour|color|gsm|material|quality|weight|dimension|specification)',
        r'(?:sample|catalog|catalogue|brochure|photo|image|pics?|pictures?)',
        # Order / purchase
        r'(?:order|book|buy|purchase|kharidna|lena\s*hai|chahiye|chaiye|mangta|mangwana)',
        r'(?:quantity|qty|minimum\s*order|moq|bulk|wholesale|retail)',
        r'(?:cod|cash\s*on\s*delivery|online\s*payment|upi|gpay|phonepe|paytm|neft|rtgs)',
        # Delivery / shipping
        r'(?:delivery|dispatch|ship|courier|transport|deliver|bhej|bhejna|bhejdo)',
        r'(?:pin\s*code|pincode|location|address|kahan|where)',
        r'(?:time|kitne\s*din|days|kab\s*tak|when|estimated)',
        # Negotiation
        r'(?:discount|kam|less|kuch\s*kam|thoda\s*kam|best\s*price|final\s*price|last\s*price)',
        # Complaint / issue
        r'(?:problem|issue|defect|broken|damaged|wrong|galat|kharab|return|refund|exchange|replace)',
        # Confirmation / follow-up
        r'(?:confirm|payment\s*done|paid|sent|bhej\s*diya|kar\s*diya|ho\s*gaya|tracking)',
        # Questions (generic business questions)
        r'(?:kaise|how|kya|what|which|konsa|kaun)',
        # Shop / godown availability
        r'(?:open|band|close|closed|chutti|holiday|timing|visit|godown|shop|dukan|warehouse)',
    ]

    for pattern in _BUSINESS_SIGNALS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return True

    # If message has a question mark, it's likely an inquiry
    if '?' in cleaned:
        return True

    # Messages with numbers often relate to quantities, prices, sizes
    if re.search(r'\d+', cleaned) and len(words) >= 2:
        return True

    # If none of the signals match but it's long enough (5+ words),
    # give it the benefit of the doubt — could be a detailed message
    if len(words) >= 5:
        return True

    return False


def is_low_quality_owner_reply(text: str) -> bool:
    """Check if an owner (Ketu) reply is too low-quality to be a learning pair.

    Catches:
    - Media-only replies (image, document, etc.)
    - Very short acknowledgments that don't teach anything
    - Pure URLs with no explanation
    But KEEPS:
    - URL + meaningful text (e.g., "Check this link for catalog: https://...")
    """
    if is_media_only_message(text):
        return True
    if is_junk_message(text):
        return True

    cleaned = text.strip()

    # Pure URL with no explanation text (just a link, no context)
    if re.match(r'^https?://\S+$', cleaned) and len(cleaned.split()) == 1:
        return True

    # URL + text combo: strip URLs and check if remaining text is meaningful
    text_without_urls = re.sub(r'https?://\S+', '', cleaned).strip()
    if text_without_urls != cleaned:  # had URLs
        # If there's meaningful text alongside the URL, keep it
        if len(text_without_urls.split()) >= _MIN_WORDS_OWNER:
            return False
        # URL + junk text like "ok https://..." is still low quality
        if not text_without_urls or is_junk_message(text_without_urls):
            return True

    # Too short for a meaningful owner reply (less than 3 words)
    if len(cleaned.split()) < _MIN_WORDS_OWNER:
        return True

    return False


def _is_owner_in_filter(m: dict, owner_key: str, owner_value: str) -> bool:
    """Determine if message is from owner — uses sender_id as authority (like main._is_owner_message).

    IMPORTANT: Do NOT trust is_owner flag alone — wwbun may send is_owner=True for ALL messages.
    Use sender_id match as the authoritative source when available.
    """
    sid = m.get("sender_id") or m.get(owner_key, "")
    sid = str(sid).strip()

    # If sender_id present and owner_value given, use sender_id as authority
    if sid and owner_value:
        ov = owner_value.strip()
        if sid == ov or ov.endswith(sid) or sid.endswith(ov):
            return True
        return False  # sender_id doesn't match → customer

    # Fallback to flags only if no sender_id
    raw_owner = m.get("is_owner", False)
    if isinstance(raw_owner, str):
        raw_owner = raw_owner.lower() in ("true", "1", "yes")
    if bool(raw_owner):
        return True
    if m.get("fromMe") or m.get("from_me"):
        return True
    return False


def filter_messages(messages: list[dict], owner_key: str = "sender", owner_value: str = "") -> tuple[list[dict], dict]:
    """Filter out junk messages, keep only knowledge-worthy ones.

    Returns (filtered_messages, stats_dict).
    """
    filtered = []
    stats = {"total": len(messages), "junk": 0, "too_short": 0, "kept": 0}

    for m in messages:
        text = m.get("text", "") or m.get("content", "")

        if is_junk_message(text):
            stats["junk"] += 1
            continue

        # Word count check — use sender_id-based owner detection (not just is_owner flag)
        word_count = len(text.split())
        is_owner = _is_owner_in_filter(m, owner_key, owner_value)

        min_words = _MIN_WORDS_OWNER if is_owner else _MIN_WORDS_CUSTOMER
        if word_count < min_words:
            stats["too_short"] += 1
            continue

        filtered.append(m)
        stats["kept"] += 1

    logger.info(
        f"Message filter: {stats['total']} total → {stats['kept']} kept "
        f"({stats['junk']} junk, {stats['too_short']} too short)"
    )
    return filtered, stats


def parse_whatsapp_export(chat_text: str) -> list[dict]:
    """Parse WhatsApp chat export text into structured messages."""
    messages = []
    # WhatsApp export format: [DD/MM/YY, HH:MM:SS] Name: Message
    pattern = r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:AM|PM|am|pm)?\]?\s*-?\s*([^:]+):\s*(.*)'

    for line in chat_text.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            date, time_str, sender, text = match.groups()
            messages.append({
                "date": date.strip(),
                "time": time_str.strip(),
                "sender": sender.strip(),
                "text": text.strip(),
            })
        elif messages and line.strip():
            # Continuation of previous message
            messages[-1]["text"] += "\n" + line.strip()

    return messages


def extract_knowledge_from_messages(messages: list[dict], ketu_name: str = "Ketu") -> dict:
    """Use Claude to extract knowledge from Ketu's manual messages."""
    client = Anthropic(api_key=settings.anthropic_api_key)

    # Smart filter: remove junk messages BEFORE sending to Claude
    filtered, filter_stats = filter_messages(messages, owner_key="sender", owner_value=ketu_name)

    # Filter only Ketu's messages (manual typing by the owner)
    ketu_messages = [m for m in filtered if ketu_name.lower() in m["sender"].lower()]

    if not ketu_messages:
        return {"status": "no_messages", "updates": [], "filter_stats": filter_stats}

    # Build conversation context for Claude to analyze (only quality messages)
    chat_sample = "\n".join(
        f"[{m['date']} {m['time']}] {m['sender']}: {m['text']}"
        for m in filtered[:200]
    )

    # NOTE: We do NOT send current traits/phrases to Claude anymore.
    # The 194KB prompt.json was adding ~8000+ tokens per call for dedup that
    # already happens in apply_knowledge_updates() AFTER Claude returns.

    prompt = f"""Analyze these WhatsApp chat messages from Ketu (business owner). Learn FROM Ketu's messages only, use customer questions as context.
Ketu's name in chat: {ketu_name}

Chat messages:
{chat_sample}

Do NOT extract price info or product details (materials, colors, sizes, GSM) — catalog is the source of truth.

Extract JSON:
1. "style_patterns": Ketu's phrases, greetings, tone, typical replies
2. "new_faqs": Customer Q + Ketu A pairs (NOT price/product FAQs)
3. "business_updates": New policies, offers, shipping info
4. "prompt_evolution": {{"new_traits": [], "new_phrases": [], "new_rules": [], "example_conversations": [{{"customer": "q", "reply": "a"}}]}}

Return ONLY valid JSON. If nothing new, return empty arrays."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text

        # Track API cost
        from core.cost_tracker import track_api_cost
        track_api_cost(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            source="whatsapp-learning",
        )

        # Log payload for dashboard debug
        from core.cloud_payload_log import log_cloud_payload
        log_cloud_payload(
            source="whatsapp-learning",
            prompt_text=prompt,
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            message_count=len(filtered),
        )

        # Try to parse JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"status": "parse_error", "raw": result_text}

    except Exception as e:
        logger.error(f"Knowledge extraction error: {e}")
        return {"status": "error", "detail": str(e)}


def extract_knowledge_from_wwbun_messages(
    messages: list[dict],
    owner_user_id: str,
) -> dict:
    """Extract knowledge from wwbun database messages.

    CRITICAL: Only learn from messages sent by the owner (manual typing).
    AI-generated messages should be tagged and excluded.

    Args:
        messages: List of message dicts from wwbun database
        owner_user_id: The user ID of Ketu (to identify his manual messages)
    """
    client = Anthropic(api_key=settings.anthropic_api_key)

    # Helper: safely parse boolean (wwbun may send string "true"/"false")
    def _safe_bool(val) -> bool:
        if isinstance(val, str):
            return val.lower() in ("true", "1", "yes")
        return bool(val)

    # Helper: check if a message is from the owner (Ketu)
    def _is_owner_msg(m: dict) -> bool:
        sid = m.get("sender_id")
        # sender_id is authoritative — if present and doesn't match owner, it's a customer
        if sid and owner_user_id:
            return sid == owner_user_id or owner_user_id.endswith(sid) or sid.endswith(owner_user_id)
        return _safe_bool(m.get("is_owner", False))

    # Detect broken sender_id: all messages have same sender_id (wwbun bug)
    unique_sids = set(str(m.get("sender_id", "")).strip() for m in messages if m.get("sender_id"))
    broken_sids = len(unique_sids) <= 1

    # Check if is_owner field has mixed values (both True and False) — means it's reliable
    _owner_true = any(_safe_bool(m.get("is_owner")) for m in messages if m.get("is_owner") is not None)
    _owner_false = any(not _safe_bool(m.get("is_owner")) for m in messages if m.get("is_owner") is not None)
    is_owner_reliable = _owner_true and _owner_false

    def _is_owner_by_flag(m: dict) -> bool:
        """Use is_owner/fromMe flags only, ignoring sender_id."""
        return _safe_bool(m.get("is_owner", False)) or _safe_bool(m.get("fromMe", False)) or _safe_bool(m.get("from_me", False))

    if broken_sids and not is_owner_reliable:
        logger.warning(
            f"[wwbun-learn] ALL sender_ids identical ({unique_sids}) AND is_owner unreliable — "
            f"using heuristic: 5+ word non-AI = manual Ketu msg."
        )
        # Last resort: treat longer non-AI messages as Ketu's manual replies
        manual_messages = [
            m for m in messages
            if not _safe_bool(m.get("is_ai_generated", False))
            and len((m.get("content", "") or m.get("text", "") or m.get("body", "")).split()) >= 5
        ]
    elif broken_sids and is_owner_reliable:
        logger.info(
            f"[wwbun-learn] sender_ids broken but is_owner field reliable — using is_owner for filtering"
        )
        manual_messages = [
            m for m in messages
            if _is_owner_by_flag(m)
            and not _safe_bool(m.get("is_ai_generated", False))
        ]
    else:
        # Normal mode: filter by owner
        manual_messages = [
            m for m in messages
            if _is_owner_msg(m)
            and not _safe_bool(m.get("is_ai_generated", False))
        ]

    if not manual_messages:
        return {"status": "no_manual_messages", "updates": [], "filter_stats": {"total": len(messages), "kept": 0, "junk": 0, "too_short": 0}}

    # Smart filter: remove junk messages BEFORE sending to Claude
    filtered, filter_stats = filter_messages(messages, owner_key="sender_id", owner_value=owner_user_id)

    # Check if we have any quality Ketu messages BEFORE calling Claude API
    quality_ketu_msgs = [
        m for m in filtered
        if _is_owner_msg(m) and not _safe_bool(m.get("is_ai_generated", False))
    ]
    if not quality_ketu_msgs:
        logger.info(f"[wwbun-learn] 0 quality Ketu messages after filtering — skipping API call (saved money)")
        return {
            "status": "no_quality_messages",
            "updates": [],
            "filter_stats": filter_stats,
            "quality_messages": [],
        }

    # Cap at 20 Ketu messages max — sending more wastes tokens with minimal learning gain
    # The threshold is 20 pairs, so we should never need more than 20 Ketu messages
    if len(quality_ketu_msgs) > 20:
        logger.info(f"[wwbun-learn] Capping quality Ketu msgs from {len(quality_ketu_msgs)} to 20")
        quality_ketu_msgs = quality_ketu_msgs[-20:]  # Keep newest 20

    # Extract only quality PAIRS (customer question + Ketu reply) to send to Claude
    # This avoids sending 100+ messages when only ~20 pairs matter (~70% token savings)
    quality_ketu_set = set(id(m) for m in quality_ketu_msgs)
    pair_messages = []
    for i, m in enumerate(filtered):
        if id(m) in quality_ketu_set:
            # Include preceding customer message for context (if exists)
            if i > 0 and id(filtered[i - 1]) not in quality_ketu_set:
                pair_messages.append(filtered[i - 1])
            pair_messages.append(m)

    # Tag conversations from buyers ([BUYER]) for sales-prioritized learning
    buyer_phones = set()
    try:
        from core.customer_memory import get_profile, STAGE_BOUGHT, STAGE_REPEAT
        for m in pair_messages:
            phone = m.get("contact_phone", "") or m.get("phone", "")
            if phone and phone not in buyer_phones:
                profile = get_profile(phone)
                if profile.get("stage") in (STAGE_BOUGHT, STAGE_REPEAT):
                    buyer_phones.add(phone)
    except Exception:
        pass

    chat_context = []
    for m in pair_messages:
        text = m.get("content", "") or m.get("text", "") or m.get("body", "")
        if broken_sids and not is_owner_reliable:
            role = "KETU" if len(text.split()) >= 5 else "CUSTOMER"
        elif broken_sids and is_owner_reliable:
            role = "KETU" if _is_owner_by_flag(m) else "CUSTOMER"
        else:
            role = "KETU" if _is_owner_msg(m) else "CUSTOMER"
        is_ai = " [AI]" if _safe_bool(m.get("is_ai_generated", False)) else ""
        phone = m.get("contact_phone", "") or m.get("phone", "")
        buyer_tag = " [BUYER]" if phone in buyer_phones else ""
        chat_context.append(f"{role}{is_ai}{buyer_tag}: {text}")

    chat_text = "\n".join(chat_context)
    # Estimate token usage: ~1.3 tokens per word for Hinglish text
    chat_words = len(chat_text.split())
    est_chat_tokens = int(chat_words * 1.3)
    logger.info(
        f"[wwbun-learn] Sending {len(pair_messages)} pair messages to Claude "
        f"(from {len(filtered)} filtered, {len(messages)} total buffer). "
        f"Chat text: {chat_words} words ≈ {est_chat_tokens} tokens, {len(chat_text)} chars"
    )
    buyer_count = len(buyer_phones)

    # NOTE: We do NOT send current traits/phrases to Claude anymore.
    # The 194KB prompt.json was adding ~8000+ tokens per call for dedup that
    # already happens in apply_knowledge_updates() AFTER Claude returns.

    prompt = f"""Analyze these WhatsApp conversations. Learn from KETU's MANUAL messages only (NOT [AI] tagged). Use CUSTOMER messages as CONTEXT to understand why Ketu replied that way.
{f"IMPORTANT: {buyer_count} conversations are from BUYERS (marked [BUYER]). Pay EXTRA attention to these — learn what led to a sale." if buyer_count > 0 else ""}

Messages:
{chat_text}

Do NOT extract price info or product details (materials, colors, sizes, GSM) — catalog is the source of truth.

Extract JSON:
1. "style_patterns": Ketu's phrases, greetings, tone, typical replies
2. "new_faqs": Customer Q + Ketu A pairs (NOT price/product FAQs)
3. "business_updates": New policies, offers, shipping info
4. "prompt_evolution": {{"new_traits": [], "new_phrases": [], "new_rules": [], "example_conversations": [{{"customer": "q", "reply": "a"}}]}}
5. "sales_patterns": [{{"pattern": "what worked", "example": "msg"}}] (from [BUYER] chats only, or [])

Return ONLY valid JSON."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text

        # Track API cost
        from core.cost_tracker import track_api_cost
        track_api_cost(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            source="wwbun-learning",
        )

        # Log payload for dashboard debug
        from core.cloud_payload_log import log_cloud_payload
        log_cloud_payload(
            source="wwbun-learning",
            prompt_text=prompt,
            model="claude-haiku-4-5-20251001",
            max_tokens=1200,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            pair_count=len(pair_messages),
            message_count=len(filtered),
        )

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["filter_stats"] = filter_stats
            result["quality_messages"] = [(m.get("content", "") or m.get("text", "") or m.get("body", "")) for m in filtered if _is_owner_msg(m) and not m.get("is_ai_generated")]
            return result
        return {"status": "parse_error", "raw": result_text, "filter_stats": filter_stats}

    except Exception as e:
        logger.error(f"Knowledge extraction error: {e}")
        return {"status": "error", "detail": str(e), "filter_stats": filter_stats}


def apply_knowledge_updates(updates: dict) -> dict:
    """Apply extracted knowledge to the JSON files."""
    applied = []

    # SKIP price_updates and product_updates from chat learning —
    # the catalog is the single source of truth for prices and product info.
    for skip_key in ("price_updates", "new_products", "product_updates"):
        if updates.pop(skip_key, None):
            logger.info(f"[apply] Skipped '{skip_key}' from chat — catalog is source of truth")

    # Convert sales_patterns to style_patterns (tagged with [Sales])
    sales_patterns = updates.pop("sales_patterns", [])
    if sales_patterns and isinstance(sales_patterns, list):
        sales_style = [
            f"[Sales] {sp['pattern']}" for sp in sales_patterns
            if isinstance(sp, dict) and sp.get("pattern")
        ]
        if sales_style:
            existing = updates.get("style_patterns", [])
            if isinstance(existing, list):
                updates["style_patterns"] = existing + sales_style
            else:
                updates["style_patterns"] = sales_style
            logger.info(f"[apply] Converted {len(sales_style)} sales patterns to style patterns")

    # Update FAQ
    if updates.get("new_faqs"):
        # First: detect contradictions with existing FAQs
        from learner.faq_validator import detect_contradiction
        contradiction_result = detect_contradiction(updates["new_faqs"])
        if contradiction_result.get("contradictions_found", 0) > 0:
            for r in contradiction_result.get("replaced", []):
                applied.append(f"FAQ replaced (contradiction): {r['old_question']}")

        # Load from DB first (source of truth), fallback to file
        faq_data = None
        try:
            from core.database import is_db_available, load_knowledge_from_db
            if is_db_available():
                faq_data = load_knowledge_from_db("faq")
        except Exception:
            pass
        if not faq_data:
            faq_path = KNOWLEDGE_DIR / "faq.json"
            with open(faq_path, "r", encoding="utf-8") as f:
                faq_data = json.load(f)

        existing_questions = {faq["question"].lower() for faq in faq_data["faqs"]}

        for faq in updates["new_faqs"]:
            q = faq.get("question", "")
            a = faq.get("answer", "")
            if q and a and q.lower() not in existing_questions:
                from datetime import datetime, timezone, timedelta
                ist = timezone(timedelta(hours=5, minutes=30))
                faq_data["faqs"].append({
                    "question": q,
                    "answer": a,
                    "keywords": faq.get("keywords", []),
                    "source": "auto_learned",
                    "added_at": datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST"),
                })
                applied.append(f"New FAQ: {q}")

        # DB first, then file
        try:
            from core.database import is_db_available, save_knowledge
            if is_db_available():
                save_knowledge("faq", faq_data)
        except Exception:
            pass
        faq_path = KNOWLEDGE_DIR / "faq.json"
        with open(faq_path, "w", encoding="utf-8") as f:
            json.dump(faq_data, f, indent=2, ensure_ascii=False)

    # Update style patterns
    if updates.get("style_patterns"):
        # Load from DB first (source of truth), fallback to file
        style_data = None
        try:
            from core.database import is_db_available, load_knowledge_from_db
            if is_db_available():
                style_data = load_knowledge_from_db("style")
        except Exception:
            pass
        if not style_data:
            style_path = KNOWLEDGE_DIR / "style.json"
            with open(style_path, "r", encoding="utf-8") as f:
                style_data = json.load(f)

        patterns = updates["style_patterns"]
        if isinstance(patterns, list):
            for pattern in patterns:
                if isinstance(pattern, str) and pattern not in style_data.get("tone_words", []):
                    style_data.setdefault("learned_patterns", []).append(pattern)
                    applied.append(f"New pattern: {pattern}")
        elif isinstance(patterns, dict):
            style_data.setdefault("learned_patterns", []).append(patterns)
            applied.append("New style patterns learned")

        # DB first, then file
        try:
            from core.database import is_db_available, save_knowledge
            if is_db_available():
                save_knowledge("style", style_data)
        except Exception:
            pass
        style_path = KNOWLEDGE_DIR / "style.json"
        with open(style_path, "w", encoding="utf-8") as f:
            json.dump(style_data, f, indent=2, ensure_ascii=False)

    # Evolve the system prompt
    if updates.get("prompt_evolution"):
        evolution = updates["prompt_evolution"]
        try:
            # Load from DB first (source of truth), fallback to file
            prompt_data = None
            try:
                from core.database import is_db_available, load_knowledge_from_db
                if is_db_available():
                    prompt_data = load_knowledge_from_db("prompt")
            except Exception:
                pass

            if not prompt_data:
                with open(PROMPT_FILE, "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)

            # Collect all existing values for dedup
            all_traits = set(
                t.lower() for t in prompt_data.get("personality_traits", [])
                + prompt_data.get("evolved_traits", [])
            )
            all_phrases = set(
                p.lower() for p in prompt_data.get("signature_phrases", [])
                + prompt_data.get("evolved_phrases", [])
            )
            all_rules = set(
                r.lower() for r in prompt_data.get("reply_rules", [])
                + prompt_data.get("evolved_rules", [])
            )

            # Add new traits
            for trait in evolution.get("new_traits", []):
                if isinstance(trait, str) and trait.lower() not in all_traits:
                    prompt_data.setdefault("evolved_traits", []).append(trait)
                    applied.append(f"Evolved trait: {trait}")

            # Add new phrases
            for phrase in evolution.get("new_phrases", []):
                if isinstance(phrase, str) and phrase.lower() not in all_phrases:
                    prompt_data.setdefault("evolved_phrases", []).append(phrase)
                    applied.append(f"Evolved phrase: {phrase}")

            # Add new rules
            for rule in evolution.get("new_rules", []):
                if isinstance(rule, str) and rule.lower() not in all_rules:
                    prompt_data.setdefault("evolved_rules", []).append(rule)
                    applied.append(f"Evolved rule: {rule}")

            # Add example conversations to style.json
            new_examples = evolution.get("example_conversations", [])
            if new_examples:
                # Load from DB first (source of truth), fallback to file
                style_data = None
                try:
                    from core.database import is_db_available, load_knowledge_from_db
                    if is_db_available():
                        style_data = load_knowledge_from_db("style")
                except Exception:
                    pass
                if not style_data:
                    style_path = KNOWLEDGE_DIR / "style.json"
                    with open(style_path, "r", encoding="utf-8") as f:
                        style_data = json.load(f)

                existing_replies = {
                    ex["reply"].lower()[:50]
                    for ex in style_data.get("example_conversations", [])
                }

                for ex in new_examples:
                    if (isinstance(ex, dict)
                            and ex.get("customer") and ex.get("reply")
                            and ex["reply"].lower()[:50] not in existing_replies):
                        style_data.setdefault("example_conversations", []).append(ex)
                        applied.append(f"New example: {ex['customer'][:40]}...")

                # DB first, then file
                try:
                    from core.database import is_db_available, save_knowledge
                    if is_db_available():
                        save_knowledge("style", style_data)
                except Exception:
                    pass
                style_path = KNOWLEDGE_DIR / "style.json"
                with open(style_path, "w", encoding="utf-8") as f:
                    json.dump(style_data, f, indent=2, ensure_ascii=False)

            # Log evolution event
            from datetime import datetime, timezone, timedelta
            ist = timezone(timedelta(hours=5, minutes=30))
            prompt_data.setdefault("evolution_log", []).append({
                "timestamp": datetime.now(ist).isoformat(),
                "changes": [a for a in applied if a.startswith("Evolved")],
            })
            prompt_data["version"] = prompt_data.get("version", 1) + 1

            # DB first, then file
            try:
                from core.database import is_db_available, save_knowledge
                if is_db_available():
                    save_knowledge("prompt", prompt_data)
            except Exception:
                pass
            with open(PROMPT_FILE, "w", encoding="utf-8") as f:
                json.dump(prompt_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Prompt evolved: {[a for a in applied if a.startswith('Evolved')]}")

        except Exception as e:
            logger.error(f"Prompt evolution error: {e}")

    # Persist to GitHub (backup)
    if applied:
        from core.git_persist import persist_knowledge_files
        persist_knowledge_files(source="knowledge-update")

    return {"applied": applied, "count": len(applied)}


    # _save_knowledge_to_db_direct removed — each section now saves to DB immediately


def learn_conversation_enders(messages: list[dict], owner_user_id: str) -> dict:
    """Learn conversation-ending patterns from real chat behavior.

    Scans wwbun messages to find patterns where:
    - Customer sent a message (the potential "ender")
    - Ketu did NOT reply after it (conversation ended naturally)

    These patterns become learned enders — Digital Ketu will stop replying
    when it sees similar messages, saving cost and being more natural.

    Also detects false positives: messages that LOOK like enders but Ketu
    actually did reply to (learned_non_enders).
    """
    if not messages or len(messages) < 2:
        return {"status": "not_enough_messages", "learned": 0}

    ENDERS_FILE = KNOWLEDGE_DIR / "conversation_enders.json"

    # Load existing enders knowledge
    try:
        from core.database import is_db_available, load_knowledge_from_db, save_knowledge
        enders_data = None
        if is_db_available():
            enders_data = load_knowledge_from_db("conversation_enders")
        if not enders_data:
            with open(ENDERS_FILE, "r", encoding="utf-8") as f:
                enders_data = json.load(f)
    except Exception:
        enders_data = {
            "hardcoded_enders": [],
            "learned_enders": [],
            "learned_non_enders": [],
            "learning_log": [],
        }

    existing_learned = set(e.lower() if isinstance(e, str) else e.get("pattern", "").lower()
                          for e in enders_data.get("learned_enders", []))
    existing_non_enders = set(e.lower() if isinstance(e, str) else e.get("pattern", "").lower()
                              for e in enders_data.get("learned_non_enders", []))

    new_enders = []
    new_non_enders = []

    # NEVER learn greetings as enders — these are conversation starters, not enders.
    # Ketu may not reply to "hi" because he's busy, but AI must always reply.
    _never_learn_as_ender = {
        "hi", "hii", "hiii", "hiiii", "hello", "hey", "heyy", "heyyy",
        "hlo", "helo", "hllo", "helloo", "hellooo",
        "namaste", "namaskar", "namaskaar",
        "good morning", "good afternoon", "good evening", "good night",
        "gm", "gn", "sir", "bhai", "bhaiya", "bro", "boss",
    }

    # Walk through messages looking for conversation gaps
    for i in range(len(messages) - 1):
        msg = messages[i]
        next_msg = messages[i + 1]

        # We want: customer message followed by another customer message (Ketu didn't reply)
        # or: customer message that was the last in the conversation
        is_customer_msg = msg.get("sender_id") != owner_user_id
        next_is_customer = next_msg.get("sender_id") != owner_user_id
        next_is_owner = next_msg.get("sender_id") == owner_user_id

        if not is_customer_msg:
            continue

        content = msg.get("content", "").strip()
        if not content or len(content) > 50:  # Only short messages can be enders
            continue

        content_lower = content.lower().rstrip("!.,?").strip()

        # Never learn greetings as enders
        if content_lower in _never_learn_as_ender:
            continue

        # Pattern 1: Customer said something, then ANOTHER customer msg came (Ketu stayed silent)
        # This means Ketu chose not to reply — it's a conversation ender
        if next_is_customer and content_lower not in existing_learned:
            # Don't learn question-like messages as enders
            if "?" not in content and not any(w in content_lower.split() for w in
                    ["kya", "kab", "kaise", "kitna", "kitne", "kaha", "price", "rate",
                     "sample", "order", "send", "bhej", "batao", "bata", "how", "what", "when"]):
                new_enders.append({
                    "pattern": content_lower,
                    "original": content,
                    "context": f"Customer said this, Ketu didn't reply",
                })

        # Pattern 2: Customer said something that LOOKS like an ender, but Ketu DID reply
        # This is a false positive — add to non-enders list
        hardcoded = set(enders_data.get("hardcoded_enders", []))
        if next_is_owner and content_lower in hardcoded and content_lower not in existing_non_enders:
            new_non_enders.append({
                "pattern": content_lower,
                "original": content,
                "ketu_replied": next_msg.get("content", "")[:80],
                "context": "Looked like ender but Ketu replied",
            })

    # Also check the last message in the batch
    last_msg = messages[-1]
    if last_msg.get("sender_id") != owner_user_id:
        content = last_msg.get("content", "").strip()
        content_lower = content.lower().rstrip("!.,?").strip()
        if (content and len(content) <= 50
                and content_lower not in existing_learned
                and content_lower not in _never_learn_as_ender):
            if "?" not in content:
                new_enders.append({
                    "pattern": content_lower,
                    "original": content,
                    "context": "Last message in conversation — Ketu didn't reply",
                })

    # Deduplicate
    seen = set()
    unique_enders = []
    for e in new_enders:
        if e["pattern"] not in seen and e["pattern"] not in existing_non_enders:
            seen.add(e["pattern"])
            unique_enders.append(e)

    # Save updates
    if unique_enders or new_non_enders:
        from datetime import datetime, timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")

        for e in unique_enders:
            e["learned_at"] = now
            enders_data.setdefault("learned_enders", []).append(e)

        for e in new_non_enders:
            e["learned_at"] = now
            enders_data.setdefault("learned_non_enders", []).append(e)
            # Remove from hardcoded if Ketu actually replies to it
            if e["pattern"] in set(enders_data.get("hardcoded_enders", [])):
                enders_data["hardcoded_enders"] = [
                    h for h in enders_data["hardcoded_enders"] if h.lower() != e["pattern"]
                ]

        enders_data.setdefault("learning_log", []).append({
            "timestamp": now,
            "new_enders": len(unique_enders),
            "new_non_enders": len(new_non_enders),
            "examples": [e["pattern"] for e in unique_enders[:5]],
        })

        # Save to DB first, then file
        try:
            from core.database import is_db_available, save_knowledge
            if is_db_available():
                save_knowledge("conversation_enders", enders_data)
        except Exception:
            pass

        with open(ENDERS_FILE, "w", encoding="utf-8") as f:
            json.dump(enders_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[Enders] Learned {len(unique_enders)} new enders, {len(new_non_enders)} non-enders")

    return {
        "status": "ok",
        "new_enders": len(unique_enders),
        "new_non_enders": len(new_non_enders),
        "examples": [e["pattern"] for e in unique_enders[:10]],
        "non_ender_examples": [e["pattern"] for e in new_non_enders[:5]],
    }


def learn_repeat_buyer_patterns(messages: list[dict], owner_user_id: str) -> dict:
    """Learn how Ketu talks to repeat/returning buyers.

    Scans conversations to find customers who bought multiple times, then
    extracts Ketu's reply patterns for these regulars. Stored separately
    so the AI knows the difference between first-time and repeat buyer tone.

    Pure Python — zero AI cost. Just keyword matching + pattern extraction.
    """
    from core.customer_memory import get_profile, STAGE_BOUGHT, STAGE_REPEAT

    REPEAT_STYLE_FILE = KNOWLEDGE_DIR / "repeat_buyer_style.json"

    # Load existing repeat buyer style
    repeat_style = None
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            repeat_style = load_knowledge_from_db("repeat_buyer_style")
    except Exception:
        pass
    if not repeat_style:
        try:
            with open(REPEAT_STYLE_FILE, "r", encoding="utf-8") as f:
                repeat_style = json.load(f)
        except Exception:
            repeat_style = {
                "repeat_buyer_replies": [],
                "returning_buyer_replies": [],
                "greeting_patterns": [],
                "learning_log": [],
            }

    # Group messages by customer phone
    conversations: dict[str, list] = {}
    for msg in messages:
        phone = msg.get("contact_phone", "") or msg.get("phone", "")
        if not phone:
            continue
        conversations.setdefault(phone, []).append(msg)

    new_repeat_replies = []
    new_returning_replies = []

    existing_replies = set(
        r.get("reply", "").lower()[:50] if isinstance(r, dict) else str(r).lower()[:50]
        for r in repeat_style.get("repeat_buyer_replies", [])
        + repeat_style.get("returning_buyer_replies", [])
    )

    for phone, conv_messages in conversations.items():
        profile = get_profile(phone)
        stage = profile.get("stage", "new")

        # Only learn from conversations with bought/repeat customers
        if stage not in (STAGE_BOUGHT, STAGE_REPEAT):
            continue

        # Extract Ketu's manual replies to this repeat buyer
        for msg in conv_messages:
            if msg.get("sender_id") != owner_user_id:
                continue
            if msg.get("is_ai_generated", False):
                continue

            content = msg.get("content", "").strip()
            if not content or len(content) < 5:
                continue

            # Skip junk
            if is_junk_message(content):
                continue

            reply_lower = content.lower()[:50]
            if reply_lower in existing_replies:
                continue

            entry = {
                "reply": content,
                "customer_phone_last4": phone[-4:] if len(phone) >= 4 else phone,
                "purchase_count": profile.get("purchase_count", 1),
            }

            days_since = profile.get("days_since_last_purchase", 0)
            if days_since >= 30:
                new_returning_replies.append(entry)
            else:
                new_repeat_replies.append(entry)
            existing_replies.add(reply_lower)

    # Save if new patterns found
    if new_repeat_replies or new_returning_replies:
        from datetime import datetime, timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist).strftime("%d %b %Y, %I:%M %p IST")

        repeat_style.setdefault("repeat_buyer_replies", []).extend(new_repeat_replies)
        repeat_style.setdefault("returning_buyer_replies", []).extend(new_returning_replies)

        # Keep only last 50 of each to avoid bloat
        repeat_style["repeat_buyer_replies"] = repeat_style["repeat_buyer_replies"][-50:]
        repeat_style["returning_buyer_replies"] = repeat_style["returning_buyer_replies"][-50:]

        repeat_style.setdefault("learning_log", []).append({
            "timestamp": now,
            "new_repeat": len(new_repeat_replies),
            "new_returning": len(new_returning_replies),
        })

        # Save to DB + file
        try:
            from core.database import is_db_available, save_knowledge
            if is_db_available():
                save_knowledge("repeat_buyer_style", repeat_style)
        except Exception:
            pass

        with open(REPEAT_STYLE_FILE, "w", encoding="utf-8") as f:
            json.dump(repeat_style, f, indent=2, ensure_ascii=False)

        logger.info(f"[RepeatBuyer] Learned {len(new_repeat_replies)} repeat replies, {len(new_returning_replies)} returning replies")

    return {
        "status": "ok",
        "new_repeat_replies": len(new_repeat_replies),
        "new_returning_replies": len(new_returning_replies),
        "total_repeat_patterns": len(repeat_style.get("repeat_buyer_replies", [])),
        "total_returning_patterns": len(repeat_style.get("returning_buyer_replies", [])),
    }


def detect_bought_customers_from_chat(messages: list[dict], owner_user_id: str) -> dict:
    """Detect customers who already bought by scanning BOTH owner and customer messages.

    When Ketu sends a bill, discusses dispatch, or confirms payment — the customer
    has bought. Mark them as 'bought' so follow-up system skips them.

    Pure Python keyword matching — zero AI cost.
    """
    from core.customer_memory import get_profile, STAGE_BOUGHT, STAGE_REPEAT

    # Signals that a purchase is complete (from EITHER side of conversation)
    bought_signals = {
        "bill", "invoice", "receipt", "billno", "bill no",
        "dispatch", "dispatched", "shipped", "ship ho gaya", "ship kar diya",
        "payment done", "payment ho gaya", "paid", "pay kar diya", "payment received",
        "order confirm", "order ho gaya", "order done", "order ready",
        "parcel", "tracking", "delivery", "deliver ho", "courier",
        "received", "mil gaya", "aa gaya", "godam", "warehouse",
        "packed", "packing done", "ready for dispatch",
    }

    # Group messages by conversation (customer phone)
    # wwbun messages have contact_phone or we derive from conversation context
    conversations: dict[str, list] = {}
    for msg in messages:
        # Try to identify the customer phone from the message
        phone = msg.get("contact_phone", "") or msg.get("phone", "")
        if not phone:
            continue
        conversations.setdefault(phone, []).append(msg)

    updated = []
    for phone, conv_messages in conversations.items():
        # Check ALL messages in conversation (both owner and customer)
        has_bought_signal = False
        for msg in conv_messages:
            content = (msg.get("content", "") or "").lower()
            if any(signal in content for signal in bought_signals):
                has_bought_signal = True
                break

        if has_bought_signal:
            profile = get_profile(phone)
            current_stage = profile.get("stage", "new")
            if current_stage not in (STAGE_BOUGHT, STAGE_REPEAT):
                profile["stage"] = STAGE_BOUGHT
                from core.customer_memory import _save_profile_to_db, _profiles
                _profiles[phone] = profile
                _save_profile_to_db(phone, "", profile)
                updated.append(phone[-4:] if len(phone) >= 4 else phone)
                logger.info(f"[BoughtDetect] Marked {phone[-4:]} as bought (from chat signals)")

    return {
        "status": "ok",
        "customers_marked_bought": len(updated),
        "phones": updated,
    }
