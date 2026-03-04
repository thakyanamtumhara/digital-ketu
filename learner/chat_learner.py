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
_MIN_WORDS_CUSTOMER = 3  # Customer messages need 3+ words for context


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

        # Word count check
        word_count = len(text.split())
        is_owner = owner_value and owner_value.lower() in str(m.get(owner_key, "")).lower()

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

    # Load current prompt config for context
    prompt_config = _load_prompt_config()
    current_traits = prompt_config.get("personality_traits", []) + prompt_config.get("evolved_traits", [])
    current_phrases = prompt_config.get("signature_phrases", []) + prompt_config.get("evolved_phrases", [])

    prompt = f"""Analyze these WhatsApp chat messages from Ketu (the business owner of Sale91.com / Own Knitted Blank Wears).

IMPORTANT: Analyze BOTH customer questions AND Ketu's replies together — the customer question gives context for WHY Ketu replied that way. But only learn FROM Ketu's messages.
Ketu's name in chat: {ketu_name}

Chat messages:
{chat_sample}

CURRENT personality traits already known:
{json.dumps(current_traits, ensure_ascii=False)}

CURRENT signature phrases already known:
{json.dumps(current_phrases, ensure_ascii=False)}

Extract the following (in JSON format):
1. "new_products": Any new products mentioned with prices
2. "price_updates": Any price changes mentioned
3. "style_patterns": How Ketu talks — phrases, greetings, closing patterns
4. "new_faqs": Customer Q + Ketu's A pairs (use customer question for context)
5. "business_updates": Any new business info (offers, policies, etc.)
6. "prompt_evolution": {{
     "new_traits": ["NEW personality traits you noticed that are NOT already in the list above"],
     "new_phrases": ["NEW signature phrases/words Ketu uses repeatedly that are NOT already known"],
     "new_rules": ["NEW reply rules/patterns you noticed — how Ketu handles specific situations"],
     "example_conversations": [{{"customer": "what customer asked", "reply": "what Ketu replied"}}]
   }}

CRITICAL for prompt_evolution:
- Only add traits/phrases/rules that are genuinely NEW (not already in current list)
- Look for Ketu's UNIQUE way of talking — his catchphrases, his way of convincing, his humor
- Notice how he handles objections, how he upsells, how he closes deals
- If you find a pattern Ketu uses 2+ times, that's a signature move — capture it

Return ONLY valid JSON. If nothing new found, return empty arrays/objects."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
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

    # Filter: only messages sent by owner, exclude AI-generated ones
    manual_messages = [
        m for m in messages
        if m.get("sender_id") == owner_user_id
        and not m.get("is_ai_generated", False)
    ]

    if not manual_messages:
        return {"status": "no_manual_messages", "updates": [], "filter_stats": {"total": len(messages), "kept": 0, "junk": 0, "too_short": 0}}

    # Smart filter: remove junk messages BEFORE sending to Claude
    filtered, filter_stats = filter_messages(messages, owner_key="sender_id", owner_value=owner_user_id)

    # Include customer messages for context (to understand what Ketu was replying to)
    # but only learn FROM Ketu's messages
    chat_context = []
    for m in filtered[:200]:
        role = "KETU" if m.get("sender_id") == owner_user_id else "CUSTOMER"
        is_ai = " [AI]" if m.get("is_ai_generated") else ""
        chat_context.append(f"{role}{is_ai}: {m.get('content', '')}")

    chat_text = "\n".join(chat_context)

    # Load current prompt config for context
    prompt_config = _load_prompt_config()
    current_traits = prompt_config.get("personality_traits", []) + prompt_config.get("evolved_traits", [])
    current_phrases = prompt_config.get("signature_phrases", []) + prompt_config.get("evolved_phrases", [])

    prompt = f"""Analyze these WhatsApp conversations. Learn from KETU's MANUAL messages only (NOT [AI] tagged). Use CUSTOMER messages as CONTEXT to understand why Ketu replied that way.

Messages:
{chat_text}

CURRENT personality traits already known:
{json.dumps(current_traits, ensure_ascii=False)}

CURRENT signature phrases already known:
{json.dumps(current_phrases, ensure_ascii=False)}

Extract in JSON format:
1. "style_patterns": How Ketu types — his phrases, greetings, tone, typical replies
2. "price_updates": Any prices Ketu mentioned
3. "new_faqs": Customer Q + Ketu's A pairs (use customer question for context)
4. "business_updates": Any new policies, offers, shipping info
5. "product_updates": Any new product info Ketu shared
6. "prompt_evolution": {{
     "new_traits": ["NEW personality traits NOT already known"],
     "new_phrases": ["NEW signature phrases/words NOT already known"],
     "new_rules": ["NEW reply patterns — how Ketu handles specific situations"],
     "example_conversations": [{{"customer": "question", "reply": "Ketu's reply"}}]
   }}

CRITICAL for prompt_evolution:
- Only add genuinely NEW traits/phrases/rules (not duplicates)
- Capture Ketu's unique selling style, humor, objection handling
- If Ketu uses a phrase 2+ times, it's a signature — add it

Return ONLY valid JSON."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
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

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            result["filter_stats"] = filter_stats
            result["quality_messages"] = [m.get("content", "") for m in filtered if m.get("sender_id") == owner_user_id and not m.get("is_ai_generated")]
            return result
        return {"status": "parse_error", "raw": result_text, "filter_stats": filter_stats}

    except Exception as e:
        logger.error(f"Knowledge extraction error: {e}")
        return {"status": "error", "detail": str(e), "filter_stats": filter_stats}


def apply_knowledge_updates(updates: dict) -> dict:
    """Apply extracted knowledge to the JSON files."""
    applied = []

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
