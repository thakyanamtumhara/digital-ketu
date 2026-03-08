"""Real-time conversation learner for Digital Ketu.

Three learning modes:
1. Conversation Learning — after every N replies, analyze patterns and learn
2. Correction Learning — when Ketu overrides AI reply, learn from the correction
3. Voice Note Learning — transcribe and learn from Ketu's voice notes

These make Digital Ketu progressively more like the real Ketu.
"""

import json
import logging
import re
import threading
from collections import deque
from datetime import datetime, timezone, timedelta

from anthropic import Anthropic

from core.config import settings, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

# --- Conversation Buffer ---
# Stores recent customer Q + AI reply pairs for batch learning
_conversation_buffer: deque[dict] = deque(maxlen=50)
_LEARN_EVERY_N = 10  # Analyze after every 10 conversations
_conversation_count = 0
_learning_lock = threading.Lock()
_buffer_loaded = False


def _load_buffer_from_db():
    """Restore conversation buffer from DB on first call (survives deploys)."""
    global _conversation_buffer, _conversation_count, _buffer_loaded
    if _buffer_loaded:
        return
    _buffer_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get("conversation_buffer")
            if data and isinstance(data, dict):
                for entry in data.get("buffer", []):
                    _conversation_buffer.append(entry)
                _conversation_count = data.get("count", 0)
    except Exception as e:
        logger.warning(f"[Buffer] DB load failed: {e}")


def _save_buffer_to_db():
    """Persist conversation buffer to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("conversation_buffer", {
                "buffer": list(_conversation_buffer),
                "count": _conversation_count,
            })
    except Exception as e:
        logger.warning(f"[Buffer] DB save failed: {e}")


def buffer_conversation(
    customer_message: str,
    ai_reply: str,
    customer_name: str = "",
    customer_phone: str = "",
):
    """Buffer a conversation pair for background learning.

    Called after every AI reply. When buffer hits threshold,
    triggers batch analysis. Junk messages are filtered out
    before buffering to keep learning data high-quality.

    Conversations from customers who later bought get priority=high,
    making them more valuable for learning what converts.
    """
    global _conversation_count

    _load_buffer_from_db()

    # Filter out junk messages — no point learning from "ok", "hmm", emojis etc.
    from learner.chat_learner import is_junk_message, is_media_only_message, has_business_intent
    if is_junk_message(customer_message) and is_junk_message(ai_reply):
        logger.debug(f"[Buffer] Skipped junk pair: '{customer_message[:30]}' / '{ai_reply[:30]}'")
        return
    # If customer sent junk but AI gave a real reply, still skip —
    # the AI reply to "ok" or "👍" has no learning value
    if is_junk_message(customer_message):
        logger.debug(f"[Buffer] Skipped junk customer msg: '{customer_message[:30]}'")
        return
    # If AI reply is media-only ([Image], [Document]), skip — can't learn text from media
    if is_media_only_message(ai_reply):
        logger.debug(f"[Buffer] Skipped media-only reply: '{ai_reply[:30]}'")
        return
    # If customer sent media-only (image/document) and AI just acknowledged, skip
    if is_media_only_message(customer_message):
        logger.debug(f"[Buffer] Skipped media-only customer msg: '{customer_message[:30]}'")
        return
    # Only buffer conversations with business intent — skip random chit-chat
    if not has_business_intent(customer_message):
        logger.debug(f"[Buffer] Skipped no-business-intent: '{customer_message[:30]}'")
        return

    # Check if this customer already bought — conversations that led to sales
    # are the most valuable learning data (priority=high)
    priority = "normal"
    if customer_phone:
        try:
            from core.customer_memory import get_profile, STAGE_BOUGHT, STAGE_REPEAT
            profile = get_profile(customer_phone)
            stage = profile.get("stage", "new")
            if stage in (STAGE_BOUGHT, STAGE_REPEAT):
                priority = "high"
                logger.info(f"[Buffer] HIGH priority pair from buyer {customer_phone[-4:]}")
        except Exception:
            pass

    _conversation_buffer.append({
        "customer": customer_message,
        "reply": ai_reply,
        "name": customer_name,
        "phone_last4": customer_phone[-4:] if customer_phone else "",
        "time": datetime.now(IST).strftime("%I:%M %p"),
        "priority": priority,
    })
    _conversation_count += 1

    _save_buffer_to_db()

    # Trigger batch learning every N conversations
    if _conversation_count >= _LEARN_EVERY_N:
        _conversation_count = 0
        # Run in background thread to not block the reply
        thread = threading.Thread(target=_batch_learn_from_conversations, daemon=True)
        thread.start()


def _batch_learn_from_conversations():
    """Analyze buffered conversations and extract patterns.

    This runs in a background thread — does NOT block customer replies.
    Focuses on:
    - Common question patterns (potential new FAQs)
    - How the AI is replying (quality check)
    - Recurring topics (what customers ask most)
    """
    if not _learning_lock.acquire(blocking=False):
        return  # Another learning thread is running

    try:
        conversations = list(_conversation_buffer)
        if len(conversations) < 5:
            return

        client = Anthropic(api_key=settings.anthropic_api_key)

        # Prioritize conversations from buyers — these are the most valuable
        # because they show what messaging style converts customers to sales
        high_priority = [c for c in conversations if c.get("priority") == "high"]
        normal = [c for c in conversations if c.get("priority") != "high"]

        # Include ALL high-priority (buyer) conversations + fill rest with normal
        # This ensures buyer conversations are always analyzed
        max_convos = 20
        selected = high_priority[:max_convos]
        remaining_slots = max_convos - len(selected)
        if remaining_slots > 0:
            selected.extend(normal[-remaining_slots:])

        if not selected:
            selected = conversations[-max_convos:]

        buyer_count = len([c for c in selected if c.get("priority") == "high"])

        conv_text = "\n".join(
            f"Customer ({c['name'] or c['phone_last4']}){' [BUYER]' if c.get('priority') == 'high' else ''}: {c['customer']}\n"
            f"Digital Ketu: {c['reply']}"
            for c in selected
        )

        prompt = f"""Analyze these recent conversations between customers and Digital Ketu (AI twin of Ketu, a t-shirt manufacturer).

{f"IMPORTANT: {buyer_count} conversations are from customers who BOUGHT (marked [BUYER]). Pay EXTRA attention to these — learn what messaging style, tone, and answers led to successful sales." if buyer_count > 0 else ""}

Conversations:
{conv_text}

Extract (JSON only):
1. "common_questions": Questions asked 2+ times (potential FAQ candidates)
   Format: [{{"question": "...", "suggested_answer": "...", "frequency": N}}]
2. "quality_issues": Any replies that seem wrong, too generic, or miss the point
   Format: [{{"customer_asked": "...", "ai_replied": "...", "issue": "..."}}]
3. "hot_topics": Top 3 topics customers are asking about right now
   Format: ["topic1", "topic2", "topic3"]
4. "sales_patterns": What reply patterns/phrases appeared in conversations that led to a sale (from [BUYER] conversations only)
   Format: [{{"pattern": "...", "example": "..."}}] or []

Return ONLY valid JSON. If nothing notable, return empty arrays."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text

        # Track API cost
        from core.cost_tracker import track_api_cost
        track_api_cost(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            source="realtime-analysis",
        )

        # Log payload for dashboard debug
        from core.cloud_payload_log import log_cloud_payload
        log_cloud_payload(
            source="realtime-analysis",
            prompt_text=prompt,
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            pair_count=len(selected),
        )

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)

        if json_match:
            analysis = json.loads(json_match.group())

            # Auto-add frequent questions as FAQs
            common_qs = analysis.get("common_questions", [])
            if common_qs:
                from learner.chat_learner import apply_knowledge_updates
                from core.knowledge import invalidate_cache
                from core.activity_log import log_activity

                new_faqs = [
                    {"question": q["question"], "answer": q["suggested_answer"]}
                    for q in common_qs
                    if q.get("frequency", 0) >= 2
                ]

                if new_faqs:
                    result = apply_knowledge_updates({"new_faqs": new_faqs})
                    if result.get("count", 0) > 0:
                        invalidate_cache()
                        log_activity(
                            source="realtime-learner",
                            action="auto-faq",
                            details={
                                "faqs_added": result.get("applied", []),
                                "from_conversations": len(conversations),
                            },
                            items_count=result.get("count", 0),
                        )
                        logger.info(f"Realtime learner: Added {result['count']} FAQs from conversation patterns")

            # Log quality issues for dashboard visibility
            quality_issues = analysis.get("quality_issues", [])
            if quality_issues:
                from core.activity_log import log_activity
                log_activity(
                    source="realtime-learner",
                    action="quality-check",
                    details={
                        "issues_found": len(quality_issues),
                        "issues": quality_issues[:3],
                        "hot_topics": analysis.get("hot_topics", []),
                    },
                    items_count=0,
                )

            # Learn sales patterns from buyer conversations → style patterns
            sales_patterns = analysis.get("sales_patterns", [])
            if sales_patterns:
                from learner.chat_learner import apply_knowledge_updates
                from core.knowledge import invalidate_cache
                from core.activity_log import log_activity

                style_patterns = [
                    f"[Sales] {sp['pattern']}" for sp in sales_patterns
                    if isinstance(sp, dict) and sp.get("pattern")
                ]
                if style_patterns:
                    result = apply_knowledge_updates({"style_patterns": style_patterns})
                    if result.get("count", 0) > 0:
                        invalidate_cache()
                        log_activity(
                            source="realtime-learner",
                            action="sales-patterns",
                            details={
                                "patterns_learned": [sp.get("pattern", "") for sp in sales_patterns[:5]],
                                "from_buyer_conversations": buyer_count,
                            },
                            items_count=result.get("count", 0),
                        )
                        logger.info(
                            f"Realtime learner: Learned {result['count']} sales patterns "
                            f"from {buyer_count} buyer conversations"
                        )

    except Exception as e:
        logger.error(f"Realtime conversation learning error: {e}")
    finally:
        _learning_lock.release()


# --- Correction Learning ---

# Track correction patterns in DB — when same mistake type repeats 3+ times,
# auto-generate a stronger rule to prevent it permanently
_CORRECTION_PATTERN_KEY = "correction_patterns"


def _load_correction_patterns() -> dict:
    """Load correction pattern history from DB."""
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get(_CORRECTION_PATTERN_KEY)
            if data and isinstance(data, dict):
                return data
    except Exception:
        pass
    return {"patterns": {}, "total_corrections": 0}


def _save_correction_patterns(data: dict):
    """Save correction pattern history to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set(_CORRECTION_PATTERN_KEY, data)
    except Exception:
        pass


def _track_correction_pattern(what_went_wrong: str, customer_message: str, ketu_correction: str):
    """Track what types of mistakes the AI keeps making.

    When the same mistake type happens 3+ times, generates a stronger
    rule and logs it for the dashboard. This is medium-impact correction
    learning — catches recurring patterns the single-correction learning misses.
    """
    if not what_went_wrong:
        return

    data = _load_correction_patterns()
    data["total_corrections"] = data.get("total_corrections", 0) + 1

    # Normalize the error type to group similar mistakes
    wrong_lower = what_went_wrong.lower()
    error_category = "other"
    if any(w in wrong_lower for w in ["too long", "wordy", "verbose", "lengthy"]):
        error_category = "reply_too_long"
    elif any(w in wrong_lower for w in ["fabricat", "made up", "fake", "invented", "wrong"]):
        error_category = "fabricated_info"
    elif any(w in wrong_lower for w in ["tone", "rude", "formal", "cold", "robotic"]):
        error_category = "wrong_tone"
    elif any(w in wrong_lower for w in ["price", "rate", "cost"]):
        error_category = "wrong_price"
    elif any(w in wrong_lower for w in ["emoji", "smiley"]):
        error_category = "unwanted_emoji"
    elif any(w in wrong_lower for w in ["language", "hindi", "english", "hinglish"]):
        error_category = "wrong_language"
    elif any(w in wrong_lower for w in ["sales pitch", "cta", "buy now", "order now"]):
        error_category = "sales_pitch"
    elif any(w in wrong_lower for w in ["generic", "template", "vague"]):
        error_category = "too_generic"

    patterns = data.get("patterns", {})
    if error_category not in patterns:
        patterns[error_category] = {
            "count": 0,
            "examples": [],
            "rule_generated": False,
        }

    cat_data = patterns[error_category]
    cat_data["count"] = cat_data.get("count", 0) + 1
    cat_data["examples"] = (cat_data.get("examples", []) + [{
        "customer": customer_message[:80],
        "correction": ketu_correction[:80],
        "error": what_went_wrong[:100],
        "time": datetime.now(IST).strftime("%d %b %I:%M %p"),
    }])[-5:]  # Keep last 5 examples

    # When same mistake happens 3+ times and rule not yet generated → create strong rule
    if cat_data["count"] >= 3 and not cat_data.get("rule_generated"):
        _generate_correction_rule(error_category, cat_data)
        cat_data["rule_generated"] = True

    data["patterns"] = patterns
    _save_correction_patterns(data)


def _generate_correction_rule(error_category: str, cat_data: dict):
    """Generate a strong rule from repeated correction patterns.

    Called when the same type of mistake happens 3+ times.
    Creates a new evolved_rule to prevent the mistake permanently.
    """
    # Map error categories to concrete rules
    rule_map = {
        "reply_too_long": "STRICT: Reply 10-15 words ONLY. Ketu har baar edit karta hai long reply ko. Chhota rakho.",
        "fabricated_info": "NEVER fabricate information — stock, delivery dates, availability. Agar nahi pata toh 'Ketu sir batayenge' bol",
        "wrong_tone": "Match Ketu's factory-owner tone — direct, confident, no unnecessary formality. Robotic/corporate tone avoid karo",
        "wrong_price": "NEVER guess prices. ONLY use prices from the PRODUCTS section. Galat price se customer confuse hota hai",
        "unwanted_emoji": "EMOJI MAT USE KAR. Ketu emoji nahi bhejta. Har baar edit karta hai. ZERO emojis",
        "wrong_language": "Customer ki language match karo — Hindi mein bole toh Hindi, English mein toh English. Default Hinglish",
        "sales_pitch": "SALES PITCH KABHI MAT KAR. 'Order now', 'Interested?' jaise CTA mat bol. Tu salesman nahi, factory owner hai",
        "too_generic": "Generic template replies avoid karo. Customer ke specific sawaal ka specific jawab de — seedha, with details",
    }

    rule = rule_map.get(error_category)
    if not rule:
        # Build from examples
        examples = cat_data.get("examples", [])
        if examples:
            rule = f"[Auto-correction] Mistake: {examples[0]['error']}. Ketu's way: {examples[0]['correction']}"
        else:
            return

    try:
        from core.database import is_db_available, load_knowledge_from_db, save_knowledge
        prompt_data = None
        if is_db_available():
            prompt_data = load_knowledge_from_db("prompt")
        if not prompt_data:
            prompt_path = KNOWLEDGE_DIR / "prompt.json"
            try:
                with open(prompt_path, "r", encoding="utf-8") as f:
                    prompt_data = json.load(f)
            except Exception:
                prompt_data = {}

        existing_rules = set(
            r.lower() for r in prompt_data.get("reply_rules", [])
            + prompt_data.get("evolved_rules", [])
        )

        if rule.lower() not in existing_rules:
            prompt_data.setdefault("evolved_rules", []).append(rule)
            if is_db_available():
                save_knowledge("prompt", prompt_data)
            # Also write to file
            prompt_path = KNOWLEDGE_DIR / "prompt.json"
            with open(prompt_path, "w", encoding="utf-8") as f:
                json.dump(prompt_data, f, indent=2, ensure_ascii=False)

            from core.knowledge import invalidate_cache
            invalidate_cache()

            from core.activity_log import log_activity
            log_activity(
                source="correction-learner",
                action="auto-rule-generated",
                details={
                    "error_category": error_category,
                    "correction_count": cat_data.get("count", 0),
                    "rule": rule,
                    "examples": cat_data.get("examples", [])[:3],
                },
                items_count=1,
            )
            logger.info(
                f"[CorrectionLearn] Auto-generated rule from {cat_data.get('count', 0)} corrections: "
                f"{error_category} → '{rule[:60]}...'"
            )
    except Exception as e:
        logger.warning(f"[CorrectionLearn] Rule generation failed (non-fatal): {e}")


def get_correction_stats() -> dict:
    """Get correction learning statistics for dashboard."""
    data = _load_correction_patterns()
    patterns = data.get("patterns", {})
    return {
        "total_corrections": data.get("total_corrections", 0),
        "error_categories": {
            cat: {
                "count": info.get("count", 0),
                "rule_generated": info.get("rule_generated", False),
                "last_example": info.get("examples", [{}])[-1] if info.get("examples") else None,
            }
            for cat, info in patterns.items()
        },
        "top_mistakes": sorted(
            [(cat, info.get("count", 0)) for cat, info in patterns.items()],
            key=lambda x: x[1], reverse=True,
        )[:5],
    }


def learn_from_correction(
    customer_message: str,
    ai_reply: str,
    ketu_correction: str,
    customer_phone: str = "",
    customer_name: str = "",
) -> dict:
    """Learn when Ketu overrides an AI-generated reply.

    This is the MOST powerful learning signal — Ketu is directly saying
    "the AI was wrong, THIS is how I would reply."

    We learn:
    1. The correct reply style/content → new FAQ or style pattern
    2. What the AI got wrong → evolve the prompt to avoid this
    3. The customer<>Ketu exchange → new example conversation
    4. Track correction patterns → auto-generate rules when same mistake repeats 3+ times
    """
    client = Anthropic(api_key=settings.anthropic_api_key)

    # Load current prompt context — DB first (survives deploys), file fallback
    prompt_data = None
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            prompt_data = load_knowledge_from_db("prompt")
    except Exception:
        pass
    if not prompt_data:
        prompt_path = KNOWLEDGE_DIR / "prompt.json"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
        except Exception:
            prompt_data = {}

    current_traits = prompt_data.get("personality_traits", []) + prompt_data.get("evolved_traits", [])
    current_rules = prompt_data.get("reply_rules", []) + prompt_data.get("evolved_rules", [])

    prompt = f"""Ketu (business owner) CORRECTED an AI-generated reply. This is a critical learning moment.

Customer asked: "{customer_message}"

AI (Digital Ketu) replied: "{ai_reply}"

Ketu CORRECTED it to: "{ketu_correction}"

CURRENT personality traits: {json.dumps(current_traits, ensure_ascii=False)}
CURRENT reply rules: {json.dumps(current_rules, ensure_ascii=False)}

Analyze the correction and extract (JSON):
1. "new_faq": If this Q&A pair is worth saving as a FAQ
   Format: {{"question": "...", "answer": "..."}} or null
2. "style_lesson": What the correction teaches about Ketu's style
   Format: "description of style difference" or null
3. "new_rule": A new reply rule to prevent this mistake
   Format: "rule text" or null
4. "example_conversation": The corrected exchange
   Format: {{"customer": "...", "reply": "..."}}
5. "what_went_wrong": Brief explanation of why AI reply was wrong
   Format: "explanation"

Return ONLY valid JSON."""

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text

        # Track API cost
        from core.cost_tracker import track_api_cost
        track_api_cost(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            source="correction-analysis",
        )

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)

        if not json_match:
            return {"status": "parse_error"}

        analysis = json.loads(json_match.group())

        # Build knowledge updates from the correction
        updates = {}

        # New FAQ from correction
        if analysis.get("new_faq"):
            faq = analysis["new_faq"]
            if faq.get("question") and faq.get("answer"):
                updates["new_faqs"] = [faq]

        # Style lesson → style pattern
        if analysis.get("style_lesson"):
            updates["style_patterns"] = [f"[Correction] {analysis['style_lesson']}"]

        # New rule → prompt evolution
        evolution = {}
        if analysis.get("new_rule"):
            evolution["new_rules"] = [analysis["new_rule"]]
        if analysis.get("example_conversation"):
            evolution["example_conversations"] = [analysis["example_conversation"]]
        if analysis.get("style_lesson"):
            evolution["new_traits"] = [analysis["style_lesson"]]

        if evolution:
            updates["prompt_evolution"] = evolution

        # Apply all updates
        from learner.chat_learner import apply_knowledge_updates
        from core.knowledge import invalidate_cache

        result = apply_knowledge_updates(updates)
        if result.get("count", 0) > 0:
            invalidate_cache()

        # Learn Ketu-Only patterns from corrections
        # If the AI fabricated info (timeline, status, price) and Ketu corrected
        # with real info only HE could know, this is a "Ketu Only" signal
        _learn_ketu_only_from_correction(
            customer_message=customer_message,
            ai_reply=ai_reply,
            ketu_correction=ketu_correction,
            what_went_wrong=analysis.get("what_went_wrong", ""),
        )

        # Track correction patterns — auto-generate rules when same mistake repeats 3+ times
        _track_correction_pattern(
            what_went_wrong=analysis.get("what_went_wrong", ""),
            customer_message=customer_message,
            ketu_correction=ketu_correction,
        )

        return {
            "status": "learned",
            "what_went_wrong": analysis.get("what_went_wrong", ""),
            "updates_applied": result.get("applied", []),
            "count": result.get("count", 0),
        }

    except Exception as e:
        logger.error(f"Correction learning error: {e}")
        from core.error_tracker import track_error
        track_error("correction-learner", str(e))
        return {"status": "error", "detail": str(e)}


# --- Ketu-Only Learning from Corrections & Manual Chats ---

# Patterns that indicate the AI fabricated something only Ketu should know
_FABRICATION_SIGNALS = [
    # AI gave a timeline but Ketu gave a different/specific one
    (r"\d+[\s-]*(?:to|se)?\s*\d*\s*(?:din|days|hafte|weeks?|mahine|months?)", "stock_restock"),
    # AI said "check karta hun" but can't actually check anything
    (r"check\s*kart?a?\s*(?:hun|hu|hoon)", "order_status"),
    # AI made up a delivery promise
    (r"(?:kal|parso|aaj)\s*(?:tak|mein|me|by)\s*(?:aa|mil|dispatch|deliver)", "delivery_specific"),
    # AI gave a custom/negotiated price
    (r"(?:special|best|discount)\s*(?:rate|price).*(?:rs|₹|rupee)\s*\d+", "custom_pricing"),
]


def _learn_ketu_only_from_correction(
    customer_message: str,
    ai_reply: str,
    ketu_correction: str,
    what_went_wrong: str = "",
):
    """Learn new Ketu-Only patterns from corrections.

    When Ketu corrects an AI reply, check if the AI fabricated information
    that only Ketu could know. If so, learn the customer's question pattern
    as a new "Ketu Only" trigger.

    This runs automatically on every correction — zero extra API cost.
    """
    try:
        from core.ketu_only import detect_ketu_only, add_learned_pattern, _load_config

        ai_lower = ai_reply.lower()
        cust_lower = customer_message.lower()

        # Check if the AI fabricated a timeline/status/price
        fabrication_detected = False
        detected_category = "learned"

        for pattern, category_id in _FABRICATION_SIGNALS:
            if re.search(pattern, ai_lower):
                fabrication_detected = True
                detected_category = category_id
                break

        # Also check the "what went wrong" analysis for fabrication signals
        wrong_lower = what_went_wrong.lower()
        fabrication_words = ["fabricat", "made up", "fake", "didn't know", "couldn't know",
                            "wrong timeline", "wrong date", "wrong price", "assumed",
                            "guessed", "invented", "incorrect"]
        if any(w in wrong_lower for w in fabrication_words):
            fabrication_detected = True

        if not fabrication_detected:
            return

        # Already detected by existing patterns? Skip learning
        existing = detect_ketu_only(customer_message)
        if existing:
            return

        # Extract the key question pattern from the customer message
        # Look for the core question (2-5 meaningful words)
        # Remove filler words to get the actual question
        filler = {"bhai", "sir", "ji", "bro", "yaar", "please", "pls",
                  "kya", "hai", "ho", "ka", "ki", "ke", "mein", "me",
                  "aur", "or", "the", "is", "a", "my", "mera", "meri"}
        words = [w for w in cust_lower.split() if w not in filler and len(w) > 1]
        if len(words) < 2:
            return  # Too short to be a meaningful pattern

        # Use the cleaned question as a learned pattern
        pattern = " ".join(words[:6])  # Max 6 words

        # Map category_id to name
        config = _load_config()
        cat_name = "Learned Pattern"
        for cat in config.get("categories", []):
            if cat["id"] == detected_category:
                cat_name = cat["name"]
                break

        add_learned_pattern(pattern, detected_category, cat_name)

        from core.activity_log import log_activity
        log_activity(
            source="ketu-only",
            action="learned-from-correction",
            details={
                "pattern": pattern,
                "category": cat_name,
                "customer_asked": customer_message[:80],
                "ai_fabricated": ai_reply[:80],
                "ketu_said": ketu_correction[:80],
            },
            items_count=1,
        )
        logger.info(f"[KetuOnly] Learned new pattern from correction: '{pattern}' -> {cat_name}")

    except Exception as e:
        logger.warning(f"[KetuOnly] Learning from correction failed (non-fatal): {e}")


def learn_ketu_only_from_manual_chat(
    customer_message: str,
    ketu_reply: str,
    customer_phone: str = "",
):
    """Learn Ketu-Only patterns from Ketu's manual chat messages.

    Called during wwbun sync when we detect Ketu manually replying to
    questions about stock, timing, pricing, etc. — things only he knows.

    This teaches the system which TYPES of questions need Ketu's intervention,
    not the actual answers (which change over time).
    """
    try:
        from core.ketu_only import detect_ketu_only, add_learned_pattern, _load_config

        cust_lower = customer_message.lower()
        ketu_lower = ketu_reply.lower()

        # Signals that Ketu gave info only HE could know
        ketu_only_signals = [
            # Ketu gave a specific date/day
            (r"(?:kal|parso|monday|tuesday|wednesday|thursday|friday|saturday|sunday|somvar|mangalvar)", "delivery_specific"),
            # Ketu gave exact stock info
            (r"(?:aa gaya|aa jayega|nahi aayega|nahi hai|khatam|next\s*(?:week|month|batch|lot|winter|summer))", "stock_restock"),
            # Ketu shared tracking/dispatch info
            (r"(?:tracking|dispatch\s*(?:kar diya|ho gaya)|bhej diya|courier)", "order_status"),
            # Ketu gave a custom price
            (r"(?:rs|₹)\s*\d+.*(?:special|tumhare liye|aapke liye|extra discount)", "custom_pricing"),
        ]

        matched_category = None
        for pattern, category_id in ketu_only_signals:
            if re.search(pattern, ketu_lower):
                matched_category = category_id
                break

        if not matched_category:
            return

        # Already detected by existing patterns? Skip
        existing = detect_ketu_only(customer_message)
        if existing:
            return

        # Extract question pattern
        filler = {"bhai", "sir", "ji", "bro", "yaar", "please", "pls",
                  "kya", "hai", "ho", "ka", "ki", "ke", "mein", "me",
                  "aur", "or", "the", "is", "a", "my", "mera", "meri"}
        words = [w for w in cust_lower.split() if w not in filler and len(w) > 1]
        if len(words) < 2:
            return

        pattern = " ".join(words[:6])

        config = _load_config()
        cat_name = "Learned Pattern"
        for cat in config.get("categories", []):
            if cat["id"] == matched_category:
                cat_name = cat["name"]
                break

        add_learned_pattern(pattern, matched_category, cat_name)

        from core.activity_log import log_activity
        log_activity(
            source="ketu-only",
            action="learned-from-chat",
            details={
                "pattern": pattern,
                "category": cat_name,
                "customer_asked": customer_message[:80],
                "ketu_replied": ketu_reply[:80],
                "phone_last4": customer_phone[-4:] if customer_phone else "",
            },
            items_count=1,
        )
        logger.info(f"[KetuOnly] Learned from manual chat: '{pattern}' -> {cat_name}")

    except Exception as e:
        logger.warning(f"[KetuOnly] Learning from chat failed (non-fatal): {e}")


# --- Voice Note Learning ---

async def learn_from_voice_note(
    audio_bytes: bytes,
    context: str = "",
    language: str = "hi",
) -> dict:
    """Learn from Ketu's own voice notes.

    Ketu often explains things in voice notes that he wouldn't type.
    This captures:
    - Product knowledge shared verbally
    - Ketu's speaking style (fillers, phrases, tone)
    - Business info mentioned casually in voice
    - Pricing/offers mentioned verbally

    Args:
        audio_bytes: Raw audio data
        context: Optional context (e.g., "Ketu explaining to customer about hoodies")
        language: Language hint for Whisper
    """
    from learner.audio_transcriber import transcribe_audio

    # Step 1: Transcribe
    transcript = await transcribe_audio(audio_bytes, language=language)
    if not transcript:
        return {"status": "transcription_failed"}

    logger.info(f"Voice note transcribed ({len(transcript)} chars): {transcript[:100]}...")

    # Step 2: Extract knowledge from transcript
    client = Anthropic(api_key=settings.anthropic_api_key)

    context_line = f"\nContext: {context}" if context else ""

    # Load current knowledge for dedup — DB first, file fallback
    prompt_data = None
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            prompt_data = load_knowledge_from_db("prompt")
    except Exception:
        pass
    if not prompt_data:
        prompt_path = KNOWLEDGE_DIR / "prompt.json"
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
        except Exception:
            prompt_data = {}

    current_phrases = prompt_data.get("signature_phrases", []) + prompt_data.get("evolved_phrases", [])

    prompt = f"""This is a transcript of Ketu's VOICE NOTE (he's a t-shirt manufacturer, owner of Sale91.com).
Ketu speaks naturally in voice — this reveals his real speaking style.
{context_line}

Voice note transcript:
"{transcript}"

CURRENT known signature phrases: {json.dumps(current_phrases, ensure_ascii=False)}

Extract from this voice note (JSON):
1. "product_info": Any product details, prices, features mentioned
   Format: [{{"info": "description"}}] or []
2. "business_knowledge": Business policies, offers, processes mentioned
   Format: ["point1", "point2"] or []
3. "speaking_style": Ketu's verbal patterns — Hindi words, fillers, catchphrases
   Format: {{
     "verbal_phrases": ["phrases Ketu uses when speaking"],
     "tone_description": "how he sounds",
     "filler_words": ["common fillers like 'matlab', 'dekho', etc."]
   }}
4. "new_faqs": If Ketu explained something that could be a FAQ
   Format: [{{"question": "...", "answer": "..."}}] or []
5. "key_points": Main points/takeaways from the voice note
   Format: ["point1", "point2"] or []

Return ONLY valid JSON. Focus on NEW info not already known."""

    try:
        import asyncio

        def _call_claude():
            return client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )

        response = await asyncio.to_thread(_call_claude)

        result_text = response.content[0].text

        # Track API cost
        from core.cost_tracker import track_api_cost
        track_api_cost(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            source="voice-analysis",
        )

        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)

        if not json_match:
            return {"status": "parse_error", "transcript": transcript}

        analysis = json.loads(json_match.group())

        # Build knowledge updates
        updates = {}

        # FAQs from voice explanation
        if analysis.get("new_faqs"):
            updates["new_faqs"] = analysis["new_faqs"]

        # Business knowledge + product info → style patterns
        patterns = []
        for info in analysis.get("product_info", []):
            if isinstance(info, dict) and info.get("info"):
                patterns.append(f"[Voice] {info['info']}")
            elif isinstance(info, str):
                patterns.append(f"[Voice] {info}")

        for point in analysis.get("business_knowledge", []):
            patterns.append(f"[Voice] {point}")

        for point in analysis.get("key_points", []):
            patterns.append(f"[Voice] {point}")

        if patterns:
            updates["style_patterns"] = patterns

        # Speaking style → prompt evolution
        speaking = analysis.get("speaking_style", {})
        evolution = {}

        verbal_phrases = speaking.get("verbal_phrases", [])
        if verbal_phrases:
            evolution["new_phrases"] = verbal_phrases

        tone = speaking.get("tone_description", "")
        if tone:
            evolution["new_traits"] = [f"Verbal style: {tone}"]

        if evolution:
            updates["prompt_evolution"] = evolution

        # Apply updates
        from learner.chat_learner import apply_knowledge_updates
        from core.knowledge import invalidate_cache

        result = await asyncio.to_thread(apply_knowledge_updates, updates)
        if result.get("count", 0) > 0:
            invalidate_cache()

        return {
            "status": "ok",
            "transcript": transcript,
            "knowledge_extracted": analysis,
            "updates_applied": result.get("applied", []),
            "count": result.get("count", 0),
        }

    except Exception as e:
        logger.error(f"Voice note learning error: {e}")
        from core.error_tracker import track_error
        track_error("voice-learner", str(e))
        return {"status": "error", "detail": str(e), "transcript": transcript}


def get_realtime_stats() -> dict:
    """Get realtime learner statistics."""
    return {
        "buffer_size": len(_conversation_buffer),
        "conversations_since_last_learn": _conversation_count,
        "learn_threshold": _LEARN_EVERY_N,
        "total_buffered": len(_conversation_buffer),
    }
