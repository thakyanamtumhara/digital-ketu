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
    triggers batch analysis.
    """
    global _conversation_count

    _load_buffer_from_db()

    _conversation_buffer.append({
        "customer": customer_message,
        "reply": ai_reply,
        "name": customer_name,
        "phone_last4": customer_phone[-4:] if customer_phone else "",
        "time": datetime.now(IST).strftime("%I:%M %p"),
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

        conv_text = "\n".join(
            f"Customer ({c['name'] or c['phone_last4']}): {c['customer']}\n"
            f"Digital Ketu: {c['reply']}"
            for c in conversations[-20:]  # Last 20 conversations
        )

        prompt = f"""Analyze these recent conversations between customers and Digital Ketu (AI twin of Ketu, a t-shirt manufacturer).

Conversations:
{conv_text}

Extract (JSON only):
1. "common_questions": Questions asked 2+ times (potential FAQ candidates)
   Format: [{{"question": "...", "suggested_answer": "...", "frequency": N}}]
2. "quality_issues": Any replies that seem wrong, too generic, or miss the point
   Format: [{{"customer_asked": "...", "ai_replied": "...", "issue": "..."}}]
3. "hot_topics": Top 3 topics customers are asking about right now
   Format: ["topic1", "topic2", "topic3"]

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

    except Exception as e:
        logger.error(f"Realtime conversation learning error: {e}")
    finally:
        _learning_lock.release()


# --- Correction Learning ---

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
