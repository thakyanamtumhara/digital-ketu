import asyncio
import logging
import time
from contextlib import asynccontextmanager

from pathlib import Path

import io
import zipfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from core.config import settings, init_knowledge_dir, KNOWLEDGE_DIR
from core.engine import generate_reply, get_customer_insights, get_faq_hit_rates, invalidate_ender_cache, get_last_escalation, ketu_manual_reply, activate_shutup, track_wwbun_insights
from core.knowledge import load_knowledge, invalidate_cache
from core.activity_log import log_activity, get_activity_log, get_today_summary, get_storage_stats
from integrations.whatsapp.webhook import router as whatsapp_router
from integrations.youtube.handler import router as youtube_router
from learner.chat_learner import (
    parse_whatsapp_export,
    extract_knowledge_from_messages,
    extract_knowledge_from_wwbun_messages,
    apply_knowledge_updates,
    learn_conversation_enders,
    detect_bought_customers_from_chat,
    learn_repeat_buyer_patterns,
    is_low_quality_owner_reply,
    has_business_intent,
)
from learner.youtube_learner import process_video
from scheduler import (
    start_scheduler,
    check_youtube_channel,
    backfill_youtube_channel,
    get_backfill_status,
    _mark_run,
    get_scheduler_status,
)
from learner.catalog_syncer import sync_catalog
from learner.faq_validator import (
    validate_faqs_against_catalog,
    get_faq_health_report,
    reactivate_faq,
)
from core.error_tracker import get_recent_errors, get_error_summary
from core.cost_tracker import get_cost_summary
from learner.realtime_learner import (
    learn_from_correction,
    learn_from_voice_note,
    get_realtime_stats,
    get_correction_stats,
)
from core.conversation_log import (
    get_recent_conversations,
    get_last_ai_reply,
    init_conversation_log_table,
)
from core.customer_memory import (
    get_profile as get_customer_profile,
    get_all_profiles_summary,
    _init_customer_table,
)
from core.escalation import LEVEL_ESCALATE
from core.followup import get_pending_followups, execute_followup, get_followup_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure knowledge directories exist
    init_knowledge_dir()

    # Initialize PostgreSQL (create tables, seed from JSON if first run)
    from core.database import init_db, seed_from_json_files, is_db_available
    db_ready = False
    try:
        db_ready = init_db()
        if db_ready:
            seed_from_json_files(KNOWLEDGE_DIR)
            init_conversation_log_table()
            _init_customer_table()
            logger.info("PostgreSQL ready — data persists across deploys")
        else:
            logger.info("No DATABASE_URL — running in JSON-only mode")
    except Exception as e:
        logger.error(f"DB init failed (non-fatal, using JSON fallback): {e}")

    # Restore evolved knowledge from GitHub (backup for JSON-only mode)
    if not db_ready:
        from core.git_persist import restore_knowledge_from_github
        try:
            restored = restore_knowledge_from_github()
            if restored:
                logger.info(f"Knowledge restored from GitHub: {restored}")
        except Exception as e:
            logger.error(f"Knowledge restore failed (non-fatal): {e}")

    # Startup: load knowledge base
    logger.info("Loading knowledge base...")
    load_knowledge()

    # Start background scheduler (YouTube auto-check, knowledge refresh)
    start_scheduler()

    logger.info(
        f"Digital Ketu is ready! Storage: {'PostgreSQL' if db_ready else 'JSON files'}. "
        f"Auto-learning scheduler active."
    )
    yield
    # Shutdown
    logger.info("Digital Ketu shutting down")


app = FastAPI(
    title="Digital Ketu",
    description="AI Digital Twin for Own Knitted Blank Wears (Sale91.com)",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount integration routers
app.include_router(whatsapp_router)
app.include_router(youtube_router)


# --- Health & Status ---


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "digital-ketu",
        "auto_reply": settings.auto_reply_enabled,
    }


@app.get("/api/health/detailed")
async def health_detailed():
    """Detailed health check — DB status, error counts, last sync times, uptime."""
    from core.database import is_db_available
    from core.error_tracker import get_error_summary

    scheduler = get_scheduler_status()
    errors = get_error_summary()

    # Check DB connectivity
    db_ok = is_db_available()

    return {
        "status": "ok" if db_ok and errors["last_hour_errors"] < 10 else "degraded",
        "service": "digital-ketu",
        "auto_reply": settings.auto_reply_enabled,
        "database": "connected" if db_ok else "disconnected",
        "errors": errors,
        "scheduler": {
            name: {
                "status": info["status"],
                "last_run_ago": info.get("last_run_ago"),
            }
            for name, info in scheduler.items()
        },
    }


@app.get("/api/errors")
async def api_errors(limit: int = 20):
    """Recent errors for dashboard."""
    return {
        "errors": get_recent_errors(limit),
        "summary": get_error_summary(),
    }


# --- AI Reply API (for wwbun integration) ---


class ReplyRequest(BaseModel):
    message: str
    customer_phone: str = ""
    customer_name: str = ""
    conversation_history: list[dict] | None = None
    # Audio fields — wwbun sends these when customer sends voice note
    audio_url: str = ""       # Direct URL to audio file (from wwbun's storage)
    audio_base64: str = ""    # Base64-encoded audio bytes
    media_id: str = ""        # WhatsApp Business API media ID


class ReplyResponse(BaseModel):
    reply: str
    status: str = "ok"
    should_reply: bool = True
    escalation_level: str = "none"
    escalation_reason: str = ""


@app.post("/api/reply", response_model=ReplyResponse)
async def api_reply(req: ReplyRequest):
    """Generate AI reply — used by wwbun to get Digital Ketu's response.

    wwbun sends customer message → Digital Ketu returns reply.
    If should_reply is false, wwbun should NOT send anything — customer
    just acknowledged (said "ok", "thanks", etc.) and conversation is done.
    """
    # Handle media markers from wwbun — these are not real text messages
    msg_lower = req.message.strip().lower()
    if msg_lower in ("[audio]", "[image]", "[video]", "[sticker]", "[document]",
                      "[location]", "[contacts]", "[system message]"):
        # Voice notes: transcribe if audio data provided, otherwise ask for text
        if msg_lower == "[audio]":
            has_audio = req.audio_url or req.audio_base64 or req.media_id
            if has_audio and settings.openai_api_key:
                # Transcribe the voice note
                from learner.audio_transcriber import process_audio_from_any_source
                try:
                    transcribed = await process_audio_from_any_source(
                        media_id=req.media_id,
                        audio_url=req.audio_url,
                        audio_base64=req.audio_base64,
                        customer_phone=req.customer_phone,
                        source="wwbun",
                    )
                except Exception as e:
                    logger.error(f"[VoiceNote] wwbun transcription failed: {e}")
                    transcribed = None

                if transcribed:
                    # Got text from voice — now generate reply as normal
                    logger.info(f"[VoiceNote] wwbun transcribed: '{transcribed[:60]}' from {req.customer_phone[-4:] if req.customer_phone else '?'}")
                    log_activity(
                        source="api-reply",
                        action="voice-transcribed",
                        details={
                            "customer_phone": req.customer_phone[-4:] if req.customer_phone else "unknown",
                            "text_preview": transcribed[:80],
                        },
                        items_count=1,
                    )
                    # Use transcribed text as the message for reply generation
                    reply = await asyncio.to_thread(
                        generate_reply,
                        message=transcribed,
                        customer_phone=req.customer_phone,
                        customer_name=req.customer_name,
                        conversation_history=req.conversation_history,
                    )
                    if not reply:
                        return ReplyResponse(reply="", status="skipped", should_reply=False)
                    escalation = get_last_escalation(req.customer_phone) if req.customer_phone else {}
                    return ReplyResponse(
                        reply=reply,
                        escalation_level=escalation.get("level", "none"),
                        escalation_reason=escalation.get("reason", ""),
                    )
                else:
                    # Transcription failed — ask for text
                    reply = "Ji sir, voice message clear nahi aa raha. Text mein bata dijiye please!"
                    return ReplyResponse(reply=reply, status="ok", should_reply=True)
            else:
                # No audio data or no OpenAI key — ask for text
                reply = "Ji sir, voice message text mein bhej dijiye please — jaldi reply karunga!"
                log_activity(
                    source="api-reply",
                    action="voice-text-request",
                    details={
                        "customer_phone": req.customer_phone[-4:] if req.customer_phone else "unknown",
                        "customer_name": req.customer_name or "unknown",
                    },
                    items_count=0,
                )
                return ReplyResponse(reply=reply, status="ok", should_reply=True)
        # Other media: skip silently (images, stickers, etc.)
        return ReplyResponse(reply="", status="skipped", should_reply=False)

    reply = await asyncio.to_thread(
        generate_reply,
        message=req.message,
        customer_phone=req.customer_phone,
        customer_name=req.customer_name,
        conversation_history=req.conversation_history,
    )

    # Empty reply = conversation ender detected, don't send
    if not reply:
        log_activity(
            source="api-reply",
            action="skipped",
            details={
                "customer_phone": req.customer_phone[-4:] if req.customer_phone else "unknown",
                "customer_name": req.customer_name or "unknown",
                "message_preview": req.message[:80],
                "reason": "conversation_ender",
            },
            items_count=0,
        )
        return ReplyResponse(reply="", status="skipped", should_reply=False)

    # Check if escalation was detected
    escalation = get_last_escalation(req.customer_phone) if req.customer_phone else {}
    esc_level = escalation.get("level", "none")
    esc_reason = escalation.get("reason", "")

    log_activity(
        source="api-reply",
        action="replied",
        details={
            "customer_phone": req.customer_phone[-4:] if req.customer_phone else "unknown",
            "customer_name": req.customer_name or "unknown",
            "message_preview": req.message[:80],
            "reply_preview": reply[:80],
            "escalation_level": esc_level,
            "escalation_reason": esc_reason,
        },
        items_count=1,
    )
    return ReplyResponse(
        reply=reply,
        escalation_level=esc_level,
        escalation_reason=esc_reason,
    )


# --- Auto-Reply Toggle ---


class ToggleRequest(BaseModel):
    enabled: bool


@app.post("/api/toggle")
async def toggle_auto_reply(req: ToggleRequest):
    """Enable/disable auto-reply. Also auto-enables follow-up when turning on."""
    settings.auto_reply_enabled = req.enabled
    # Auto-enable followup when auto-reply turns on
    if req.enabled:
        settings.followup_enabled = True
    status = "enabled" if req.enabled else "disabled"
    logger.info(f"Auto-reply {status} (followup: {'on' if settings.followup_enabled else 'off'})")
    return {
        "status": status,
        "auto_reply": settings.auto_reply_enabled,
        "followup_enabled": settings.followup_enabled,
    }


@app.get("/api/toggle")
async def get_toggle_status():
    return {
        "auto_reply": settings.auto_reply_enabled,
        "followup_enabled": settings.followup_enabled,
    }


# --- Follow-up Toggle (independent control) ---


@app.post("/api/followup/toggle")
async def toggle_followup(req: ToggleRequest):
    """Enable/disable follow-up independently of auto-reply."""
    settings.followup_enabled = req.enabled
    status = "enabled" if req.enabled else "disabled"
    logger.info(f"Follow-up {status}")
    return {"followup_enabled": settings.followup_enabled}


@app.get("/api/followup/toggle")
async def get_followup_toggle_status():
    return {"followup_enabled": settings.followup_enabled}


# --- Knowledge Management ---


@app.get("/api/knowledge/reload")
async def reload_knowledge():
    """Force reload knowledge base from JSON files."""
    invalidate_cache()
    knowledge = load_knowledge()
    return {
        "status": "reloaded",
        "sections": list(knowledge.keys()),
    }


@app.get("/api/knowledge/cleanup-enders")
async def cleanup_bad_enders():
    """Remove greetings and junk from learned enders list.

    Fixes the bug where greetings like 'hi', 'hello' were incorrectly
    learned as conversation enders because Ketu was busy and didn't reply.
    """
    from core.database import is_db_available, load_knowledge_from_db, save_knowledge

    # Greetings that should NEVER be enders
    never_enders = {
        "hi", "hii", "hiii", "hiiii", "hello", "hey", "heyy", "heyyy",
        "hlo", "helo", "hllo", "helloo", "hellooo",
        "namaste", "namaskar", "namaskaar",
        "good morning", "good afternoon", "good evening", "good night",
        "gm", "gn", "sir", "bhai", "bhaiya", "bro", "boss",
        "hello sir", "hi sir", "hey sir", "hello bhai", "hi bhai",
    }

    # Also remove obvious junk (gibberish, system messages, etc.)
    junk_prefixes = ["[image", "[audio", "[video", "[sticker", "[system", "[order", "[reacted"]

    enders_data = None
    if is_db_available():
        enders_data = load_knowledge_from_db("conversation_enders")

    if not enders_data:
        return {"status": "no_data", "removed": 0}

    learned = enders_data.get("learned_enders", [])
    original_count = len(learned)

    # Filter out greetings and junk
    cleaned = []
    removed = []
    for e in learned:
        pattern = e.get("pattern", e) if isinstance(e, dict) else e
        pattern_lower = pattern.lower().strip()

        # Remove greetings
        if pattern_lower in never_enders:
            removed.append(pattern_lower)
            continue

        # Remove junk prefixes
        if any(pattern_lower.startswith(p) for p in junk_prefixes):
            removed.append(pattern_lower)
            continue

        cleaned.append(e)

    enders_data["learned_enders"] = cleaned

    # Save back
    save_knowledge("conversation_enders", enders_data)

    # Also invalidate the engine's cached ender patterns
    invalidate_cache()

    return {
        "status": "cleaned",
        "removed_count": len(removed),
        "removed_patterns": removed,
        "remaining_count": len(cleaned),
        "original_count": original_count,
    }


@app.get("/api/knowledge")
async def get_knowledge():
    """View current knowledge base."""
    return load_knowledge()


# --- Learner APIs ---


class LearnWhatsAppRequest(BaseModel):
    chat_text: str
    ketu_name: str = "Ketu"


@app.post("/api/learn/whatsapp-export")
async def learn_from_whatsapp_export(req: LearnWhatsAppRequest):
    """Learn from a WhatsApp chat export (text file content).

    Upload your WhatsApp chat export text and Digital Ketu
    will extract knowledge from YOUR messages (not customers/AI).
    """
    messages = parse_whatsapp_export(req.chat_text)
    if not messages:
        return {"status": "no_messages_found"}

    knowledge = await asyncio.to_thread(extract_knowledge_from_messages, messages, req.ketu_name)
    result = await asyncio.to_thread(apply_knowledge_updates, knowledge)

    invalidate_cache()

    _mark_run("whatsapp")

    log_activity(
        source="whatsapp-export",
        action="learned",
        details={
            "messages_parsed": len(messages),
            "ketu_name": req.ketu_name,
            "updates_applied": result.get("applied", []),
        },
        items_count=result.get("count", 0),
    )

    return {
        "status": "ok",
        "messages_parsed": len(messages),
        "knowledge_extracted": knowledge,
        "updates_applied": result,
    }


class LearnWwbunRequest(BaseModel):
    messages: list[dict]
    owner_user_id: str


# --- WhatsApp Chat Sync Stats (persistent, survives deploys) ---
_wwbun_stats = {
    "total_syncs": 0,
    "total_messages_received": 0,
    "total_quality_messages": 0,
    "total_junk_filtered": 0,
    "total_too_short_filtered": 0,
    "total_knowledge_applied": 0,
    "total_enders_learned": 0,
    "today_date": "",
    "today_syncs": 0,
    "today_messages": 0,
    "today_quality": 0,
    "today_knowledge": 0,
    "last_sync_time": "",
    "last_sync_details": {},
    "recent_quality_messages": [],  # Last 20 quality message previews
}
_wwbun_stats_loaded = False


def _load_wwbun_stats():
    global _wwbun_stats, _wwbun_stats_loaded
    if _wwbun_stats_loaded:
        return
    _wwbun_stats_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            saved = kv_get("wwbun_sync_stats")
            if saved and isinstance(saved, dict):
                _wwbun_stats.update(saved)
                logger.info(
                    f"[wwbun-stats LOAD] Loaded from DB: total_quality={saved.get('total_quality_messages', 0)}, "
                    f"today_quality={saved.get('today_quality', 0)}, total_syncs={saved.get('total_syncs', 0)}"
                )
            else:
                logger.warning(f"[wwbun-stats LOAD] No saved stats in DB (saved={type(saved).__name__})")
        else:
            logger.warning("[wwbun-stats LOAD] DB not available — stats will reset on deploy!")
    except Exception as e:
        logger.error(f"[wwbun-stats LOAD] Failed to load: {e}")


def _save_wwbun_stats():
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("wwbun_sync_stats", _wwbun_stats)
            logger.debug(
                f"[wwbun-stats SAVE] Saved: total_quality={_wwbun_stats['total_quality_messages']}, "
                f"today_quality={_wwbun_stats['today_quality']}"
            )
        else:
            logger.error("[wwbun-stats SAVE] DB not available — stats NOT persisted!")
    except Exception as e:
        logger.error(f"[wwbun-stats SAVE] Failed to save: {e}")


def _track_wwbun_sync(
    total_messages: int,
    quality_count: int,
    junk_count: int,
    short_count: int,
    knowledge_count: int,
    enders_learned: int,
    quality_previews: list,
    details: dict,
):
    """Track a wwbun sync event for dashboard stats."""
    _load_wwbun_stats()

    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    today = now.strftime("%d %b %Y")

    # Reset daily counters if new day
    if _wwbun_stats["today_date"] != today:
        _wwbun_stats["today_date"] = today
        _wwbun_stats["today_syncs"] = 0
        _wwbun_stats["today_messages"] = 0
        _wwbun_stats["today_quality"] = 0
        _wwbun_stats["today_knowledge"] = 0

    # Update totals
    _wwbun_stats["total_syncs"] += 1
    _wwbun_stats["total_messages_received"] += total_messages
    _wwbun_stats["total_quality_messages"] += quality_count
    _wwbun_stats["total_junk_filtered"] += junk_count
    _wwbun_stats["total_too_short_filtered"] += short_count
    _wwbun_stats["total_knowledge_applied"] += knowledge_count
    _wwbun_stats["total_enders_learned"] += enders_learned

    # Update today's counters
    _wwbun_stats["today_syncs"] += 1
    _wwbun_stats["today_messages"] += total_messages
    _wwbun_stats["today_quality"] += quality_count
    _wwbun_stats["today_knowledge"] += knowledge_count

    # Last sync info
    _wwbun_stats["last_sync_time"] = now.strftime("%I:%M %p, %d %b")
    _wwbun_stats["last_sync_details"] = details

    # Recent quality pairs (for live preview — customer Q + Ketu reply)
    # Take the LAST 5 (newest) pairs, not the first 5 (oldest)
    # Deduplicate against existing pairs to avoid re-adding same pairs from full buffer
    existing_keys = set()
    for existing in _wwbun_stats["recent_quality_messages"]:
        key = (existing.get("customer", ""), existing.get("ketu", ""))
        existing_keys.add(key)

    for msg in quality_previews[-5:]:
        if isinstance(msg, dict) and msg.get("customer") and msg.get("ketu"):
            key = (msg["customer"][:100], msg["ketu"][:120])
            if key in existing_keys:
                continue  # Skip duplicate pair
            _wwbun_stats["recent_quality_messages"].append({
                "customer": msg["customer"][:100],
                "ketu": msg["ketu"][:120],
                "ai": msg.get("ai", False),
                "time": now.strftime("%I:%M %p"),
                "phone": msg.get("phone_hint", ""),
                "chat_id": msg.get("chat_id", ""),
                "name": msg.get("contact_name", ""),
            })
            existing_keys.add(key)
        elif isinstance(msg, str):
            # Backward compatible: old-style single message
            _wwbun_stats["recent_quality_messages"].append({
                "ketu": msg[:120],
                "time": now.strftime("%I:%M %p"),
            })
    _wwbun_stats["recent_quality_messages"] = _wwbun_stats["recent_quality_messages"][-20:]

    _save_wwbun_stats()


@app.get("/api/wwbun/stats")
async def wwbun_sync_stats():
    """WhatsApp Chat Sync live stats — messages synced, quality, junk, knowledge extracted."""
    _load_wwbun_stats()

    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d %b %Y")
    if _wwbun_stats["today_date"] != today:
        _wwbun_stats["today_date"] = today
        _wwbun_stats["today_syncs"] = 0
        _wwbun_stats["today_messages"] = 0
        _wwbun_stats["today_quality"] = 0
        _wwbun_stats["today_knowledge"] = 0

    return {
        "total_syncs": _wwbun_stats["total_syncs"],
        "total_messages_received": _wwbun_stats["total_messages_received"],
        "total_quality_messages": _wwbun_stats["total_quality_messages"],
        "total_junk_filtered": _wwbun_stats["total_junk_filtered"],
        "total_too_short_filtered": _wwbun_stats["total_too_short_filtered"],
        "total_knowledge_applied": _wwbun_stats["total_knowledge_applied"],
        "total_enders_learned": _wwbun_stats["total_enders_learned"],
        "today_syncs": _wwbun_stats["today_syncs"],
        "today_messages": _wwbun_stats["today_messages"],
        "today_quality": _wwbun_stats["today_quality"],
        "today_knowledge": _wwbun_stats["today_knowledge"],
        "last_sync_time": _wwbun_stats["last_sync_time"],
        "last_sync_details": _wwbun_stats["last_sync_details"],
        "recent_quality_messages": _wwbun_stats["recent_quality_messages"][-10:],
    }


@app.get("/api/wwbun/debug")
async def wwbun_debug():
    """Diagnostic endpoint — shows raw buffer state, message classification, and stats.
    Use this to debug why quality messages might show as 0.
    """
    _load_wwbun_stats()
    buffer = _get_learning_buffer()
    owner_user_id = _get_owner_user_id()

    # Classify each message in buffer
    classifications = []
    for i, m in enumerate(buffer[-20:]):  # Last 20 messages
        is_owner = _is_owner_message(m, owner_user_id) if owner_user_id else None
        is_ai = _safe_bool(m.get("is_ai_generated"))
        text = m.get("content", "") or m.get("text", "") or m.get("body", "")
        classifications.append({
            "index": i,
            "sender_id": m.get("sender_id"),
            "is_owner_flag": m.get("is_owner"),
            "is_owner_computed": is_owner,
            "is_ai": is_ai,
            "content_preview": text[:80] if text else "<EMPTY>",
            "word_count": len(text.split()) if text else 0,
            "fields": list(m.keys()),
        })

    # Use the shared pair extraction
    pairs = _extract_conversation_pairs(buffer, owner_user_id) if owner_user_id else []
    quality_count = len(pairs)

    broken_sids = _all_sender_ids_same(buffer) if buffer else False
    is_owner_reliable = _is_owner_field_reliable(buffer) if buffer else False

    # Unique sender_ids for diagnosis
    unique_sids = list(set(str(m.get("sender_id", "")) for m in buffer)) if buffer else []

    return {
        "owner_user_id": owner_user_id,
        "buffer_size": len(buffer),
        "broken_sender_ids": broken_sids,
        "is_owner_field_reliable": is_owner_reliable,
        "detection_mode": (
            "word_count_heuristic" if broken_sids and not is_owner_reliable
            else "is_owner_flag" if broken_sids and is_owner_reliable
            else "sender_id_match"
        ),
        "unique_sender_ids": unique_sids,
        "quality_pairs_in_buffer": quality_count,
        "pairs_preview": [{"c": p["customer"][:50], "k": p["ketu"][:50]} for p in pairs[:5]],
        "threshold": _LEARNING_BUFFER_MIN_PAIRS,
        "stats_from_db": {
            "total_quality": _wwbun_stats["total_quality_messages"],
            "today_quality": _wwbun_stats["today_quality"],
            "total_syncs": _wwbun_stats["total_syncs"],
            "today_syncs": _wwbun_stats["today_syncs"],
            "total_messages": _wwbun_stats["total_messages_received"],
            "last_sync": _wwbun_stats["last_sync_time"],
        },
        "last_20_messages": classifications,
    }


def _extract_conversation_pairs(messages: list[dict], owner_user_id: str) -> list[dict]:
    """Single source of truth for extracting customer→Ketu pairs.

    Used by BOTH dashboard display AND ketu-only learning. ONE function, no duplication.

    Detection priority:
    1. sender_id match → most reliable
    2. is_owner flag → if sender_ids are broken but flag is mixed (reliable)
    3. Word-count heuristic → last resort when everything is broken

    CRITICAL: broken_sids and is_owner reliability are checked BUFFER-WIDE,
    not per-chat. Per-chat groups (2-3 msgs) are too small for reliable detection.
    """
    # Detect broken data BUFFER-WIDE (more statistically robust than per-chat)
    broken_sids = _all_sender_ids_same(messages)
    use_flag = broken_sids and _is_owner_field_reliable(messages)

    if broken_sids:
        if use_flag:
            logger.info("[extract-pairs] sender_ids broken but is_owner field reliable → using is_owner flag")
        else:
            logger.warning("[extract-pairs] sender_ids AND is_owner BOTH unreliable → word-count heuristic")

    # Group by chat_id (same conversation)
    # CRITICAL: when chat_id is missing (wwbun doesn't send it), fall back to
    # sender_id for NON-owner messages so different customers stay separated.
    # Owner messages go into the same group as the customer they're replying to.
    by_chat: dict[str, list] = {}
    for m in messages:
        chat_id = m.get("chat_id", m.get("remote_jid", ""))
        if not chat_id:
            # No chat_id — use sender_id to group customer messages separately.
            # Owner messages: group by "unknown" for now, then re-assign below.
            sid = str(m.get("sender_id", ""))
            is_owner = _is_owner_by_flag(m) if use_flag else _is_owner_message(m, owner_user_id)
            if is_owner:
                chat_id = f"_owner_{sid}"
            else:
                chat_id = f"_cust_{sid}"
        by_chat.setdefault(chat_id, []).append(m)

    # When chat_id is missing, owner messages are in separate "_owner_*" groups.
    # We need to assign each owner reply to the correct customer thread.
    # Strategy: process chronologically — each Ketu reply goes to the thread
    # with the oldest UNANSWERED customer message (not yet replied to since
    # that customer's last message).
    has_synthetic_groups = any(k.startswith("_cust_") or k.startswith("_owner_") for k in by_chat)
    if has_synthetic_groups:
        # Separate customer threads and owner messages
        cust_threads: dict[str, list] = {}  # sender_id → messages
        owner_msgs: list = []
        real_groups: dict[str, list] = {}

        for k, v in by_chat.items():
            if k.startswith("_cust_"):
                sid = k[len("_cust_"):]
                cust_threads[sid] = sorted(v, key=lambda x: x.get("timestamp", x.get("created_at", "")))
            elif k.startswith("_owner_"):
                owner_msgs.extend(v)
            else:
                real_groups[k] = v

        if cust_threads:
            # Sort owner messages by timestamp
            owner_msgs.sort(key=lambda x: x.get("timestamp", x.get("created_at", "")))

            # Build a timeline of ALL customer messages across all threads
            # to know when each thread has "new" unanswered messages.
            # Track: last_owner_reply_ts per thread — any customer msg AFTER this
            # means the thread has unanswered messages.
            last_reply_ts: dict[str, str] = {}  # sid → timestamp of last assigned owner reply

            for owner_msg in owner_msgs:
                owner_ts = owner_msg.get("timestamp", owner_msg.get("created_at", ""))
                owner_content = (owner_msg.get("content", "") or owner_msg.get("text", "") or owner_msg.get("body", "")).strip()
                owner_is_low_quality = is_low_quality_owner_reply(owner_content)

                # Find threads with unanswered customer messages:
                # A thread is "unanswered" if it has customer messages with
                # timestamps AFTER the last owner reply assigned to it.
                unanswered = []
                for sid, msgs in cust_threads.items():
                    last_reply = last_reply_ts.get(sid, "")
                    # Find the latest customer msg in this thread
                    latest_cust_ts = ""
                    for m in msgs:
                        m_ts = m.get("timestamp", m.get("created_at", ""))
                        m_is_owner = _is_owner_by_flag(m) if use_flag else _is_owner_message(m, owner_user_id)
                        if not m_is_owner and m_ts > latest_cust_ts:
                            latest_cust_ts = m_ts
                    # Thread has unanswered msgs if latest customer msg is after last reply
                    if latest_cust_ts and latest_cust_ts > last_reply:
                        # Use the FIRST unanswered customer msg timestamp for ordering
                        first_unanswered = ""
                        for m in msgs:
                            m_ts = m.get("timestamp", m.get("created_at", ""))
                            m_is_owner = _is_owner_by_flag(m) if use_flag else _is_owner_message(m, owner_user_id)
                            if not m_is_owner and m_ts > last_reply:
                                first_unanswered = m_ts
                                break
                        unanswered.append((sid, first_unanswered))

                if unanswered:
                    # Assign to thread with oldest unanswered message
                    unanswered.sort(key=lambda x: x[1])
                    target_sid = unanswered[0][0]
                else:
                    # All threads answered — assign to thread with oldest first msg
                    target_sid = sorted(
                        cust_threads.keys(),
                        key=lambda s: cust_threads[s][0].get("timestamp", cust_threads[s][0].get("created_at", ""))
                    )[0]

                cust_threads[target_sid].append(owner_msg)
                # FIX: Low-quality owner replies (e.g. "Ok", "Done", "Hmm") should NOT
                # mark a thread as "answered". They won't create learning pairs anyway,
                # and marking them as answered causes wrong assignment when Ketu replies
                # out of order (newer buyer first, older buyer second).
                if not owner_is_low_quality:
                    last_reply_ts[target_sid] = owner_ts

            # Rebuild by_chat with proper thread groups
            rebuilt = {}
            for sid, msgs in cust_threads.items():
                rebuilt[f"_thread_{sid}"] = sorted(msgs, key=lambda x: x.get("timestamp", x.get("created_at", "")))
            by_chat = {**real_groups, **rebuilt}

    all_pairs = []
    skipped_low_quality = 0
    skipped_no_intent = 0
    # Max time gap (5 min) between consecutive customer messages to combine them.
    # If gap > 5 min, treat as a new conversation thread and reset the buffer.
    _COMBINE_MAX_GAP_SEC = 300  # 5 minutes

    def _parse_ts(msg_dict: dict):
        """Parse timestamp from message, return datetime or None."""
        ts = msg_dict.get("timestamp") or msg_dict.get("created_at") or ""
        if not ts:
            return None
        try:
            from datetime import datetime as _dt
            if isinstance(ts, str):
                # ISO format: 2024-01-15T10:30:00.000Z
                return _dt.fromisoformat(ts.replace("Z", "+00:00"))
            return ts  # already a datetime
        except Exception:
            return None

    for chat_id, chat_msgs in by_chat.items():
        chat_msgs.sort(key=lambda x: x.get("timestamp", x.get("created_at", "")))
        # Extract contact_name from any message in this chat (for debug display)
        _chat_contact_name = ""
        for _cm in chat_msgs:
            _cn = _cm.get("contact_name", "")
            if _cn:
                _chat_contact_name = _cn
                break
        # Multi-message combining: accumulate consecutive customer messages
        customer_msgs_buffer: list[str] = []
        last_customer_ts = None  # track timestamp of last buffered customer msg

        for msg in chat_msgs:
            content = (msg.get("content", "") or msg.get("text", "") or msg.get("body", "")).strip()
            if not content:
                continue

            msg_ts = _parse_ts(msg)
            is_ai = _safe_bool(msg.get("is_ai_generated"))

            def _add_to_buffer(text: str):
                """Add customer message to buffer, respecting time gap."""
                nonlocal last_customer_ts
                # If time gap > 5 min from last buffered msg, reset buffer (new thread)
                if customer_msgs_buffer and msg_ts and last_customer_ts:
                    gap = (msg_ts - last_customer_ts).total_seconds()
                    if gap > _COMBINE_MAX_GAP_SEC:
                        customer_msgs_buffer.clear()
                customer_msgs_buffer.append(text)
                last_customer_ts = msg_ts

            def _try_create_pair(owner_content: str, is_ai_flag: bool):
                """Try to create a quality pair from buffered customer msgs + owner reply."""
                nonlocal last_customer_ts
                if is_low_quality_owner_reply(owner_content):
                    return "low_quality"
                combined_customer = " | ".join(customer_msgs_buffer)
                if not has_business_intent(combined_customer):
                    return "no_intent"
                # Extract phone hint from chat_id for debug display
                _phone_hint = ""
                _cid = chat_id or ""
                if _cid.startswith("_thread_"):
                    _phone_hint = _cid[len("_thread_"):][-4:]  # last 4 digits
                elif "@" in _cid:
                    _phone_hint = _cid.split("@")[0][-4:]
                elif _cid.startswith("_cust_"):
                    _phone_hint = _cid[len("_cust_"):][-4:]
                else:
                    _phone_hint = _cid[-4:] if _cid else ""
                all_pairs.append({
                    "customer": combined_customer[:200],
                    "ketu": owner_content[:120],
                    "ai": is_ai_flag,
                    "chat_id": chat_id,
                    "phone_hint": _phone_hint,  # last 4 digits for debug
                    "contact_name": _chat_contact_name,
                })
                return "ok"

            if broken_sids and not use_flag:
                # Last resort heuristic: short (≤4 words) = customer, long (5+ words) = Ketu
                words = len(content.split())
                if is_ai:
                    # AI reply — clear customer buffer (don't let messages leak past)
                    customer_msgs_buffer.clear()
                    last_customer_ts = None
                elif words <= 4:
                    _add_to_buffer(content)
                elif words >= 5 and customer_msgs_buffer:
                    result = _try_create_pair(content, is_ai)
                    if result == "low_quality":
                        skipped_low_quality += 1
                        # Don't clear buffer — same fix as sender_id path
                    elif result == "no_intent":
                        skipped_no_intent += 1
                        customer_msgs_buffer.clear()
                        last_customer_ts = None
                    else:
                        customer_msgs_buffer.clear()
                        last_customer_ts = None
            else:
                is_owner = _is_owner_by_flag(msg) if use_flag else _is_owner_message(msg, owner_user_id)
                if not is_owner:
                    _add_to_buffer(content)
                elif is_owner and not is_ai and customer_msgs_buffer:
                    result = _try_create_pair(content, is_ai)
                    if result == "low_quality":
                        skipped_low_quality += 1
                        # FIX: Don't clear customer buffer on low-quality reply.
                        # Low-quality replies like "Ok", "Done" may be mis-assigned
                        # to the wrong thread (out-of-order reply bug). Keeping the
                        # buffer lets the NEXT real owner reply pair correctly.
                    elif result == "no_intent":
                        skipped_no_intent += 1
                        customer_msgs_buffer.clear()
                        last_customer_ts = None
                    else:
                        customer_msgs_buffer.clear()
                        last_customer_ts = None
                elif is_owner and is_ai:
                    # AI-generated owner reply — clear customer buffer to prevent
                    # customer messages from leaking past AI replies into the next
                    # manual Ketu reply. Without this, messages accumulate wrongly.
                    customer_msgs_buffer.clear()
                    last_customer_ts = None

    logger.info(
        f"[extract-pairs] msgs={len(messages)}, chats={len(by_chat)}, "
        f"pairs={len(all_pairs)}, skipped_low_quality={skipped_low_quality}, "
        f"skipped_no_intent={skipped_no_intent}, "
        f"broken_sids={broken_sids}, use_flag={use_flag}"
    )
    return all_pairs


def _learn_ketu_only_pairs(messages: list[dict], owner_user_id: str, learn_fn):
    """Learn ketu-only patterns from extracted pairs. Uses _extract_conversation_pairs."""
    pairs = _extract_conversation_pairs(messages, owner_user_id)
    for p in pairs:
        if not p.get("ai"):  # Only learn from manual (non-AI) replies
            customer_phone = p.get("chat_id", "unknown").split("@")[0]
            learn_fn(
                customer_message=p["customer"],
                ketu_reply=p["ketu"],
                customer_phone=customer_phone,
            )
    if pairs:
        logger.info(f"[KetuOnly] Checked {len(pairs)} customer→Ketu pairs from wwbun sync")


# --- Message Accumulator for Batch Learning ---
# wwbun sends small batches (2-3 msgs) frequently.
# We buffer them and only call Claude when we have 10+ quality pairs.
# Free features (enders, bought detection, repeat buyer) still run immediately.

_LEARNING_BUFFER_MIN_PAIRS = 20  # Need 20 quality Ketu manual pairs before learning (bigger batch = better pattern detection)
_LEARNING_FLUSH_COOLDOWN = 1800  # 30 minutes minimum between Claude learning calls (was 10 min — saves ~50% learning cost)
_LEARNING_FORCE_FLUSH_PAIRS = 50  # Force flush if 50+ quality pairs (too much data waiting)
_last_flush_time: float = 0  # Timestamp of last Claude learning flush
_last_owner_user_id = ""  # Remember last owner_user_id from sync calls


def _save_owner_user_id(uid: str):
    """Persist owner_user_id so buffer-status works after redeploys."""
    global _last_owner_user_id
    _last_owner_user_id = uid
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("last_owner_user_id", uid)
    except Exception:
        pass


def _get_owner_user_id() -> str:
    """Get owner_user_id from memory or DB."""
    global _last_owner_user_id
    if _last_owner_user_id:
        return _last_owner_user_id
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            val = kv_get("last_owner_user_id")
            if val:
                _last_owner_user_id = val
                return val
    except Exception:
        pass
    return ""


def _get_learning_buffer() -> list[dict]:
    """Load accumulated messages from DB."""
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get("wwbun_learning_buffer")
            if data and isinstance(data, list):
                return data
    except Exception as e:
        logger.warning(f"[Buffer] Load failed: {e}")
    return []


def _save_learning_buffer(buffer: list[dict]):
    """Save accumulated messages to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("wwbun_learning_buffer", buffer)
    except Exception as e:
        logger.warning(f"[Buffer] Save failed: {e}")


def _message_fingerprint(m: dict) -> str:
    """Create a unique fingerprint for a message to detect duplicates.

    Uses chat_id + sender_id + content + timestamp to identify the same message
    sent across multiple wwbun syncs.
    """
    chat_id = m.get("chat_id", m.get("remote_jid", ""))
    sender = m.get("sender_id", "")
    content = (m.get("content", "") or m.get("text", "") or m.get("body", "")).strip()
    ts = str(m.get("timestamp", m.get("created_at", "")))
    if not content:
        return ""  # Skip empty messages
    return f"{chat_id}|{sender}|{content[:100]}|{ts}"


def _safe_bool(val) -> bool:
    """Safely parse boolean — wwbun may send string 'true'/'false' or None."""
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "1", "yes")
    return bool(val)


def _is_owner_message(m: dict, owner_user_id: str) -> bool:
    """Check if a message is from the owner (Ketu). Handles multiple field formats.

    IMPORTANT: sender_id match is the most reliable signal. If sender_id is present
    and doesn't match owner_user_id, the message is from a customer — regardless of
    what the is_owner flag says (wwbun may incorrectly set is_owner=True for all msgs).
    """
    sid = m.get("sender_id")

    # If sender_id is present, use it as the authoritative source
    if sid and owner_user_id:
        # Exact match or suffix match (sender_id may be truncated)
        if sid == owner_user_id or owner_user_id.endswith(sid) or sid.endswith(owner_user_id):
            return True
        # sender_id present but doesn't match owner — this is a customer message
        return False

    # Fallback: no sender_id available, use explicit flags
    if _safe_bool(m.get("is_owner")):
        return True
    if _safe_bool(m.get("fromMe")) or _safe_bool(m.get("from_me")):
        return True
    return False


def _all_sender_ids_same(buffer: list[dict]) -> bool:
    """Detect if wwbun sent the same sender_id for ALL messages (broken data).
    When this happens, we can't distinguish customer vs owner messages.
    """
    sids = set(str(m.get("sender_id", "")).strip() for m in buffer if m.get("sender_id"))
    return len(sids) <= 1


def _is_owner_field_reliable(buffer: list[dict]) -> bool:
    """Check if the is_owner field has MIXED values (both True and False).
    When wwbun sends is_owner based on !contactId, it should have both values.
    If all are True or all are False/missing, it's unreliable.
    """
    has_true = False
    has_false = False
    for m in buffer:
        raw = m.get("is_owner")
        if raw is None:
            continue
        if _safe_bool(raw):
            has_true = True
        else:
            has_false = True
        if has_true and has_false:
            return True
    return False


def _is_owner_by_flag(m: dict) -> bool:
    """Check is_owner using only the is_owner/fromMe flags, ignoring sender_id."""
    if _safe_bool(m.get("is_owner")):
        return True
    if _safe_bool(m.get("fromMe")) or _safe_bool(m.get("from_me")):
        return True
    return False


def _count_quality_owner_messages(buffer: list[dict], owner_user_id: str) -> int:
    """Count quality messages for learning readiness threshold.

    When data is GOOD (sender_id works or is_owner is reliable): count pairs.
    When data is BROKEN (same sender_id + all is_owner same): count individual
    substantive messages. Claude will figure out roles from conversation context.
    """
    from learner.chat_learner import is_junk_message

    broken_sids = _all_sender_ids_same(buffer)
    use_flag = broken_sids and _is_owner_field_reliable(buffer)

    if broken_sids and not use_flag:
        # Data is fully broken — count individual quality messages for threshold
        # Claude can determine roles when it gets the full conversation
        count = 0
        for m in buffer:
            is_ai = _safe_bool(m.get("is_ai_generated"))
            if is_ai:
                continue
            text = m.get("content", "") or m.get("text", "") or m.get("body", "")
            if not text or is_junk_message(text):
                continue
            if len(text.split()) < 2:
                continue
            count += 1
        logger.info(f"[quality-count] broken data → counting individual msgs: {count}")
        return count

    # Good data — count pairs (more accurate)
    return len(_extract_pairs_from_buffer(buffer, owner_user_id))


def _extract_pairs_from_buffer(buffer: list[dict], owner_user_id: str) -> list[dict]:
    """Extract pairs from buffer — thin wrapper around _extract_conversation_pairs."""
    return _extract_conversation_pairs(buffer, owner_user_id)


def _flush_learning_buffer(owner_user_id: str) -> dict:
    """Flush the buffer: send all accumulated messages to Claude for learning."""
    buffer = _get_learning_buffer()
    if not buffer:
        return {"status": "empty_buffer", "count": 0}

    # Extract pairs for preview BEFORE flushing
    quality_pairs = _extract_pairs_from_buffer(buffer, owner_user_id)

    knowledge = extract_knowledge_from_wwbun_messages(
        messages=buffer,
        owner_user_id=owner_user_id,
    )
    result = apply_knowledge_updates(knowledge)
    invalidate_cache()

    filter_stats = knowledge.get("filter_stats", {})
    quality_messages = knowledge.get("quality_messages", [])

    # Log what Claude extracted (even if deduplicated as "already known")
    extracted_summary = []
    if knowledge.get("new_faqs"):
        extracted_summary.append(f"{len(knowledge['new_faqs'])} FAQs")
    if knowledge.get("style_patterns"):
        extracted_summary.append("style patterns")
    # price_updates and product_updates are no longer extracted from chat —
    # the catalog is the single source of truth for prices and product info.
    if knowledge.get("business_updates"):
        extracted_summary.append("business updates")
    if knowledge.get("prompt_evolution"):
        evo = knowledge["prompt_evolution"]
        evo_parts = []
        if evo.get("new_traits"): evo_parts.append(f"{len(evo['new_traits'])} traits")
        if evo.get("new_phrases"): evo_parts.append(f"{len(evo['new_phrases'])} phrases")
        if evo.get("new_rules"): evo_parts.append(f"{len(evo['new_rules'])} rules")
        if evo_parts:
            extracted_summary.append(f"prompt evolution ({', '.join(evo_parts)})")

    logger.info(
        f"[Buffer] Claude extracted: {extracted_summary or ['nothing']}. "
        f"Applied (new): {result.get('applied', [])}. "
        f"Status: {knowledge.get('status', 'unknown')}"
    )

    log_activity(
        source="wwbun-sync",
        action="batch-learned",
        details={
            "total_buffered": len(buffer),
            "quality_pairs_count": len(quality_pairs),
            "quality_messages_count": filter_stats.get("kept", 0),
            "junk_skipped": filter_stats.get("junk", 0),
            "too_short_skipped": filter_stats.get("too_short", 0),
            "quality_messages_preview": quality_messages[:5],
            "updates_applied": result.get("applied", []),
            "extracted_summary": extracted_summary,
        },
        items_count=result.get("count", 0),
    )

    # Clear buffer after successful learning
    _save_learning_buffer([])

    logger.info(
        f"[Buffer] Flushed {len(buffer)} messages → "
        f"{filter_stats.get('kept', 0)} quality → "
        f"{result.get('count', 0)} knowledge updates"
    )

    return {
        "status": "learned",
        "buffer_flushed": len(buffer),
        "filter_stats": filter_stats,
        "quality_messages": quality_messages,
        "quality_pairs": quality_pairs,
        "updates_applied": result,
    }


@app.post("/api/learn/wwbun-sync")
async def learn_from_wwbun(req: LearnWwbunRequest):
    """Learn from wwbun database messages.

    Messages are ACCUMULATED in a buffer. Claude is only called when
    we have 10+ quality manual message pairs — this gives Haiku enough
    data to extract meaningful patterns (traits, phrases, rules).

    Free features (enders, bought detection, repeat buyer) run immediately
    on every call — they use keyword matching, zero AI cost.
    """
    _save_owner_user_id(req.owner_user_id)

    # --- FREE features: run immediately on every call (zero AI cost) ---

    # Learn conversation-ending patterns
    ender_result = await asyncio.to_thread(
        learn_conversation_enders,
        messages=req.messages,
        owner_user_id=req.owner_user_id,
    )

    # Detect bought customers from chat signals
    bought_result = await asyncio.to_thread(
        detect_bought_customers_from_chat,
        messages=req.messages,
        owner_user_id=req.owner_user_id,
    )

    # Learn repeat/returning buyer patterns
    repeat_result = await asyncio.to_thread(
        learn_repeat_buyer_patterns,
        messages=req.messages,
        owner_user_id=req.owner_user_id,
    )

    # Learn Ketu-Only patterns from manual chats (keyword matching, free)
    try:
        from learner.realtime_learner import learn_ketu_only_from_manual_chat
        _learn_ketu_only_pairs(req.messages, req.owner_user_id, learn_ketu_only_from_manual_chat)
    except Exception as e:
        logger.warning(f"Ketu-only learning from wwbun failed (non-fatal): {e}")

    invalidate_ender_cache()
    _mark_run("whatsapp")

    # --- Track ALL customer messages for insights (free, no AI cost) ---
    insights_result = await asyncio.to_thread(
        track_wwbun_insights,
        messages=req.messages,
        owner_user_id=req.owner_user_id,
    )

    # --- PAID feature: accumulate messages for batch Claude learning ---

    # Debug: log what wwbun is sending so we can trace pairing issues
    if req.messages:
        unique_sids = set(str(m.get("sender_id", "")) for m in req.messages)
        owner_count = sum(1 for m in req.messages if _is_owner_message(m, req.owner_user_id))
        has_is_owner = any(m.get("is_owner") is not None for m in req.messages)
        has_content = sum(1 for m in req.messages if m.get("content"))
        has_text = sum(1 for m in req.messages if m.get("text"))
        has_ai_flag = sum(1 for m in req.messages if m.get("is_ai_generated") is not None)
        ai_true_count = sum(1 for m in req.messages if _safe_bool(m.get("is_ai_generated")))
        logger.info(
            f"[wwbun-sync DEBUG] owner_user_id={req.owner_user_id!r}, "
            f"total={len(req.messages)}, owner={owner_count}, cust={len(req.messages)-owner_count}, "
            f"has_is_owner_field={has_is_owner}, "
            f"unique_sender_ids={unique_sids}, "
            f"has_content={has_content}, has_text={has_text}, "
            f"has_ai_flag={has_ai_flag}, ai_true={ai_true_count}, "
            f"keys={list(req.messages[0].keys()) if req.messages else []}"
        )
        # Log first 3 messages in full for debugging
        for i, m in enumerate(req.messages[:3]):
            logger.info(
                f"[wwbun-sync MSG {i}] sender_id={m.get('sender_id')!r}, "
                f"is_owner={m.get('is_owner')!r}, is_ai={m.get('is_ai_generated')!r}, "
                f"chat_id={m.get('chat_id', m.get('remote_jid', 'N/A'))!r}, "
                f"content={str(m.get('content', '') or m.get('text', '') or m.get('body', ''))[:80]!r}, "
                f"_is_owner_result={_is_owner_message(m, req.owner_user_id)}"
            )
    else:
        logger.warning("[wwbun-sync DEBUG] Received EMPTY messages list!")

    # Add new messages to buffer — DEDUPLICATE to prevent wwbun sending
    # same "last 20 messages" across multiple syncs from creating duplicate pairs.
    buffer = _get_learning_buffer()

    # Build a set of existing message fingerprints for dedup
    existing_fps = set()
    for m in buffer:
        fp = _message_fingerprint(m)
        if fp:
            existing_fps.add(fp)

    # Only add genuinely new messages
    new_count = 0
    for m in req.messages:
        fp = _message_fingerprint(m)
        if fp and fp in existing_fps:
            continue  # Skip duplicate
        buffer.append(m)
        if fp:
            existing_fps.add(fp)
        new_count += 1

    logger.info(
        f"[wwbun-sync DEDUP] received={len(req.messages)}, new={new_count}, "
        f"duplicates_skipped={len(req.messages) - new_count}, buffer_total={len(buffer)}"
    )

    # Cap buffer at 500 messages to prevent unbounded growth
    if len(buffer) > 500:
        buffer = buffer[-500:]

    _save_learning_buffer(buffer)

    # Extract pairs for display and count quality for threshold (may differ when data is broken)
    buffer_pairs = _extract_pairs_from_buffer(buffer, req.owner_user_id)
    quality_count = _count_quality_owner_messages(buffer, req.owner_user_id)

    # Track Ketu's manual reply lengths and peak hours from new pairs
    if buffer_pairs:
        from core.reply_length import track_ketu_reply as _track_ketu_len
        from core.peak_hours import track_ketu_replies_batch
        manual_pairs = [p for p in buffer_pairs if not p.get("ai")]
        for p in manual_pairs:
            _track_ketu_len(p["ketu"])
        if manual_pairs:
            track_ketu_replies_batch(len(manual_pairs))
    logger.info(
        f"[wwbun-sync QUALITY] buffer_size={len(buffer)}, quality_count={quality_count}, "
        f"pairs_for_display={len(buffer_pairs)}, "
        f"threshold={_LEARNING_BUFFER_MIN_PAIRS}, will_flush={quality_count >= _LEARNING_BUFFER_MIN_PAIRS}, "
        f"pairs_preview={[p.get('customer','')[:30] + ' → ' + p.get('ketu','')[:30] for p in buffer_pairs[:3]]}"
    )

    # Decide: learn now or wait for more messages
    learn_result = {"status": "buffered", "count": 0}
    filter_stats = {"total": len(req.messages), "kept": 0, "junk": 0, "too_short": 0}
    quality_messages = []

    global _last_flush_time
    time_since_flush = time.time() - _last_flush_time
    cooldown_active = time_since_flush < _LEARNING_FLUSH_COOLDOWN
    force_flush = quality_count >= _LEARNING_FORCE_FLUSH_PAIRS  # Too much data waiting

    should_flush = quality_count >= _LEARNING_BUFFER_MIN_PAIRS and (not cooldown_active or force_flush)

    if force_flush and cooldown_active:
        logger.info(
            f"[wwbun-sync] Force flushing: {quality_count} quality pairs waiting "
            f"(>{_LEARNING_FORCE_FLUSH_PAIRS} threshold). Cooldown overridden."
        )

    if should_flush:
        # Enough data AND (cooldown expired OR too much data waiting)
        _last_flush_time = time.time()
        learn_result = await asyncio.to_thread(
            _flush_learning_buffer, req.owner_user_id
        )
        filter_stats = learn_result.get("filter_stats", filter_stats)
        quality_messages = learn_result.get("quality_messages", [])
        quality_pairs_preview = learn_result.get("quality_pairs", [])
        invalidate_cache()

        # Track wwbun sync stats for dashboard
        _track_wwbun_sync(
            total_messages=learn_result.get("buffer_flushed", len(req.messages)),
            quality_count=filter_stats.get("kept", 0),
            junk_count=filter_stats.get("junk", 0),
            short_count=filter_stats.get("too_short", 0),
            knowledge_count=learn_result.get("updates_applied", {}).get("count", 0),
            enders_learned=ender_result.get("new_enders", 0),
            quality_previews=quality_pairs_preview or quality_messages,
            details={
                "filter_stats": filter_stats,
                "updates_applied": learn_result.get("updates_applied", {}).get("applied", [])[:5],
                "enders": ender_result.get("examples", [])[:3],
            },
        )
    else:
        # Not enough data or cooldown active — buffer the messages
        reason = "cooldown_active" if cooldown_active and quality_count >= _LEARNING_BUFFER_MIN_PAIRS else "insufficient_data"
        cooldown_remaining = max(0, int(_LEARNING_FLUSH_COOLDOWN - time_since_flush)) if cooldown_active else 0
        if reason == "cooldown_active":
            logger.info(
                f"[wwbun-sync] Enough data ({quality_count} pairs) but cooldown active "
                f"({cooldown_remaining}s remaining). Buffering for next flush."
            )
        log_activity(
            source="wwbun-sync",
            action="buffered",
            details={
                "messages_added": len(req.messages),
                "buffer_total": len(buffer),
                "quality_in_buffer": quality_count,
                "needed": _LEARNING_BUFFER_MIN_PAIRS,
                "remaining": max(0, _LEARNING_BUFFER_MIN_PAIRS - quality_count),
                "reason": reason,
                "cooldown_remaining_sec": cooldown_remaining,
            },
            items_count=0,
        )

        # Still track sync stats even when buffering
        # Use buffer_pairs directly — same pairs that were counted = same pairs shown
        _track_wwbun_sync(
            total_messages=len(req.messages),
            quality_count=quality_count,
            junk_count=0,
            short_count=0,
            knowledge_count=0,
            enders_learned=ender_result.get("new_enders", 0),
            quality_previews=buffer_pairs,
            details={
                "buffered": True,
                "quality_in_buffer": quality_count,
                "needed": _LEARNING_BUFFER_MIN_PAIRS,
            },
        )

    return {
        "status": learn_result.get("status", "buffered"),
        "buffer": {
            "quality_pairs": quality_count,
            "threshold": _LEARNING_BUFFER_MIN_PAIRS,
            "force_flush_threshold": _LEARNING_FORCE_FLUSH_PAIRS,
            "total_buffered": len(buffer) if learn_result.get("status") == "buffered" else 0,
            "flush_cooldown_sec": max(0, int(_LEARNING_FLUSH_COOLDOWN - (time.time() - _last_flush_time))),
        },
        "learning": {
            "filter_stats": filter_stats,
            "quality_messages": quality_messages,
            "knowledge_extracted": {k: v for k, v in learn_result.items() if k not in ("status", "buffer_flushed", "filter_stats", "quality_messages", "updates_applied")},
            "updates_applied": learn_result.get("updates_applied", {}),
        },
        "free_features": {
            "conversation_enders": ender_result,
            "bought_detection": {
                "customers_marked": bought_result.get("customers_marked_bought", 0),
                "phones": bought_result.get("phones", []),
            },
            "repeat_buyer_learning": {
                "new_repeat_replies": repeat_result.get("new_repeat_replies", 0),
                "new_returning_replies": repeat_result.get("new_returning_replies", 0),
            },
        },
    }


class FlushBufferRequest(BaseModel):
    owner_user_id: str


@app.post("/api/learn/clear-buffer")
async def clear_learning_buffer():
    """Clear the learning buffer without learning — use when buffer has bad data."""
    buffer = _get_learning_buffer()
    count = len(buffer)
    _save_learning_buffer([])
    logger.info(f"[Buffer] Cleared {count} messages from buffer")
    return {"status": "cleared", "messages_cleared": count}


@app.post("/api/learn/flush-buffer")
async def flush_learning_buffer(req: FlushBufferRequest):
    """Manually flush the learning buffer — force Claude to learn from whatever is buffered.

    Use this if you want to force learning even with fewer than 10 quality pairs.
    """
    buffer = _get_learning_buffer()
    if not buffer:
        return {"status": "empty", "message": "No messages in buffer"}

    quality_count = len(_extract_pairs_from_buffer(buffer, req.owner_user_id))

    result = await asyncio.to_thread(_flush_learning_buffer, req.owner_user_id)

    return {
        "status": result.get("status", "error"),
        "messages_flushed": result.get("buffer_flushed", 0),
        "quality_pairs": quality_count,
        "updates_applied": result.get("updates_applied", {}),
    }


@app.get("/api/learn/buffer-status")
async def learning_buffer_status(owner_user_id: str = ""):
    """Check current learning buffer status."""
    if not owner_user_id:
        owner_user_id = _get_owner_user_id()
    buffer = _get_learning_buffer()
    # quality_count = threshold count (individual msgs when broken, pairs when good)
    # buffer_pairs = pairs for display on dashboard
    buffer_pairs = _extract_pairs_from_buffer(buffer, owner_user_id) if owner_user_id else []
    quality = _count_quality_owner_messages(buffer, owner_user_id) if owner_user_id else 0

    # Show sample messages for debugging
    sample_msgs = []
    for m in buffer[-10:]:
        raw_owner = m.get("is_owner")
        raw_ai = m.get("is_ai_generated")
        parsed_owner = _is_owner_message(m, owner_user_id)
        parsed_ai = _safe_bool(raw_ai)
        sample_msgs.append({
            "content": (m.get("content", "") or m.get("text", "") or m.get("body", ""))[:80],
            "is_owner_raw": repr(raw_owner),
            "is_owner_parsed": parsed_owner,
            "sender_id": str(m.get("sender_id", ""))[-6:],
            "is_ai_raw": repr(raw_ai),
            "is_ai_parsed": parsed_ai,
            "all_keys": list(m.keys()),
        })

    # Count owner vs customer in buffer
    owner_count = sum(1 for m in buffer if _is_owner_message(m, owner_user_id))
    customer_count = len(buffer) - owner_count
    unique_sids = list(set(str(m.get("sender_id", ""))[-8:] for m in buffer))

    return {
        "total_buffered": len(buffer),
        "owner_user_id": owner_user_id[:10] + "..." if len(owner_user_id) > 10 else owner_user_id,
        "owner_messages": owner_count,
        "customer_messages": customer_count,
        "unique_sender_ids": unique_sids,
        "quality_pairs": quality,
        "display_pairs": len(buffer_pairs),
        "threshold": _LEARNING_BUFFER_MIN_PAIRS,
        "remaining_needed": max(0, _LEARNING_BUFFER_MIN_PAIRS - quality),
        "ready_to_learn": quality >= _LEARNING_BUFFER_MIN_PAIRS,
        "sample_messages": sample_msgs,
    }


class LearnYouTubeRequest(BaseModel):
    video_url: str
    video_title: str = ""


@app.post("/api/learn/youtube")
async def learn_from_youtube(req: LearnYouTubeRequest):
    """Learn from a YouTube video transcript.

    Extracts product info, pricing, business knowledge from video.
    """
    result = await asyncio.to_thread(process_video, req.video_url, req.video_title)
    if result.get("knowledge"):
        applied = await asyncio.to_thread(apply_knowledge_updates, result["knowledge"])
        invalidate_cache()
        log_activity(
            source="youtube",
            action="learned",
            details={
                "video_url": req.video_url,
                "video_title": req.video_title,
                "updates_applied": applied.get("applied", []),
            },
            items_count=applied.get("count", 0),
        )
    return result


@app.post("/api/learn/youtube-scan")
async def scan_youtube_channel_endpoint():
    """Manually trigger YouTube channel scan for new videos.

    This runs the same check that the background scheduler does every 12 hours.
    Finds new videos, extracts transcripts, learns knowledge automatically.
    """
    await check_youtube_channel()
    return {"status": "scan_complete"}


@app.post("/api/learn/youtube-backfill")
async def youtube_backfill_endpoint(batch_size: int = 3):
    """Manually trigger YouTube backfill — process old videos in batch.

    First call fetches ALL video IDs from channel.
    Each call processes `batch_size` unprocessed videos (default 3).
    Auto-runs every 24 hours in background, but you can trigger manually too.

    Safety: Max 5 transcript fetches per day (survives redeploys).
    30-second delay between each video. 6-hour cooldown on rate limit.

    Query params:
    - batch_size: number of videos to process this run (default 3, max 5)
    """
    batch_size = min(batch_size, 5)  # Hard cap at 5 to protect YouTube channel
    result = await backfill_youtube_channel(batch_size=batch_size)
    return result


@app.get("/api/learn/youtube-backfill/status")
async def youtube_backfill_status():
    """Check YouTube backfill progress — how many videos done, how many left."""
    return get_backfill_status()


# --- YouTube OAuth Setup (one-time authorization flow) ---

@app.get("/api/youtube/auth")
async def youtube_oauth_start():
    """Step 1: Get the authorization URL. Open this in your browser to authorize.

    Prerequisites:
    1. Set YOUTUBE_CLIENT_ID in Railway env vars
    2. Set YOUTUBE_CLIENT_SECRET in Railway env vars
    3. Open the returned URL in your browser
    4. After authorizing, Google redirects to /api/youtube/callback with a code
    5. The callback automatically exchanges the code for a refresh_token
    6. Copy the refresh_token and set it as YOUTUBE_REFRESH_TOKEN in Railway
    """
    if not settings.youtube_client_id or not settings.youtube_client_secret:
        return {
            "status": "not_configured",
            "message": "Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in Railway env vars first.",
            "steps": [
                "1. Go to Google Cloud Console > APIs & Services > Credentials",
                "2. Create OAuth 2.0 Client ID (type: Web application)",
                "3. Add redirect URI: https://YOUR-RAILWAY-URL/api/youtube/callback",
                "4. Copy Client ID → YOUTUBE_CLIENT_ID env var",
                "5. Copy Client Secret → YOUTUBE_CLIENT_SECRET env var",
                "6. Redeploy, then visit this endpoint again",
            ],
        }

    from urllib.parse import urlencode

    # Request access to YouTube captions (force.captions scope)
    auth_params = urlencode({
        "client_id": settings.youtube_client_id,
        "redirect_uri": _get_oauth_redirect_uri(),
        "response_type": "code",
        "scope": "https://www.googleapis.com/auth/youtube.force-ssl",
        "access_type": "offline",  # This gives us a refresh_token
        "prompt": "consent",  # Force consent screen to always get refresh_token
    })

    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{auth_params}"

    return {
        "status": "ready",
        "message": "Open this URL in your browser to authorize YouTube access:",
        "auth_url": auth_url,
        "next_step": "After authorizing, Google will redirect to /api/youtube/callback automatically.",
    }


@app.get("/api/youtube/callback")
async def youtube_oauth_callback(code: str = "", error: str = ""):
    """Step 2: Google redirects here after you authorize. Exchanges code for refresh_token."""
    if error:
        return {"status": "error", "message": f"Authorization denied: {error}"}

    if not code:
        return {"status": "error", "message": "No authorization code received"}

    import httpx as _httpx

    try:
        resp = _httpx.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.youtube_client_id,
                "client_secret": settings.youtube_client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": _get_oauth_redirect_uri(),
            },
            timeout=15,
        )
        resp.raise_for_status()
        token_data = resp.json()

        refresh_token = token_data.get("refresh_token")
        access_token = token_data.get("access_token")

        if not refresh_token:
            return {
                "status": "error",
                "message": "No refresh_token received. Try revoking app access in Google Account settings and authorize again.",
                "token_data": token_data,
            }

        # Test the token — try listing captions for any video
        test_result = "not_tested"
        if access_token and settings.youtube_channel_id:
            try:
                test_resp = _httpx.get(
                    "https://www.googleapis.com/youtube/v3/channels",
                    params={"part": "snippet", "id": settings.youtube_channel_id},
                    headers={"Authorization": f"Bearer {access_token}"},
                    timeout=10,
                )
                if test_resp.status_code == 200:
                    ch_data = test_resp.json()
                    ch_name = ch_data.get("items", [{}])[0].get("snippet", {}).get("title", "Unknown")
                    test_result = f"Connected to channel: {ch_name}"
            except Exception:
                test_result = "token_valid_but_test_failed"

        return {
            "status": "success",
            "message": "YouTube OAuth authorized! Copy the refresh_token below and set it as YOUTUBE_REFRESH_TOKEN in Railway.",
            "refresh_token": refresh_token,
            "test": test_result,
            "next_steps": [
                f"1. Copy this refresh_token: {refresh_token}",
                "2. Go to Railway > Variables",
                "3. Add: YOUTUBE_REFRESH_TOKEN = (paste the token)",
                "4. Redeploy — Digital Ketu will now use the official YouTube Captions API!",
            ],
        }

    except Exception as e:
        return {"status": "error", "message": f"Token exchange failed: {e}"}


def _get_oauth_redirect_uri() -> str:
    """Build the OAuth redirect URI based on the current server."""
    # In production (Railway), use HTTPS
    # Check common env vars for the public URL
    import os
    railway_url = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
    if railway_url:
        return f"https://{railway_url}/api/youtube/callback"
    # Fallback for local dev
    return f"http://localhost:{settings.port}/api/youtube/callback"


@app.get("/api/learned-files")
async def list_learned_files_endpoint():
    """List all learned files with rich details (title, key_points for YT videos)."""
    import json as _json
    from core.database import is_db_available, load_learned_file

    def _parse_yt_details(filename: str, content: str | None) -> dict:
        """Extract title and key_points from a YouTube learned file."""
        info = {"file": filename}
        if not content:
            return info
        try:
            data = _json.loads(content)
            info["title"] = data.get("title", "")
            info["video_url"] = data.get("video_url", "")
            knowledge = data.get("knowledge", {})
            info["key_points"] = knowledge.get("key_points", [])
            info["has_product_info"] = bool(knowledge.get("product_info"))
            info["has_pricing"] = bool(knowledge.get("pricing"))
            info["has_faqs"] = bool(knowledge.get("faqs_covered"))
        except Exception:
            pass
        return info

    # Try DB first
    if is_db_available():
        from core.database import list_learned_files, count_learned_files
        raw_files = list_learned_files()
        files = []
        for f in raw_files:
            fname = f.get("file", "")
            entry = {**f}
            if fname.startswith("yt_"):
                content = load_learned_file(fname)
                entry.update(_parse_yt_details(fname, content))
            files.append(entry)
        return {"total": count_learned_files(), "files": files}

    # Fallback to local files
    learned_dir = KNOWLEDGE_DIR / "learned"
    if not learned_dir.exists():
        return {"total": 0, "files": []}

    files = []
    for f in sorted(learned_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not f.name.startswith("_"):
            entry = {"file": f.name, "size": f.stat().st_size}
            if f.name.startswith("yt_"):
                try:
                    entry.update(_parse_yt_details(f.name, f.read_text(encoding="utf-8")))
                except Exception:
                    pass
            files.append(entry)
    return {"total": len(files), "files": files}


@app.get("/api/youtube/backfill-status")
async def youtube_backfill_status_short():
    """Shortcut for YouTube backfill status."""
    return get_backfill_status()


@app.get("/api/youtube/processed-videos")
async def youtube_processed_videos():
    """Get list of all YouTube videos with their processing status.

    Combines backfill state (all video IDs + titles) with processed_videos
    set to show which videos have been processed and which are pending.
    """
    from core.database import is_db_available, kv_get

    all_videos = []
    processed_ids = set()

    if is_db_available():
        # Get backfill state (has all video IDs with titles)
        backfill = kv_get("_backfill_state", {})
        if backfill and isinstance(backfill, dict):
            all_videos = backfill.get("all_video_ids", [])

        # Get processed video IDs
        pv = kv_get("_processed_videos", {})
        if pv and isinstance(pv, dict):
            processed_ids = set(pv.get("video_ids", []))

    videos = []
    for v in all_videos:
        vid = v.get("video_id", "") if isinstance(v, dict) else str(v)
        title = v.get("title", "Untitled") if isinstance(v, dict) else "Untitled"
        videos.append({
            "video_id": vid,
            "title": title,
            "processed": vid in processed_ids,
        })

    return {
        "total": len(videos),
        "processed": len([v for v in videos if v["processed"]]),
        "pending": len([v for v in videos if not v["processed"]]),
        "videos": videos,
    }


@app.post("/api/learn/catalog-sync")
async def sync_catalog_endpoint():
    """Manually trigger catalog sync from GitHub repo.

    Fetches latest products.json from github.com/thakyanamtumhara/catalog
    and updates Digital Ketu's product knowledge with latest prices, colors, etc.

    This also runs automatically every 6 hours via the background scheduler.
    """
    result = await sync_catalog()
    if result.get("status") == "ok":
        invalidate_cache()
        diff = result.get("diff", {})
        log_activity(
            source="catalog-sync",
            action="synced",
            details={
                "products_synced": result.get("products_synced", 0),
                "added": diff.get("added", []),
                "removed": diff.get("removed", []),
                "price_changes": diff.get("price_changes", []),
            },
            items_count=result.get("products_synced", 0),
        )
    return result


# --- Correction Learning ---


class CorrectionRequest(BaseModel):
    customer_message: str
    ai_reply: str
    ketu_correction: str
    customer_phone: str = ""
    customer_name: str = ""


@app.post("/api/learn/correction")
async def learn_correction(req: CorrectionRequest):
    """Learn when Ketu overrides an AI reply.

    wwbun calls this when Ketu manually edits/replaces a Digital Ketu reply.
    This is the most powerful learning signal — Ketu directly shows
    how the AI should have replied.

    Send:
    - customer_message: what the customer asked
    - ai_reply: what Digital Ketu replied (wrong/incomplete)
    - ketu_correction: what Ketu actually sent instead
    """
    result = await asyncio.to_thread(
        learn_from_correction,
        customer_message=req.customer_message,
        ai_reply=req.ai_reply,
        ketu_correction=req.ketu_correction,
        customer_phone=req.customer_phone,
        customer_name=req.customer_name,
    )

    if result.get("status") == "learned":
        log_activity(
            source="correction-learner",
            action="learned",
            details={
                "customer_message": req.customer_message[:80],
                "ai_reply": req.ai_reply[:80],
                "ketu_correction": req.ketu_correction[:80],
                "what_went_wrong": result.get("what_went_wrong", ""),
                "updates_applied": result.get("updates_applied", []),
            },
            items_count=result.get("count", 0),
        )

    return result


# --- Voice Note Learning ---


class VoiceNoteRequest(BaseModel):
    audio_url: str = ""
    media_id: str = ""
    context: str = ""
    language: str = "hi"


@app.post("/api/learn/voice-note")
async def learn_voice_note(req: VoiceNoteRequest):
    """Learn from Ketu's voice notes.

    Transcribes the voice note and extracts:
    - Product knowledge shared verbally
    - Ketu's speaking style and phrases
    - Business info mentioned casually
    - New FAQs from verbal explanations

    Send either:
    - media_id: WhatsApp media ID (will download from Meta API)
    - audio_url: Direct URL to audio file
    """
    from learner.audio_transcriber import download_whatsapp_media

    audio_bytes = None

    if req.media_id:
        audio_bytes = await download_whatsapp_media(req.media_id)
    elif req.audio_url:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(req.audio_url)
                resp.raise_for_status()
                audio_bytes = resp.content
        except Exception as e:
            return {"status": "error", "detail": f"Failed to download audio: {e}"}

    if not audio_bytes:
        return {"status": "error", "detail": "No audio data — provide media_id or audio_url"}

    result = await learn_from_voice_note(
        audio_bytes=audio_bytes,
        context=req.context,
        language=req.language,
    )

    if result.get("status") == "ok":
        log_activity(
            source="voice-learner",
            action="learned",
            details={
                "transcript_preview": result.get("transcript", "")[:100],
                "context": req.context,
                "updates_applied": result.get("updates_applied", []),
            },
            items_count=result.get("count", 0),
        )
        # Track that this voice note added to knowledge
        from learner.audio_transcriber import track_knowledge_from_voice
        track_knowledge_from_voice()

    return result


# --- Conversation Log (for correction review) ---


@app.get("/api/conversations/recent")
async def recent_conversations(limit: int = 20):
    """Recent AI conversations for review.

    Ketu uses this to check what AI replied to customers.
    Also used by wwbun to detect corrections.
    """
    return {"conversations": get_recent_conversations(limit)}


@app.get("/api/conversations/last-reply")
async def last_reply_to_customer(phone: str):
    """Get the last AI reply to a specific customer.

    wwbun calls this when Ketu manually messages a customer —
    to check if AI already replied (potential correction scenario).
    """
    entry = get_last_ai_reply(phone)
    if not entry:
        return {"found": False}
    return {"found": True, "conversation": entry}


# --- Ketu Manual Reply (Shut Up Mode) ---


class KetuRepliedRequest(BaseModel):
    customer_phone: str
    ketu_message: str = ""
    minutes: float = 10


@app.post("/api/ketu-replied")
async def api_ketu_replied(req: KetuRepliedRequest):
    """Tell Digital Ketu that Ketu manually replied to a customer.

    wwbun calls this when Ketu types a manual message in WhatsApp.
    This activates "shut up" mode — AI won't reply to this customer
    for the next N minutes (default 10), because Ketu is handling it.

    If the customer asks a NEW question (contains ?, price, rate, etc.),
    the cooldown breaks automatically and AI resumes.

    Also logs the customer's last question to the ketu-only queue so
    the dashboard shows what Ketu had to handle manually.
    """
    ketu_manual_reply(req.customer_phone, reply_text=req.ketu_message or "")

    # Log the customer's last question to ketu-only queue
    # so "Needs Ketu's Reply" section shows what Ketu handled
    from core.engine import get_conversation_history
    from core.ketu_only import log_manual_takeover
    history = get_conversation_history(req.customer_phone)
    customer_last_msg = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            customer_last_msg = msg.get("content", "")
            break
    # Only log if we found a customer message and it's not trivial
    if customer_last_msg and len(customer_last_msg.strip()) > 3:
        log_manual_takeover(
            customer_phone=req.customer_phone,
            customer_message=customer_last_msg,
            ketu_reply=req.ketu_message or "",
        )

    log_activity(
        source="ketu-replied",
        action="shutup-activated",
        details={
            "customer_phone": req.customer_phone[-4:] if req.customer_phone else "unknown",
            "ketu_message": req.ketu_message[:80] if req.ketu_message else "",
            "cooldown_minutes": req.minutes,
        },
    )
    return {
        "status": "ok",
        "message": f"AI silenced for {req.customer_phone[-4:]} for {req.minutes} minutes",
    }


# --- Realtime Learner Stats ---


@app.get("/api/learn/realtime/stats")
async def realtime_learner_stats():
    """Get realtime learner statistics — buffer size, learning progress."""
    return get_realtime_stats()


@app.get("/api/learn/correction-stats")
async def correction_learning_stats():
    """Correction learning statistics — what mistakes AI keeps making.

    Shows error categories, how many times each type of mistake happened,
    and whether an auto-rule was generated to prevent it.
    """
    return get_correction_stats()


@app.get("/api/costs")
async def api_costs():
    """API cost tracking — today, this week, this month.

    Shows cost breakdown by source (replies, learning, YouTube analysis),
    by model (Haiku vs Sonnet), and per-reply average cost.
    Costs shown in both USD and INR.
    """
    return get_cost_summary()


@app.get("/api/audio/stats")
async def audio_stats():
    """Audio transcription stats — how many voice notes transcribed today/total.

    Shows that Digital Ketu is receiving audio messages, converting them to text
    via Whisper, and using the text for replies and knowledge.
    """
    from learner.audio_transcriber import get_audio_stats
    return get_audio_stats()


@app.get("/api/rate-limit/stats")
async def rate_limit_stats():
    """Rate limiting stats for dashboard.

    Shows how many messages were blocked today, top blocked phones,
    and current limit values.
    """
    from integrations.whatsapp.webhook import get_rate_limit_stats
    return get_rate_limit_stats()


# --- FAQ Validation ---


@app.get("/api/faq/health")
async def faq_health():
    """FAQ health report — shows active, inactive, price-warned FAQs.

    Use this to see which FAQs are outdated or have wrong prices.
    """
    return get_faq_health_report()


@app.post("/api/faq/validate")
async def faq_validate():
    """Manually trigger FAQ validation against current catalog.

    Cross-checks all FAQs with product catalog:
    - FAQs mentioning removed products → deactivated
    - FAQs with wrong prices → flagged

    This also runs automatically after every catalog sync (every 6h).
    """
    result = validate_faqs_against_catalog()
    if result.get("deactivated") or result.get("price_warnings"):
        invalidate_cache()
        log_activity(
            source="faq-validator",
            action="validated",
            details={
                "deactivated": result.get("deactivated", 0),
                "price_warnings": result.get("price_warnings", 0),
                "flagged": result.get("flagged_faqs", []),
                "price_issues": result.get("price_issues", []),
            },
            items_count=result.get("deactivated", 0),
        )
    return result


class ReactivateFaqRequest(BaseModel):
    question: str


@app.post("/api/faq/reactivate")
async def faq_reactivate(req: ReactivateFaqRequest):
    """Reactivate a previously deactivated FAQ (owner override).

    If a product comes back or the FAQ is still valid, use this to bring it back.
    """
    result = reactivate_faq(req.question)
    if result.get("status") == "reactivated":
        invalidate_cache()
    return result


# --- Dashboard & Monitoring ---


@app.get("/api/dashboard")
async def dashboard():
    """Live dashboard — today's summary + recent activity.

    Shows what Digital Ketu learned today, from where, and how much.
    """
    return {
        "today": get_today_summary(),
        "recent_activity": get_activity_log(limit=20),
    }


@app.get("/api/dashboard/activity")
async def dashboard_activity(
    limit: int = 50,
    source: str | None = None,
    date: str | None = None,
):
    """Full activity log with filters.

    Query params:
    - limit: max entries (default 50)
    - source: filter by source (wwbun-sync, whatsapp-export, youtube, catalog-sync, api-reply)
    - date: filter by date (e.g., "04 Mar 2026")
    """
    return {
        "entries": get_activity_log(limit=limit, source_filter=source, date_filter=date),
    }


@app.get("/api/learn/history")
async def learn_history(limit: int = 20):
    """Show what messages Digital Ketu learned from.

    Returns recent wwbun-sync events with:
    - Which quality messages were used for learning
    - How many junk messages were filtered out
    - What knowledge was extracted
    """
    entries = get_activity_log(limit=limit, source_filter="wwbun-sync")

    history = []
    for entry in entries:
        details = entry.get("details", {})
        history.append({
            "time": entry.get("time", ""),
            "date": entry.get("date", ""),
            "total_received": details.get("total_messages", 0),
            "manual_messages": details.get("manual_messages", 0),
            "quality_kept": details.get("quality_messages_count", 0),
            "junk_skipped": details.get("junk_skipped", 0),
            "too_short_skipped": details.get("too_short_skipped", 0),
            "quality_messages": details.get("quality_messages_preview", []),
            "learned": details.get("updates_applied", []),
        })

    return {
        "total_syncs": len(history),
        "history": history,
    }


@app.get("/api/dashboard/storage")
async def dashboard_storage():
    """Knowledge base storage stats.

    Shows file sizes, item counts, what's stored where.
    """
    return get_storage_stats()


# --- Customer Insights ---


@app.get("/api/insights/customers")
async def customer_insights():
    """Customer message analytics — top customers, peak hours."""
    return get_customer_insights()


@app.post("/api/insights/customers/reset")
async def reset_customer_insights():
    """Reset customer insights counters. Use after fixing tracking bugs."""
    from core.database import is_db_available, kv_set
    if is_db_available():
        kv_set("customer_insights", {
            "message_counts": {},
            "names": {},
            "hourly": {},
        })
        kv_set("customer_insights_total", {
            "message_counts": {},
            "names": {},
            "hourly": {},
        })
    # Also clear in-memory
    from core.engine import _customer_message_counts, _customer_names, _hourly_message_counts
    from core.engine import _total_customer_counts, _total_hourly_counts
    _customer_message_counts.clear()
    _customer_names.clear()
    _hourly_message_counts.clear()
    _total_customer_counts.clear()
    _total_hourly_counts.clear()
    return {"status": "reset", "message": "Both AI and Total customer insights cleared. Will rebuild from incoming messages."}


# --- FAQ Hit Rate ---


@app.get("/api/insights/faq-hits")
async def faq_hit_rates():
    """FAQ hit rates — which FAQs are used most in replies."""
    return {"faq_hits": get_faq_hit_rates()}


# --- Customer Memory ---


@app.get("/api/customer/{phone}")
async def get_customer(phone: str):
    """Get a customer's memory profile — interests, stage, preferences."""
    profile = get_customer_profile(phone)
    return {"phone": phone, "profile": profile}


@app.get("/api/customers/summary")
async def customers_summary():
    """Customer memory summary — total customers, by buying stage."""
    return get_all_profiles_summary()


# --- Follow-up Intelligence ---


@app.get("/api/followup/pending")
async def pending_followups():
    """Get customers who need a follow-up (interested but didn't order).

    Only shows when auto_reply AND followup are both enabled.
    These are pre-generated simple messages — no AI cost.
    """
    return {"followups": get_pending_followups()}


@app.post("/api/followup/send/{phone}")
async def send_followup(phone: str):
    """Execute follow-up for a specific customer.

    Returns the message to send. The actual WhatsApp send is wwbun's job.
    Marks the customer so they don't get another follow-up.
    """
    result = execute_followup(phone)
    if result["status"] == "sent":
        log_activity(
            source="followup",
            action="sent",
            details={
                "customer_phone": phone[-4:] if phone else "unknown",
                "message": result.get("message", ""),
            },
            items_count=1,
        )
    return result


@app.get("/api/followup/stats")
async def followup_stats_endpoint():
    """Follow-up intelligence stats for dashboard."""
    return get_followup_stats()


@app.post("/api/followup/send-all")
async def send_all_followups():
    """Send follow-ups to all pending customers at once.

    Returns list of messages to send. wwbun handles actual delivery.
    No AI cost — just template messages.
    """
    pending = get_pending_followups()
    results = []
    for customer in pending:
        result = execute_followup(customer["phone"])
        results.append(result)
        if result["status"] == "sent":
            log_activity(
                source="followup",
                action="sent",
                details={
                    "customer_phone": customer["phone"][-4:] if customer["phone"] else "unknown",
                    "message": result.get("message", ""),
                },
                items_count=1,
            )
    return {
        "total_sent": len([r for r in results if r["status"] == "sent"]),
        "already_sent": len([r for r in results if r["status"] == "already_sent"]),
        "results": results,
    }


# --- WhatsApp Message Editing ---


class EditMessageRequest(BaseModel):
    phone: str
    new_text: str
    message_id: str = ""
    send_to_customer: bool = False  # Default: DO NOT send/edit to buyer


@app.post("/api/whatsapp/edit")
async def edit_whatsapp_message(req: EditMessageRequest):
    """Save Ketu's correction and learn from it — WITHOUT resending to buyer.

    When AI replies wrong and Ketu edits:
    1. Save the correction locally
    2. Digital Ketu learns from it (new FAQ, style, rules)
    3. Mark conversation as corrected
    4. DO NOT send/edit anything to the buyer (no message flooding)

    The buyer already got the AI reply. Sending again = unnecessary messages.
    Digital Ketu silently learns and gives better replies next time.

    Set send_to_customer=True ONLY if you explicitly want to edit on WhatsApp.
    """
    from integrations.whatsapp.sender import edit_message, edit_last_message, get_last_sent_message
    from core.conversation_log import get_last_ai_reply, mark_corrected
    from learner.realtime_learner import learn_from_correction

    # Get the original AI reply (needed for learning)
    original_ai_data = get_last_sent_message(req.phone)
    original_text = original_ai_data["text"] if original_ai_data else ""

    # Look up what the customer asked and what AI replied
    last_convo = get_last_ai_reply(req.phone)

    # --- OPTIONAL: Edit on WhatsApp (only if Ketu explicitly wants) ---
    whatsapp_edited = False
    if req.send_to_customer:
        if req.message_id:
            result = await edit_message(to=req.phone, message_id=req.message_id, new_text=req.new_text)
        else:
            result = await edit_last_message(to=req.phone, new_text=req.new_text)
        whatsapp_edited = result is not None

    # --- CORRECTION LEARNING (always happens, no message sent to buyer) ---
    if last_convo and (original_text or last_convo.get("ai_reply")):
        customer_message = last_convo.get("customer_message", "")
        ai_reply = last_convo.get("ai_reply", original_text)
        customer_name = last_convo.get("customer_name", "")

        # Only learn if the edit is actually different from the AI reply
        if req.new_text.strip() != ai_reply.strip() and customer_message:
            # Mark conversation as corrected
            mark_corrected(req.phone)

            log_activity(
                source="whatsapp",
                action="correction-saved",
                details={
                    "customer_phone": req.phone[-4:] if req.phone else "unknown",
                    "ketu_correction": req.new_text[:100],
                    "original_ai_reply": ai_reply[:100],
                    "sent_to_customer": whatsapp_edited,
                },
            )

            # Learn from the correction in background thread
            import threading
            def _learn_from_edit():
                try:
                    learn_result = learn_from_correction(
                        customer_message=customer_message,
                        ai_reply=ai_reply,
                        ketu_correction=req.new_text,
                        customer_phone=req.phone,
                        customer_name=customer_name,
                    )
                    if learn_result.get("status") == "learned":
                        log_activity(
                            source="correction-learner",
                            action="learned-from-edit",
                            details={
                                "customer_message": customer_message[:80],
                                "ai_reply": ai_reply[:80],
                                "ketu_correction": req.new_text[:80],
                                "what_went_wrong": learn_result.get("what_went_wrong", ""),
                                "updates_applied": learn_result.get("updates_applied", []),
                                "via": "whatsapp-edit",
                            },
                            items_count=learn_result.get("count", 0),
                        )
                        logger.info(
                            f"[Edit→Learn] Learned from edit for ...{req.phone[-4:]}: "
                            f"{learn_result.get('what_went_wrong', 'unknown')}"
                        )
                except Exception as e:
                    logger.error(f"[Edit→Learn] Learning failed (non-fatal): {e}")

            thread = threading.Thread(target=_learn_from_edit, daemon=True)
            thread.start()

            return {
                "status": "saved",
                "learning": "correction sent to learner",
                "sent_to_customer": whatsapp_edited,
                "original_ai_reply": ai_reply[:100],
                "customer_question": customer_message[:100],
            }

        return {
            "status": "saved",
            "learning": "no change detected (same text)",
            "sent_to_customer": whatsapp_edited,
        }

    return {
        "status": "saved",
        "learning": "no matching conversation found — correction noted",
        "sent_to_customer": whatsapp_edited,
    }


@app.get("/api/whatsapp/last-sent/{phone}")
async def last_sent_message(phone: str):
    """Check the last message sent to a customer — is it still editable?"""
    from integrations.whatsapp.sender import get_last_sent_message
    last = get_last_sent_message(phone)
    if not last:
        return {"found": False}
    return {"found": True, **last}


# --- Ketu-Only Queue (Questions deferred to real Ketu) ---


@app.get("/api/ketu-only/queue")
async def ketu_only_queue(limit: int = 50, pending_only: bool = False):
    """Get questions deferred to real Ketu.

    These are questions the AI cannot answer (stock timelines, order status,
    custom pricing, etc.) and has told the customer to wait for Ketu's reply.
    """
    from core.ketu_only import get_deferred_queue, get_deferred_stats
    return {
        "queue": get_deferred_queue(limit=limit, pending_only=pending_only),
        "stats": get_deferred_stats(),
    }


@app.post("/api/ketu-only/resolve/{index}")
async def ketu_only_resolve(index: int):
    """Mark a deferred question as resolved (Ketu replied manually)."""
    from core.ketu_only import mark_resolved
    ok = mark_resolved(index)
    return {"resolved": ok}


@app.get("/api/ketu-only/categories")
async def ketu_only_categories():
    """Get the current ketu-only question categories."""
    from core.ketu_only import _load_config
    config = _load_config()
    return {
        "categories": [
            {"id": c["id"], "name": c["name"], "description": c.get("description", ""),
             "keywords_count": len(c.get("keywords", [])), "patterns_count": len(c.get("patterns", []))}
            for c in config.get("categories", [])
        ],
        "learned_count": len(config.get("learned_patterns", [])),
    }


# --- Knowledge Backup/Export ---


@app.get("/api/backup")
async def backup_knowledge():
    """Download entire knowledge base as ZIP file."""
    import json

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add all knowledge JSON files
        for f in KNOWLEDGE_DIR.glob("*.json"):
            if f.name == "activity_log.json":
                continue
            zf.write(f, f"knowledge/{f.name}")

        # Add learned files
        learned_dir = KNOWLEDGE_DIR / "learned"
        if learned_dir.exists():
            for f in learned_dir.iterdir():
                if not f.name.startswith("_"):
                    zf.write(f, f"knowledge/learned/{f.name}")

    buf.seek(0)
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    date_str = datetime.now(ist).strftime("%Y%m%d_%H%M")
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=digital-ketu-backup-{date_str}.zip"},
    )


# --- Confidence & Peak Hours & Reply Length Stats ---


@app.get("/api/confidence/stats")
async def confidence_stats():
    """Get reply confidence scoring stats."""
    from core.peak_hours import get_peak_hours_stats
    from core.reply_length import get_length_stats
    return {
        "peak_hours": get_peak_hours_stats(),
        "reply_length": get_length_stats(),
    }


# --- Scheduler Status ---


@app.get("/api/scheduler/status")
async def scheduler_status():
    """Next sync countdown for all scheduled tasks."""
    return get_scheduler_status()


# --- Dashboard UI ---

_dashboard_html = (Path(__file__).parent / "static" / "dashboard.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    """Digital Ketu Live Dashboard UI."""
    return _dashboard_html


# --- Run ---

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, reload=True)
