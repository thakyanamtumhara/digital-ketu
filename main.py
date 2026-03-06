import asyncio
import logging
from contextlib import asynccontextmanager

from pathlib import Path

import io
import zipfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from core.config import settings, init_knowledge_dir, KNOWLEDGE_DIR
from core.engine import generate_reply, get_customer_insights, get_faq_hit_rates, invalidate_ender_cache, get_last_escalation, ketu_manual_reply, activate_shutup
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
    except Exception:
        pass


def _save_wwbun_stats():
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            kv_set("wwbun_sync_stats", _wwbun_stats)
    except Exception:
        pass


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
    for msg in quality_previews[:5]:
        if isinstance(msg, dict) and msg.get("customer") and msg.get("ketu"):
            _wwbun_stats["recent_quality_messages"].append({
                "customer": msg["customer"][:100],
                "ketu": msg["ketu"][:120],
                "ai": msg.get("ai", False),
                "time": now.strftime("%I:%M %p"),
            })
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


def _learn_ketu_only_pairs(messages: list[dict], owner_user_id: str, learn_fn):
    """Extract customer→Ketu pairs from wwbun messages and learn ketu-only patterns.

    Groups messages by conversation (same chat). For each Ketu manual reply,
    finds the preceding customer message to form a Q→A pair. Feeds these pairs
    to the ketu-only learner which detects if Ketu gave info only he could know.
    """
    # Group by chat_id (same conversation)
    by_chat: dict[str, list] = {}
    for m in messages:
        chat_id = m.get("chat_id", m.get("remote_jid", "unknown"))
        by_chat.setdefault(chat_id, []).append(m)

    pairs_checked = 0
    for chat_id, chat_msgs in by_chat.items():
        # Sort by timestamp if available
        chat_msgs.sort(key=lambda x: x.get("timestamp", x.get("created_at", "")))

        last_customer_msg = ""
        customer_phone = chat_id.split("@")[0] if "@" in chat_id else chat_id

        for msg in chat_msgs:
            content = msg.get("content", "").strip()
            if not content:
                continue

            is_owner = msg.get("sender_id") == owner_user_id
            is_ai = msg.get("is_ai_generated", False)

            if not is_owner:
                # Customer message — remember it
                last_customer_msg = content
            elif is_owner and not is_ai and last_customer_msg:
                # Ketu's MANUAL reply to a customer message
                learn_fn(
                    customer_message=last_customer_msg,
                    ketu_reply=content,
                    customer_phone=customer_phone,
                )
                pairs_checked += 1
                last_customer_msg = ""  # Reset

    if pairs_checked > 0:
        logger.info(f"[KetuOnly] Checked {pairs_checked} customer→Ketu pairs from wwbun sync")


# --- Message Accumulator for Batch Learning ---
# wwbun sends small batches (2-3 msgs) frequently.
# We buffer them and only call Claude when we have 10+ quality pairs.
# Free features (enders, bought detection, repeat buyer) still run immediately.

_LEARNING_BUFFER_MIN_PAIRS = 10  # Need 10 quality Ketu manual messages before learning


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


def _count_quality_owner_messages(buffer: list[dict], owner_user_id: str) -> int:
    """Count quality PAIRS in the buffer (customer Q + Ketu manual reply = 1 pair).

    A pair is: a customer message followed by Ketu's manual (non-AI) reply.
    Both must pass quality checks (not junk, 3+ words for Ketu's reply).
    This ensures Claude gets meaningful context for learning.
    """
    from learner.chat_learner import is_junk_message
    pairs = 0
    last_customer_msg = None

    for m in buffer:
        raw_is_owner = m.get("is_owner", False)
        if isinstance(raw_is_owner, str):
            raw_is_owner = raw_is_owner.lower() in ("true", "1", "yes")
        is_owner = bool(raw_is_owner) or m.get("sender_id") == owner_user_id
        raw_ai = m.get("is_ai_generated", False)
        if isinstance(raw_ai, str):
            raw_ai = raw_ai.lower() in ("true", "1", "yes")
        is_ai = bool(raw_ai)
        text = m.get("content", "")

        if not is_owner:
            # Customer message — remember it as potential pair start
            if not is_junk_message(text):
                last_customer_msg = text
            continue

        # This is a Ketu message
        if is_ai:
            continue  # Skip AI-generated replies

        if is_junk_message(text):
            continue
        if len(text.split()) < 3:
            continue

        # Quality Ketu manual reply — check if we have a customer message before it
        if last_customer_msg:
            pairs += 1
            last_customer_msg = None  # Consume the pair

    return pairs


def _extract_pairs_from_buffer(buffer: list[dict], owner_user_id: str) -> list[dict]:
    """Extract customer→Ketu pairs from buffer for dashboard preview.
    Includes AI-generated replies so user can see all conversations.
    """
    pairs = []
    last_customer_msg = None
    for m in buffer:
        raw_is_owner = m.get("is_owner", False)
        if isinstance(raw_is_owner, str):
            raw_is_owner = raw_is_owner.lower() in ("true", "1", "yes")
        is_owner = bool(raw_is_owner) or m.get("sender_id") == owner_user_id
        if not is_owner:
            text = m.get("content", "")
            if text and len(text.strip()) > 0:
                last_customer_msg = text[:100]
            continue
        text = m.get("content", "")
        if not text or len(text.split()) < 2:
            continue
        raw_ai = m.get("is_ai_generated", False)
        if isinstance(raw_ai, str):
            raw_ai = raw_ai.lower() in ("true", "1", "yes")
        is_ai = bool(raw_ai)
        if last_customer_msg:
            pairs.append({"customer": last_customer_msg, "ketu": text[:120], "ai": is_ai})
            last_customer_msg = None
    return pairs


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
    if knowledge.get("price_updates"):
        extracted_summary.append(f"{len(knowledge['price_updates'])} prices")
    if knowledge.get("product_updates"):
        extracted_summary.append("product info")
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

    # --- PAID feature: accumulate messages for batch Claude learning ---

    # Debug: log what wwbun is sending so we can trace pairing issues
    if req.messages:
        sample = req.messages[:5]
        owner_flags = []
        for m in sample:
            is_own = m.get("is_owner", False)
            is_own_type = type(is_own).__name__
            sid_match = m.get("sender_id") == req.owner_user_id
            owner_flags.append(f"is_owner={is_own!r}({is_own_type}),sid_match={sid_match}")
        logger.info(
            f"[wwbun-sync DEBUG] owner_user_id={req.owner_user_id!r}, "
            f"total_msgs={len(req.messages)}, "
            f"sample_flags=[{', '.join(owner_flags)}]"
        )
        for i, m in enumerate(sample):
            logger.info(
                f"[wwbun-sync DEBUG] msg[{i}]: is_owner={m.get('is_owner')!r} "
                f"sender_id={m.get('sender_id', '')!r} "
                f"is_ai={m.get('is_ai_generated')!r} "
                f"content={m.get('content', '')[:50]!r}"
            )

    # Add new messages to buffer
    buffer = _get_learning_buffer()
    buffer.extend(req.messages)

    # Cap buffer at 500 messages to prevent unbounded growth
    if len(buffer) > 500:
        buffer = buffer[-500:]

    _save_learning_buffer(buffer)

    # Count quality manual messages in buffer
    quality_count = _count_quality_owner_messages(buffer, req.owner_user_id)

    # Decide: learn now or wait for more messages
    learn_result = {"status": "buffered", "count": 0}
    filter_stats = {"total": len(req.messages), "kept": 0, "junk": 0, "too_short": 0}
    quality_messages = []

    if quality_count >= _LEARNING_BUFFER_MIN_PAIRS:
        # Enough data! Flush buffer and learn
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
        # Not enough yet — just log the buffering
        log_activity(
            source="wwbun-sync",
            action="buffered",
            details={
                "messages_added": len(req.messages),
                "buffer_total": len(buffer),
                "quality_in_buffer": quality_count,
                "needed": _LEARNING_BUFFER_MIN_PAIRS,
                "remaining": _LEARNING_BUFFER_MIN_PAIRS - quality_count,
            },
            items_count=0,
        )

        # Count quality in THIS batch for accurate stats tracking
        # (even though we're not learning yet, dashboard should show quality messages arriving)
        from learner.chat_learner import filter_messages as _filter_msgs
        _, batch_stats = _filter_msgs(req.messages, owner_key="sender_id", owner_value=req.owner_user_id)
        batch_quality = batch_stats.get("kept", 0)

        # Collect quality PAIRS from this batch for dashboard preview
        # Include AI-generated replies too (user wants to see all conversations flowing)
        batch_quality_previews = []
        last_customer_msg = None
        _debug_owner_count = 0
        _debug_customer_count = 0
        for m in req.messages:
            raw_is_owner = m.get("is_owner", False)
            # Handle string "true"/"false" from wwbun (JS might send strings)
            if isinstance(raw_is_owner, str):
                raw_is_owner = raw_is_owner.lower() in ("true", "1", "yes")
            is_owner = bool(raw_is_owner) or m.get("sender_id") == req.owner_user_id
            if not is_owner:
                _debug_customer_count += 1
                # Customer message — remember for pairing
                text = m.get("content", "")
                if text and len(text.strip()) > 0:
                    last_customer_msg = text[:100]
                continue
            _debug_owner_count += 1
            # Ketu's message (manual or AI) — pair with customer
            text = m.get("content", "")
            if not text or len(text.split()) < 2:
                continue
            raw_ai = m.get("is_ai_generated", False)
            if isinstance(raw_ai, str):
                raw_ai = raw_ai.lower() in ("true", "1", "yes")
            is_ai = bool(raw_ai)
            if last_customer_msg:
                batch_quality_previews.append({
                    "customer": last_customer_msg,
                    "ketu": text[:120],
                    "ai": is_ai,
                })
                last_customer_msg = None
        logger.info(
            f"[wwbun-sync PAIRS] batch={len(req.messages)} msgs, "
            f"owner={_debug_owner_count}, customer={_debug_customer_count}, "
            f"pairs_found={len(batch_quality_previews)}"
        )

        logger.info(f"[wwbun-sync] Found {len(batch_quality_previews)} preview pairs in batch of {len(req.messages)} msgs")

        # Still track sync stats even when buffering
        _track_wwbun_sync(
            total_messages=len(req.messages),
            quality_count=batch_quality,
            junk_count=batch_stats.get("junk", 0),
            short_count=batch_stats.get("too_short", 0),
            knowledge_count=0,
            enders_learned=ender_result.get("new_enders", 0),
            quality_previews=batch_quality_previews,
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
            "total_buffered": len(buffer) if learn_result.get("status") == "buffered" else 0,
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


@app.post("/api/learn/flush-buffer")
async def flush_learning_buffer(req: FlushBufferRequest):
    """Manually flush the learning buffer — force Claude to learn from whatever is buffered.

    Use this if you want to force learning even with fewer than 10 quality pairs.
    """
    buffer = _get_learning_buffer()
    if not buffer:
        return {"status": "empty", "message": "No messages in buffer"}

    quality_count = _count_quality_owner_messages(buffer, req.owner_user_id)

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
    buffer = _get_learning_buffer()
    quality = _count_quality_owner_messages(buffer, owner_user_id) if owner_user_id else 0

    # Show sample messages for debugging
    sample_msgs = []
    for m in buffer[-10:]:
        raw_owner = m.get("is_owner", False)
        raw_ai = m.get("is_ai_generated", False)
        sample_msgs.append({
            "content": m.get("content", "")[:80],
            "is_owner": raw_owner,
            "is_owner_type": type(raw_owner).__name__,
            "sender_id": m.get("sender_id", "")[-6:],  # Last 6 chars for privacy
            "is_ai_generated": raw_ai,
            "is_ai_type": type(raw_ai).__name__,
        })

    return {
        "total_buffered": len(buffer),
        "quality_pairs": quality,
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
    """
    ketu_manual_reply(req.customer_phone)
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
    message_id: str = ""  # Optional — if empty, edits the last sent message


@app.post("/api/whatsapp/edit")
async def edit_whatsapp_message(req: EditMessageRequest):
    """Edit a previously sent WhatsApp message.

    Works within 15 minutes of sending (WhatsApp API limit).
    Customer sees the updated message with "(edited)" label.

    If message_id is empty, edits the LAST message sent to that phone.
    """
    from integrations.whatsapp.sender import edit_message, edit_last_message, get_last_sent_message

    if req.message_id:
        result = await edit_message(to=req.phone, message_id=req.message_id, new_text=req.new_text)
    else:
        result = await edit_last_message(to=req.phone, new_text=req.new_text)

    if result:
        log_activity(
            source="whatsapp",
            action="message-edited",
            details={
                "customer_phone": req.phone[-4:] if req.phone else "unknown",
                "new_text": req.new_text[:100],
            },
        )
        return {"status": "edited", "result": result}

    # Check why it failed
    last = get_last_sent_message(req.phone)
    if not last:
        return {"status": "error", "detail": "No sent message found for this phone number"}
    if not last["editable"]:
        return {"status": "error", "detail": f"Edit window expired ({last['seconds_ago']}s ago, limit is 15 min)"}
    return {"status": "error", "detail": "Edit failed — check error logs"}


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
