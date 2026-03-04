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
from core.engine import generate_reply, get_customer_insights, get_faq_hit_rates
from core.knowledge import load_knowledge, invalidate_cache
from core.activity_log import log_activity, get_activity_log, get_today_summary, get_storage_stats
from integrations.whatsapp.webhook import router as whatsapp_router
from integrations.indiamart.handler import router as indiamart_router
from integrations.youtube.handler import router as youtube_router
from learner.chat_learner import (
    parse_whatsapp_export,
    extract_knowledge_from_messages,
    extract_knowledge_from_wwbun_messages,
    apply_knowledge_updates,
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
app.include_router(indiamart_router)
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


@app.post("/api/reply", response_model=ReplyResponse)
async def api_reply(req: ReplyRequest):
    """Generate AI reply — used by wwbun to get Digital Ketu's response.

    wwbun sends customer message → Digital Ketu returns reply.
    """
    reply = generate_reply(
        message=req.message,
        customer_phone=req.customer_phone,
        customer_name=req.customer_name,
        conversation_history=req.conversation_history,
    )
    log_activity(
        source="api-reply",
        action="replied",
        details={
            "customer_phone": req.customer_phone[-4:] if req.customer_phone else "unknown",
            "customer_name": req.customer_name or "unknown",
            "message_preview": req.message[:80],
            "reply_preview": reply[:80],
        },
        items_count=1,
    )
    return ReplyResponse(reply=reply)


# --- Auto-Reply Toggle ---


class ToggleRequest(BaseModel):
    enabled: bool


@app.post("/api/toggle")
async def toggle_auto_reply(req: ToggleRequest):
    """Enable/disable auto-reply."""
    settings.auto_reply_enabled = req.enabled
    status = "enabled" if req.enabled else "disabled"
    logger.info(f"Auto-reply {status}")
    return {"status": status, "auto_reply": settings.auto_reply_enabled}


@app.get("/api/toggle")
async def get_toggle_status():
    return {"auto_reply": settings.auto_reply_enabled}


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

    knowledge = extract_knowledge_from_messages(messages, req.ketu_name)
    result = apply_knowledge_updates(knowledge)

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


@app.post("/api/learn/wwbun-sync")
async def learn_from_wwbun(req: LearnWwbunRequest):
    """Learn from wwbun database messages.

    wwbun sends recent messages → Digital Ketu extracts knowledge
    from ONLY Ketu's manual messages (not AI-generated ones).

    Each message should have:
    - sender_id: who sent it
    - content: message text
    - is_ai_generated: bool (true if AI sent it, false if manual)
    """
    knowledge = extract_knowledge_from_wwbun_messages(
        messages=req.messages,
        owner_user_id=req.owner_user_id,
    )
    result = apply_knowledge_updates(knowledge)

    invalidate_cache()

    _mark_run("whatsapp")

    filter_stats = knowledge.get("filter_stats", {})
    quality_messages = knowledge.get("quality_messages", [])

    log_activity(
        source="wwbun-sync",
        action="learned",
        details={
            "total_messages": len(req.messages),
            "manual_messages": len([m for m in req.messages if not m.get("is_ai_generated")]),
            "quality_messages_count": filter_stats.get("kept", 0),
            "junk_skipped": filter_stats.get("junk", 0),
            "too_short_skipped": filter_stats.get("too_short", 0),
            "quality_messages_preview": quality_messages[:5],
            "updates_applied": result.get("applied", []),
        },
        items_count=result.get("count", 0),
    )

    return {
        "status": "ok",
        "filter_stats": filter_stats,
        "quality_messages": quality_messages,
        "knowledge_extracted": {k: v for k, v in knowledge.items() if k not in ("filter_stats", "quality_messages")},
        "updates_applied": result,
    }


class LearnYouTubeRequest(BaseModel):
    video_url: str
    video_title: str = ""


@app.post("/api/learn/youtube")
async def learn_from_youtube(req: LearnYouTubeRequest):
    """Learn from a YouTube video transcript.

    Extracts product info, pricing, business knowledge from video.
    """
    result = process_video(req.video_url, req.video_title)
    if result.get("knowledge"):
        applied = apply_knowledge_updates(result["knowledge"])
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
async def youtube_backfill_endpoint(batch_size: int = 5):
    """Manually trigger YouTube backfill — process old videos in batch.

    First call fetches ALL video IDs from channel.
    Each call processes `batch_size` unprocessed videos (default 5).
    Auto-runs every 6 hours in background, but you can trigger manually too.

    Query params:
    - batch_size: number of videos to process this run (default 5, max 20)
    """
    batch_size = min(batch_size, 20)  # Cap at 20 to avoid API overload
    result = await backfill_youtube_channel(batch_size=batch_size)
    return result


@app.get("/api/learn/youtube-backfill/status")
async def youtube_backfill_status():
    """Check YouTube backfill progress — how many videos done, how many left."""
    return get_backfill_status()


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
    result = learn_from_correction(
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


# --- Realtime Learner Stats ---


@app.get("/api/learn/realtime/stats")
async def realtime_learner_stats():
    """Get realtime learner statistics — buffer size, learning progress."""
    return get_realtime_stats()


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
