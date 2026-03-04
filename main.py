import logging
from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.config import settings, init_knowledge_dir
from core.engine import generate_reply
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure knowledge directories exist
    init_knowledge_dir()

    # Startup: load knowledge base
    logger.info("Loading knowledge base...")
    load_knowledge()

    # Start background scheduler (YouTube auto-check, knowledge refresh)
    start_scheduler()

    logger.info("Digital Ketu is ready! Auto-learning scheduler active.")
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
        log_activity(
            source="catalog-sync",
            action="synced",
            details={
                "products_synced": result.get("products_synced", 0),
            },
            items_count=result.get("products_synced", 0),
        )
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
