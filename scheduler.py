"""Background scheduler for Digital Ketu auto-learning.

Runs periodic tasks:
1. YouTube channel check - every 12 hours (detect new videos, extract knowledge)
2. Catalog repo sync - every 6 hours (fetch latest products/prices from GitHub)
3. Knowledge refresh - every 6 hours (reload from updated JSON files)

WhatsApp learning is triggered by wwbun calling /api/learn/wwbun-sync
whenever Ketu sends manual messages. No polling needed from our side.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx

from core.config import settings
from core.knowledge import invalidate_cache
from core.activity_log import log_activity
from learner.youtube_learner import process_video
from learner.chat_learner import apply_knowledge_updates
from learner.catalog_syncer import sync_catalog

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
LEARNED_DIR = KNOWLEDGE_DIR / "learned"

# Track processed videos to avoid re-processing
_processed_videos_file = LEARNED_DIR / "_processed_videos.json"

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"

# Schedule intervals (in seconds)
YOUTUBE_CHECK_INTERVAL = 12 * 60 * 60  # 12 hours
CATALOG_SYNC_INTERVAL = 6 * 60 * 60  # 6 hours
KNOWLEDGE_REFRESH_INTERVAL = 6 * 60 * 60  # 6 hours

# --- Next Sync Tracking ---
_scheduler_state: dict[str, dict] = {
    "youtube": {"last_run": None, "next_run": None, "interval": YOUTUBE_CHECK_INTERVAL, "status": "waiting"},
    "catalog": {"last_run": None, "next_run": None, "interval": CATALOG_SYNC_INTERVAL, "status": "waiting"},
    "knowledge_refresh": {"last_run": None, "next_run": None, "interval": KNOWLEDGE_REFRESH_INTERVAL, "status": "waiting"},
    "whatsapp": {"last_run": None, "next_run": None, "interval": 0, "status": "on-demand"},
}


def _mark_run(task_name: str):
    """Mark a task as just completed, compute next_run."""
    now = time.time()
    state = _scheduler_state[task_name]
    state["last_run"] = now
    state["status"] = "completed"
    if state["interval"] > 0:
        state["next_run"] = now + state["interval"]


def get_scheduler_status() -> dict:
    """Return current scheduler state with countdowns."""
    now = time.time()
    result = {}
    for name, state in _scheduler_state.items():
        entry = {
            "interval_hours": round(state["interval"] / 3600, 1) if state["interval"] else None,
            "status": state["status"],
            "last_run_ago": None,
            "next_run_in": None,
        }
        if state["last_run"]:
            entry["last_run_ago"] = int(now - state["last_run"])
        if state["next_run"] and state["next_run"] > now:
            entry["next_run_in"] = int(state["next_run"] - now)
        elif state["next_run"] and state["next_run"] <= now:
            entry["next_run_in"] = 0
            entry["status"] = "due"
        result[name] = entry
    return result


def _load_processed_videos() -> set[str]:
    """Load set of already-processed YouTube video IDs."""
    if _processed_videos_file.exists():
        try:
            with open(_processed_videos_file, "r") as f:
                data = json.load(f)
                return set(data.get("video_ids", []))
        except Exception:
            pass
    return set()


def _save_processed_video(video_id: str):
    """Add a video ID to the processed list."""
    LEARNED_DIR.mkdir(exist_ok=True)
    processed = _load_processed_videos()
    processed.add(video_id)
    with open(_processed_videos_file, "w") as f:
        json.dump({"video_ids": list(processed)}, f)


async def check_youtube_channel():
    """Check Sale91 YouTube channel for new videos and learn from them."""
    if not settings.youtube_api_key or not settings.youtube_channel_id:
        logger.info("YouTube API not configured, skipping channel check")
        return

    logger.info("Checking YouTube channel for new videos...")
    processed = _load_processed_videos()

    try:
        url = f"{YOUTUBE_API_URL}/search"
        params = {
            "key": settings.youtube_api_key,
            "channelId": settings.youtube_channel_id,
            "part": "snippet",
            "order": "date",
            "type": "video",
            "maxResults": 5,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        videos = data.get("items", [])
        new_learned = 0

        for video in videos:
            video_id = video["id"]["videoId"]
            title = video["snippet"]["title"]

            if video_id in processed:
                continue

            logger.info(f"New video found: {title} ({video_id})")

            # Process video — get transcript and extract knowledge
            result = process_video(
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_title=title,
            )

            if result.get("status") == "ok" and result.get("knowledge"):
                knowledge = result["knowledge"]

                # Map YouTube extraction format to apply_knowledge_updates format
                mapped_updates = _map_youtube_knowledge(knowledge)
                applied = apply_knowledge_updates(mapped_updates)

                if applied.get("count", 0) > 0:
                    logger.info(f"Applied {applied['count']} updates from video: {title}")
                    invalidate_cache()

                log_activity(
                    source="youtube",
                    action="learned",
                    details={
                        "video_id": video_id,
                        "video_title": title,
                        "updates_applied": applied.get("applied", []),
                    },
                    items_count=applied.get("count", 0),
                )
                new_learned += 1

            _save_processed_video(video_id)
            logger.info(f"Processed video: {title}")

        if new_learned > 0:
            logger.info(f"Learned from {new_learned} new video(s)")
        else:
            logger.info("No new videos to learn from")

    except Exception as e:
        logger.error(f"YouTube channel check failed: {e}")


def _map_youtube_knowledge(knowledge: dict) -> dict:
    """Map YouTube learner output format to apply_knowledge_updates format.

    YouTube learner extracts: product_info, pricing, business_knowledge, faqs_covered, key_points
    apply_knowledge_updates expects: new_faqs, style_patterns, price_updates, etc.
    """
    mapped = {}

    # Map faqs_covered → new_faqs
    faqs = knowledge.get("faqs_covered", [])
    if faqs:
        new_faqs = []
        for faq in faqs:
            if isinstance(faq, dict) and faq.get("question") and faq.get("answer"):
                new_faqs.append({
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "keywords": faq.get("keywords", []),
                })
            elif isinstance(faq, str):
                # Sometimes Claude returns FAQ as just a string
                new_faqs.append({
                    "question": faq,
                    "answer": faq,
                    "keywords": [],
                })
        if new_faqs:
            mapped["new_faqs"] = new_faqs

    # Map key_points as style/business patterns
    key_points = knowledge.get("key_points", [])
    if key_points:
        mapped["style_patterns"] = key_points

    return mapped


async def catalog_sync_loop():
    """Background loop that syncs product catalog from GitHub repo."""
    # Sync immediately on startup (after 30 second delay)
    await asyncio.sleep(30)

    while True:
        _scheduler_state["catalog"]["status"] = "running"
        try:
            result = await sync_catalog()
            if result.get("status") == "ok":
                invalidate_cache()
                log_activity(
                    source="catalog-sync",
                    action="synced",
                    details={"products_synced": result.get("products_synced", 0)},
                    items_count=result.get("products_synced", 0),
                )
                logger.info(f"Catalog synced: {result.get('products_synced')} products")
            else:
                logger.warning(f"Catalog sync issue: {result}")
        except Exception as e:
            logger.error(f"Catalog sync loop error: {e}")

        _mark_run("catalog")
        await asyncio.sleep(CATALOG_SYNC_INTERVAL)


async def youtube_check_loop():
    """Background loop that checks YouTube channel periodically."""
    # Wait 60 seconds after startup before first check
    await asyncio.sleep(60)

    while True:
        _scheduler_state["youtube"]["status"] = "running"
        try:
            await check_youtube_channel()
        except Exception as e:
            logger.error(f"YouTube check loop error: {e}")

        _mark_run("youtube")
        await asyncio.sleep(YOUTUBE_CHECK_INTERVAL)


async def knowledge_refresh_loop():
    """Background loop that refreshes knowledge cache periodically."""
    while True:
        await asyncio.sleep(KNOWLEDGE_REFRESH_INTERVAL)
        _scheduler_state["knowledge_refresh"]["status"] = "running"
        try:
            invalidate_cache()
            logger.info("Knowledge cache refreshed")
        except Exception as e:
            logger.error(f"Knowledge refresh error: {e}")
        _mark_run("knowledge_refresh")


def start_scheduler():
    """Start all background tasks. Call this from FastAPI lifespan."""
    loop = asyncio.get_event_loop()
    loop.create_task(catalog_sync_loop())
    loop.create_task(youtube_check_loop())
    loop.create_task(knowledge_refresh_loop())
    logger.info(
        "Background scheduler started — "
        "Catalog sync every 6h, YouTube check every 12h, knowledge refresh every 6h"
    )
