"""Background scheduler for Digital Ketu auto-learning.

Runs periodic tasks:
1. YouTube channel check - every 12 hours (detect new videos, extract knowledge)
2. YouTube backfill - every 6 hours (process 5 old videos per batch until all done)
3. Catalog repo sync - every 6 hours (fetch latest products/prices from GitHub)
4. Knowledge refresh - every 6 hours (reload from updated JSON files)

WhatsApp learning is triggered by wwbun calling /api/learn/wwbun-sync
whenever Ketu sends manual messages. No polling needed from our side.
"""

import asyncio
import json
import logging
import time

import httpx

from core.config import settings, KNOWLEDGE_DIR, LEARNED_DIR
from core.knowledge import invalidate_cache
from core.activity_log import log_activity
from learner.youtube_learner import process_video
from learner.chat_learner import apply_knowledge_updates
from learner.catalog_syncer import sync_catalog
from learner.faq_validator import validate_faqs_against_catalog

logger = logging.getLogger(__name__)

# Track processed videos to avoid re-processing
_processed_videos_file = LEARNED_DIR / "_processed_videos.json"

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"

# Schedule intervals (in seconds)
YOUTUBE_CHECK_INTERVAL = 12 * 60 * 60  # 12 hours
YOUTUBE_BACKFILL_INTERVAL = 6 * 60 * 60  # 6 hours — process old videos in batches
YOUTUBE_BACKFILL_BATCH_SIZE = 5  # Process 5 old videos per cycle
CATALOG_SYNC_INTERVAL = 6 * 60 * 60  # 6 hours
KNOWLEDGE_REFRESH_INTERVAL = 6 * 60 * 60  # 6 hours
DATA_CLEANUP_INTERVAL = 24 * 60 * 60  # 24 hours — daily cleanup
DATA_RETENTION_DAYS = 90  # Keep raw data for 90 days

# Backfill state file — stores list of ALL channel video IDs for batch processing
_backfill_state_file = LEARNED_DIR / "_backfill_state.json"

# --- Next Sync Tracking ---
_scheduler_state: dict[str, dict] = {
    "youtube": {"last_run": None, "next_run": None, "interval": YOUTUBE_CHECK_INTERVAL, "status": "waiting"},
    "youtube_backfill": {"last_run": None, "next_run": None, "interval": YOUTUBE_BACKFILL_INTERVAL, "status": "waiting"},
    "catalog": {"last_run": None, "next_run": None, "interval": CATALOG_SYNC_INTERVAL, "status": "waiting"},
    "knowledge_refresh": {"last_run": None, "next_run": None, "interval": KNOWLEDGE_REFRESH_INTERVAL, "status": "waiting"},
    "whatsapp": {"last_run": None, "next_run": None, "interval": 0, "status": "on-demand"},
    "data_cleanup": {"last_run": None, "next_run": None, "interval": DATA_CLEANUP_INTERVAL, "status": "waiting"},
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
    """Load set of already-processed YouTube video IDs. DB first, JSON fallback."""
    from core.database import is_db_available, kv_get
    if is_db_available():
        data = kv_get("_processed_videos", {})
        if data and isinstance(data, dict) and data.get("video_ids"):
            return set(data["video_ids"])

    if _processed_videos_file.exists():
        try:
            with open(_processed_videos_file, "r") as f:
                data = json.load(f)
                return set(data.get("video_ids", []))
        except Exception:
            pass
    return set()


def _save_processed_video(video_id: str):
    """Add a video ID to the processed list. Save to DB + JSON + GitHub."""
    LEARNED_DIR.mkdir(exist_ok=True)
    processed = _load_processed_videos()
    processed.add(video_id)

    data = {"video_ids": list(processed)}

    # DB first (primary — survives deploys)
    from core.database import is_db_available, kv_set
    if is_db_available():
        kv_set("_processed_videos", data)

    # Then JSON file
    with open(_processed_videos_file, "w") as f:
        json.dump(data, f)

    # Persist to GitHub (backup)
    from core.git_persist import persist_single_file
    persist_single_file(
        "knowledge/learned/_processed_videos.json",
        _processed_videos_file,
        source="processed-videos-update",
    )


def _load_backfill_state() -> dict:
    """Load backfill state. DB first, JSON fallback."""
    from core.database import is_db_available, kv_get
    if is_db_available():
        data = kv_get("_backfill_state", {})
        if data and isinstance(data, dict) and data.get("fetched"):
            return data

    if _backfill_state_file.exists():
        try:
            with open(_backfill_state_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"all_video_ids": [], "total": 0, "fetched": False}


def _save_backfill_state(state: dict):
    """Save backfill state to DB + JSON + GitHub."""
    LEARNED_DIR.mkdir(exist_ok=True)

    # DB first (primary — survives deploys)
    from core.database import is_db_available, kv_set
    if is_db_available():
        kv_set("_backfill_state", state)

    with open(_backfill_state_file, "w") as f:
        json.dump(state, f)

    # Persist to GitHub (backup)
    from core.git_persist import persist_single_file
    persist_single_file(
        "knowledge/learned/_backfill_state.json",
        _backfill_state_file,
        source="backfill-state-update",
    )


async def _fetch_all_channel_videos() -> list[dict]:
    """Fetch ALL videos from channel using uploads playlist (with pagination).

    Uses Channels API to get uploads playlist, then PlaylistItems API
    with pageToken pagination to get every single video.
    Returns list of {video_id, title} dicts.
    """
    if not settings.youtube_api_key or not settings.youtube_channel_id:
        return []

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: Get uploads playlist ID
        ch_resp = await client.get(
            f"{YOUTUBE_API_URL}/channels",
            params={
                "key": settings.youtube_api_key,
                "id": settings.youtube_channel_id,
                "part": "contentDetails",
            },
        )
        ch_resp.raise_for_status()
        ch_data = ch_resp.json()

        items = ch_data.get("items", [])
        if not items:
            logger.error("Channel not found or no contentDetails")
            return []

        uploads_playlist = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        logger.info(f"Uploads playlist: {uploads_playlist}")

        # Step 2: Paginate through ALL videos in uploads playlist
        all_videos = []
        page_token = None

        while True:
            params = {
                "key": settings.youtube_api_key,
                "playlistId": uploads_playlist,
                "part": "snippet",
                "maxResults": 50,  # Max allowed by API
            }
            if page_token:
                params["pageToken"] = page_token

            resp = await client.get(f"{YOUTUBE_API_URL}/playlistItems", params=params)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("items", []):
                vid_id = item["snippet"]["resourceId"]["videoId"]
                title = item["snippet"]["title"]
                all_videos.append({"video_id": vid_id, "title": title})

            page_token = data.get("nextPageToken")
            if not page_token:
                break

        logger.info(f"Found {len(all_videos)} total videos on channel")
        return all_videos


async def backfill_youtube_channel(batch_size: int | None = None) -> dict:
    """Process old YouTube videos in batches.

    First run: fetches ALL video IDs from channel and saves them.
    Each run: picks next batch of unprocessed videos and learns from them.
    Returns status with progress info.
    """
    if not settings.youtube_api_key or not settings.youtube_channel_id:
        return {"status": "not_configured"}

    if batch_size is None:
        batch_size = YOUTUBE_BACKFILL_BATCH_SIZE

    state = _load_backfill_state()
    processed = _load_processed_videos()

    # First time: fetch all video IDs from channel
    if not state.get("fetched"):
        logger.info("Backfill: Fetching all channel videos for the first time...")
        try:
            all_videos = await _fetch_all_channel_videos()
            state = {
                "all_video_ids": [v for v in all_videos],
                "total": len(all_videos),
                "fetched": True,
            }
            _save_backfill_state(state)
            logger.info(f"Backfill: Saved {len(all_videos)} video IDs for processing")
        except Exception as e:
            logger.error(f"Backfill: Failed to fetch channel videos: {e}")
            return {"status": "error", "detail": str(e)}

    # Find unprocessed videos
    all_videos = state.get("all_video_ids", [])
    pending = [v for v in all_videos if v["video_id"] not in processed]

    if not pending:
        logger.info("Backfill complete! All videos processed.")
        return {
            "status": "complete",
            "total": state["total"],
            "processed": len(processed),
            "remaining": 0,
        }

    # Process next batch
    batch = pending[:batch_size]
    learned = 0

    for video in batch:
        video_id = video["video_id"]
        title = video["title"]

        logger.info(f"Backfill: Processing [{len(processed) + 1}/{state['total']}] {title}")

        try:
            result = process_video(
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_title=title,
            )

            if result.get("status") == "ok" and result.get("knowledge"):
                knowledge = result["knowledge"]
                mapped_updates = _map_youtube_knowledge(knowledge)
                applied = apply_knowledge_updates(mapped_updates)

                if applied.get("count", 0) > 0:
                    logger.info(f"Backfill: Applied {applied['count']} updates from: {title}")
                    invalidate_cache()

                log_activity(
                    source="youtube",
                    action="learned",
                    details={
                        "video_id": video_id,
                        "video_title": title,
                        "backfill": True,
                        "updates_applied": applied.get("applied", []),
                    },
                    items_count=applied.get("count", 0),
                )
                learned += 1

            _save_processed_video(video_id)

        except Exception as e:
            logger.error(f"Backfill: Error processing {title}: {e}")
            # Don't mark as processed — will retry on next backfill run

    remaining = len(pending) - len(batch)
    logger.info(f"Backfill: Processed {len(batch)} videos, {remaining} remaining")

    return {
        "status": "in_progress" if remaining > 0 else "complete",
        "total": state["total"],
        "processed_this_batch": len(batch),
        "learned_this_batch": learned,
        "processed_total": state["total"] - remaining,
        "remaining": remaining,
    }


def get_backfill_status() -> dict:
    """Get current backfill progress without running anything."""
    state = _load_backfill_state()
    processed = _load_processed_videos()

    if not state.get("fetched"):
        return {
            "status": "not_started",
            "total": 0,
            "processed": len(processed),
            "remaining": 0,
            "progress_pct": 0,
        }

    total = state.get("total", 0)
    all_videos = state.get("all_video_ids", [])
    pending = [v for v in all_videos if v["video_id"] not in processed]
    done = total - len(pending)

    return {
        "status": "complete" if len(pending) == 0 else "in_progress",
        "total": total,
        "processed": done,
        "remaining": len(pending),
        "progress_pct": round((done / total) * 100, 1) if total > 0 else 0,
    }


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
        from core.error_tracker import track_error
        track_error("youtube-check", str(e))


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
                logger.info(f"Catalog synced: {result.get('products_synced')} products")

                # Validate FAQs against updated catalog
                try:
                    validation = validate_faqs_against_catalog()
                    if validation.get("deactivated") or validation.get("price_warnings"):
                        log_activity(
                            source="faq-validator",
                            action="validated",
                            details={
                                "deactivated": validation.get("deactivated", 0),
                                "price_warnings": validation.get("price_warnings", 0),
                                "flagged": validation.get("flagged_faqs", []),
                                "price_issues": validation.get("price_issues", []),
                            },
                            items_count=validation.get("deactivated", 0),
                        )
                        logger.info(
                            f"FAQ validation: {validation['deactivated']} deactivated, "
                            f"{validation['price_warnings']} price warnings"
                        )
                except Exception as e:
                    logger.error(f"FAQ validation error: {e}")
            else:
                logger.warning(f"Catalog sync issue: {result}")
        except Exception as e:
            logger.error(f"Catalog sync loop error: {e}")
            from core.error_tracker import track_error
            track_error("catalog-sync", str(e))

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


async def youtube_backfill_loop():
    """Background loop that processes old YouTube videos in batches."""
    # Wait 2 minutes after startup (let other things initialize first)
    await asyncio.sleep(120)

    while True:
        _scheduler_state["youtube_backfill"]["status"] = "running"
        try:
            result = await backfill_youtube_channel()
            if result.get("status") == "complete":
                logger.info("YouTube backfill complete — all videos processed!")
                _mark_run("youtube_backfill")
                _scheduler_state["youtube_backfill"]["status"] = "complete"
                return  # Stop the loop — all done!
            elif result.get("status") == "in_progress":
                remaining = result.get("remaining", 0)
                logger.info(f"YouTube backfill: {remaining} videos remaining")
        except Exception as e:
            logger.error(f"YouTube backfill loop error: {e}")

        _mark_run("youtube_backfill")
        await asyncio.sleep(YOUTUBE_BACKFILL_INTERVAL)


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


async def data_cleanup_loop():
    """Background loop that cleans up old data (90-day retention).

    Runs daily. Deletes:
    - Activity logs older than 90 days
    - Learned files older than 90 days (except catalog files)
    Knowledge (prompt, FAQs, patterns) is PERMANENT — never deleted.
    """
    # Wait 5 minutes after startup
    await asyncio.sleep(300)

    while True:
        _scheduler_state["data_cleanup"]["status"] = "running"
        try:
            from core.database import is_db_available, cleanup_old_activity_logs, cleanup_old_learned_files

            if is_db_available():
                activity_deleted = cleanup_old_activity_logs(days=DATA_RETENTION_DAYS)
                learned_deleted = cleanup_old_learned_files(
                    days=DATA_RETENTION_DAYS,
                    keep_patterns=["catalog_%"],  # Never delete catalog files
                )

                if activity_deleted or learned_deleted:
                    log_activity(
                        source="data-cleanup",
                        action="cleaned",
                        details={
                            "retention_days": DATA_RETENTION_DAYS,
                            "activity_logs_deleted": activity_deleted,
                            "learned_files_deleted": learned_deleted,
                        },
                        items_count=activity_deleted + learned_deleted,
                    )
                    logger.info(
                        f"Data cleanup: {activity_deleted} old activity logs, "
                        f"{learned_deleted} old learned files deleted (>{DATA_RETENTION_DAYS} days)"
                    )
                else:
                    logger.info(f"Data cleanup: nothing to clean (all within {DATA_RETENTION_DAYS} days)")
            else:
                logger.info("Data cleanup skipped — no DB available")

        except Exception as e:
            logger.error(f"Data cleanup error: {e}")

        _mark_run("data_cleanup")
        await asyncio.sleep(DATA_CLEANUP_INTERVAL)


def start_scheduler():
    """Start all background tasks. Call this from FastAPI lifespan."""
    loop = asyncio.get_event_loop()
    loop.create_task(catalog_sync_loop())
    loop.create_task(youtube_check_loop())
    loop.create_task(youtube_backfill_loop())
    loop.create_task(knowledge_refresh_loop())
    loop.create_task(data_cleanup_loop())
    logger.info(
        "Background scheduler started — "
        "Catalog sync every 6h, YouTube check every 12h, "
        "YouTube backfill every 6h (5 old videos/batch), knowledge refresh every 6h, "
        f"data cleanup daily ({DATA_RETENTION_DAYS}-day retention)"
    )
