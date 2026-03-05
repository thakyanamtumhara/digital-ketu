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
YOUTUBE_CHECK_INTERVAL = 24 * 60 * 60  # 24 hours — check for new videos once a day
YOUTUBE_BACKFILL_INTERVAL = 24 * 60 * 60  # 24 hours — process old videos once a day
YOUTUBE_BACKFILL_BATCH_SIZE = 3  # Process only 3 old videos per cycle
CATALOG_SYNC_INTERVAL = 6 * 60 * 60  # 6 hours
KNOWLEDGE_REFRESH_INTERVAL = 6 * 60 * 60  # 6 hours
DATA_CLEANUP_INTERVAL = 24 * 60 * 60  # 24 hours — daily cleanup
DATA_RETENTION_DAYS = 90  # Keep raw data for 90 days

# --- YouTube Safety Gates ---
# Delay between each video transcript fetch (seconds) — prevents rate limiting
YOUTUBE_PER_VIDEO_DELAY = 30  # 30 seconds between each video
# Max YouTube API calls per day — hard safety limit
YOUTUBE_DAILY_QUOTA_LIMIT = 5  # Max 5 transcript fetches per day — very conservative
# Cooldown after a 429 error (seconds) — stop all YouTube for this long
YOUTUBE_RATE_LIMIT_COOLDOWN = 6 * 60 * 60  # 6 hours cooldown after rate limit hit

# Track daily YouTube usage — loaded from DB on startup to survive deploys
_youtube_daily_state = {
    "date": None,  # Current date string
    "api_calls": 0,  # Transcript fetches today
    "rate_limited_until": 0,  # Timestamp — don't call YouTube until this time
}
_youtube_state_loaded = False  # Whether we've loaded from DB yet

# Backfill state file — stores list of ALL channel video IDs for batch processing
_backfill_state_file = LEARNED_DIR / "_backfill_state.json"

# --- Next Sync Tracking ---
_scheduler_state: dict[str, dict] = {
    "youtube": {"last_run": None, "next_run": None, "interval": YOUTUBE_CHECK_INTERVAL, "status": "waiting"},
    "youtube_backfill": {"last_run": None, "next_run": None, "interval": YOUTUBE_BACKFILL_INTERVAL, "status": "waiting"},
    "youtube_quota": {"daily_limit": YOUTUBE_DAILY_QUOTA_LIMIT, "per_video_delay_sec": YOUTUBE_PER_VIDEO_DELAY, "status": "active"},
    "catalog": {"last_run": None, "next_run": None, "interval": CATALOG_SYNC_INTERVAL, "status": "waiting"},
    "knowledge_refresh": {"last_run": None, "next_run": None, "interval": KNOWLEDGE_REFRESH_INTERVAL, "status": "waiting"},
    "whatsapp": {"last_run": None, "next_run": None, "interval": 0, "status": "on-demand"},
    "data_cleanup": {"last_run": None, "next_run": None, "interval": DATA_CLEANUP_INTERVAL, "status": "waiting"},
}


def _load_youtube_daily_state():
    """Load YouTube daily state from DB — survives deploys."""
    global _youtube_state_loaded
    if _youtube_state_loaded:
        return

    from core.database import is_db_available, kv_get
    if is_db_available():
        saved = kv_get("_youtube_daily_state", {})
        if saved and isinstance(saved, dict):
            _youtube_daily_state["date"] = saved.get("date")
            _youtube_daily_state["api_calls"] = saved.get("api_calls", 0)
            _youtube_daily_state["rate_limited_until"] = saved.get("rate_limited_until", 0)
            logger.info(
                f"YouTube quota loaded from DB: {_youtube_daily_state['api_calls']} calls today "
                f"(date: {_youtube_daily_state['date']}, limit: {YOUTUBE_DAILY_QUOTA_LIMIT})"
            )
    _youtube_state_loaded = True


def _save_youtube_daily_state():
    """Save YouTube daily state to DB — survives deploys."""
    from core.database import is_db_available, kv_set
    if is_db_available():
        kv_set("_youtube_daily_state", {
            "date": _youtube_daily_state["date"],
            "api_calls": _youtube_daily_state["api_calls"],
            "rate_limited_until": _youtube_daily_state["rate_limited_until"],
        })


def _youtube_can_fetch() -> tuple[bool, str]:
    """Safety gate: check if we're allowed to fetch from YouTube right now.
    Returns (allowed, reason). Loads state from DB on first call.
    """
    import datetime

    # Load from DB on first call (survives deploys!)
    _load_youtube_daily_state()

    now = time.time()
    today = datetime.date.today().isoformat()

    # Gate 1: Rate limit cooldown — if YouTube returned 429, wait
    if _youtube_daily_state["rate_limited_until"] > now:
        remaining = int((_youtube_daily_state["rate_limited_until"] - now) / 60)
        return False, f"Rate limit cooldown active — {remaining} minutes remaining"

    # Gate 2: Daily quota limit — max transcript fetches per day
    if _youtube_daily_state["date"] != today:
        # New day — reset counter
        _youtube_daily_state["date"] = today
        _youtube_daily_state["api_calls"] = 0
        _save_youtube_daily_state()

    if _youtube_daily_state["api_calls"] >= YOUTUBE_DAILY_QUOTA_LIMIT:
        return False, f"Daily quota reached ({YOUTUBE_DAILY_QUOTA_LIMIT} fetches/day). Resets tomorrow."

    return True, "ok"


def _youtube_record_fetch():
    """Record a successful YouTube transcript fetch. Saves to DB."""
    import datetime
    today = datetime.date.today().isoformat()
    if _youtube_daily_state["date"] != today:
        _youtube_daily_state["date"] = today
        _youtube_daily_state["api_calls"] = 0
    _youtube_daily_state["api_calls"] += 1
    _save_youtube_daily_state()
    logger.info(f"YouTube quota: {_youtube_daily_state['api_calls']}/{YOUTUBE_DAILY_QUOTA_LIMIT} fetches today")


def _youtube_record_rate_limit():
    """Record a 429 rate limit — activate cooldown. Saves to DB."""
    _youtube_daily_state["rate_limited_until"] = time.time() + YOUTUBE_RATE_LIMIT_COOLDOWN
    _save_youtube_daily_state()
    logger.warning(
        f"YouTube rate limit hit! Cooldown activated for "
        f"{YOUTUBE_RATE_LIMIT_COOLDOWN // 3600} hours. No YouTube calls until cooldown expires."
    )


def _load_scheduler_timestamps():
    """Load last_run timestamps from DB — survives deploys."""
    try:
        from core.database import is_db_available, kv_get
        if not is_db_available():
            return
        saved = kv_get("_scheduler_timestamps", {})
        if saved and isinstance(saved, dict):
            now = time.time()
            for name, ts in saved.items():
                if name in _scheduler_state and isinstance(ts, (int, float)):
                    _scheduler_state[name]["last_run"] = ts
                    interval = _scheduler_state[name].get("interval", 0)
                    if interval > 0:
                        _scheduler_state[name]["next_run"] = ts + interval
                    ago = int(now - ts)
                    logger.info(f"[Scheduler] {name} last ran {ago}s ago")
    except Exception as e:
        logger.warning(f"[Scheduler] Failed to load timestamps from DB: {e}")


def _mark_run(task_name: str):
    """Mark a task as just completed, compute next_run. Persists to DB."""
    now = time.time()
    state = _scheduler_state[task_name]
    state["last_run"] = now
    state["status"] = "completed"
    if state["interval"] > 0:
        state["next_run"] = now + state["interval"]

    # Persist all timestamps to DB so they survive deploys
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            timestamps = {
                name: s["last_run"]
                for name, s in _scheduler_state.items()
                if s.get("last_run") is not None
            }
            kv_set("_scheduler_timestamps", timestamps)
    except Exception:
        pass


def get_scheduler_status() -> dict:
    """Return current scheduler state with countdowns."""
    import datetime

    # Load YouTube state from DB if not loaded yet
    _load_youtube_daily_state()

    now = time.time()
    result = {}
    for name, state in _scheduler_state.items():
        if name == "youtube_quota":
            # Special handling for quota display
            today = datetime.date.today().isoformat()
            calls_today = _youtube_daily_state["api_calls"] if _youtube_daily_state["date"] == today else 0
            cooldown_remaining = 0
            if _youtube_daily_state["rate_limited_until"] > now:
                cooldown_remaining = int((_youtube_daily_state["rate_limited_until"] - now) / 60)

            result[name] = {
                "daily_limit": YOUTUBE_DAILY_QUOTA_LIMIT,
                "calls_today": calls_today,
                "remaining_today": max(0, YOUTUBE_DAILY_QUOTA_LIMIT - calls_today),
                "per_video_delay_sec": YOUTUBE_PER_VIDEO_DELAY,
                "rate_limit_cooldown_minutes": cooldown_remaining,
                "status": "rate_limited" if cooldown_remaining > 0 else ("quota_reached" if calls_today >= YOUTUBE_DAILY_QUOTA_LIMIT else "active"),
            }
            continue

        entry = {
            "interval_hours": round(state["interval"] / 3600, 1) if state.get("interval") else None,
            "status": state["status"],
            "last_run_ago": None,
            "next_run_in": None,
        }
        if state.get("last_run"):
            entry["last_run_ago"] = int(now - state["last_run"])
        if state.get("next_run") and state["next_run"] > now:
            entry["next_run_in"] = int(state["next_run"] - now)
        elif state.get("next_run") and state["next_run"] <= now:
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

            # Delay between pages to avoid hitting YouTube API rate limits
            await asyncio.sleep(5)

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

    # Safety gate check before starting batch
    allowed, reason = _youtube_can_fetch()
    if not allowed:
        logger.info(f"Backfill: Skipped — {reason}")
        return {"status": "rate_limited", "reason": reason, "remaining": len(pending)}

    # Process next batch
    batch = pending[:batch_size]
    learned = 0

    for video in batch:
        # Safety gate check before EACH video
        allowed, reason = _youtube_can_fetch()
        if not allowed:
            logger.info(f"Backfill: Stopping mid-batch — {reason}")
            break

        video_id = video["video_id"]
        title = video["title"]

        logger.info(f"Backfill: Processing [{len(processed) + 1}/{state['total']}] {title}")

        try:
            # Run synchronous process_video in thread to avoid blocking event loop
            result = await asyncio.to_thread(
                process_video,
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_title=title,
            )

            # Check for rate limit in result
            if result.get("status") == "rate_limited":
                _youtube_record_rate_limit()
                logger.warning(f"Backfill: Rate limited on {title} — stopping batch")
                break

            _youtube_record_fetch()

            if result.get("status") == "ok" and result.get("knowledge"):
                knowledge = result["knowledge"]
                mapped_updates = _map_youtube_knowledge(knowledge)
                applied = await asyncio.to_thread(apply_knowledge_updates, mapped_updates)

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
            error_str = str(e).lower()
            if "429" in error_str or "too many" in error_str or "rate" in error_str:
                _youtube_record_rate_limit()
                logger.warning(f"Backfill: Rate limited on {title} — stopping batch")
                break
            logger.error(f"Backfill: Error processing {title}: {e}")
            # Don't mark as processed — will retry on next backfill run

        # Delay between videos to be gentle on YouTube
        logger.info(f"Backfill: Waiting {YOUTUBE_PER_VIDEO_DELAY}s before next video...")
        await asyncio.sleep(YOUTUBE_PER_VIDEO_DELAY)

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

    # Safety gate check
    allowed, reason = _youtube_can_fetch()
    if not allowed:
        logger.info(f"YouTube check: Skipped — {reason}")
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
            "maxResults": 3,  # Reduced from 5 — only check 3 most recent
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(url, params=params)

            # Handle rate limit response
            if response.status_code == 429:
                _youtube_record_rate_limit()
                return

            response.raise_for_status()
            data = response.json()

        videos = data.get("items", [])
        new_learned = 0

        for video in videos:
            video_id = video["id"]["videoId"]
            title = video["snippet"]["title"]

            if video_id in processed:
                continue

            # Safety gate check before each video
            allowed, reason = _youtube_can_fetch()
            if not allowed:
                logger.info(f"YouTube check: Stopping — {reason}")
                break

            logger.info(f"New video found: {title} ({video_id})")

            # Process video in thread — avoid blocking event loop
            result = await asyncio.to_thread(
                process_video,
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_title=title,
            )

            # Check for rate limit in result
            if result.get("status") == "rate_limited":
                _youtube_record_rate_limit()
                break

            _youtube_record_fetch()

            if result.get("status") == "ok" and result.get("knowledge"):
                knowledge = result["knowledge"]

                # Map YouTube extraction format to apply_knowledge_updates format
                mapped_updates = _map_youtube_knowledge(knowledge)
                applied = await asyncio.to_thread(apply_knowledge_updates, mapped_updates)

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

            # Delay between videos to be gentle on YouTube
            await asyncio.sleep(YOUTUBE_PER_VIDEO_DELAY)

        if new_learned > 0:
            logger.info(f"Learned from {new_learned} new video(s)")
        else:
            logger.info("No new videos to learn from")

    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "too many" in error_str or "rate" in error_str:
            _youtube_record_rate_limit()
            return
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
    """Background loop that syncs product catalog from GitHub repo.

    Respects DB-persisted last_run — won't re-run after deploy if recent.
    """
    # Wait 30 seconds after startup
    await asyncio.sleep(30)

    # Check if we ran recently (survives deploys)
    last_run = _scheduler_state["catalog"].get("last_run")
    if last_run:
        elapsed = time.time() - last_run
        remaining = CATALOG_SYNC_INTERVAL - elapsed
        if remaining > 0:
            logger.info(f"Catalog sync: Last ran {int(elapsed)}s ago, sleeping {int(remaining)}s")
            await asyncio.sleep(remaining)

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
                    validation = await asyncio.to_thread(validate_faqs_against_catalog)
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
    """Background loop that checks YouTube channel periodically.

    Respects DB-persisted last_run — won't re-run after deploy if recent.
    """
    # Wait 60 seconds after startup before first check
    await asyncio.sleep(60)

    # Check if we ran recently (survives deploys)
    last_run = _scheduler_state["youtube"].get("last_run")
    if last_run:
        elapsed = time.time() - last_run
        remaining = YOUTUBE_CHECK_INTERVAL - elapsed
        if remaining > 0:
            logger.info(f"YouTube check: Last ran {int(elapsed)}s ago, sleeping {int(remaining)}s")
            await asyncio.sleep(remaining)

    while True:
        _scheduler_state["youtube"]["status"] = "running"
        try:
            await check_youtube_channel()
        except Exception as e:
            logger.error(f"YouTube check loop error: {e}")

        _mark_run("youtube")
        await asyncio.sleep(YOUTUBE_CHECK_INTERVAL)


async def youtube_backfill_loop():
    """Background loop that processes old YouTube videos in batches.

    Persists last_run to DB so the 24h cooldown survives deploys.
    On startup, skips if the last run was less than YOUTUBE_BACKFILL_INTERVAL ago.
    """
    # Wait 2 minutes after startup (let other things initialize first)
    await asyncio.sleep(120)

    # Check if we ran recently (survives deploys) — skip if within interval
    last_run = _scheduler_state["youtube_backfill"].get("last_run")
    if last_run:
        elapsed = time.time() - last_run
        remaining = YOUTUBE_BACKFILL_INTERVAL - elapsed
        if remaining > 0:
            logger.info(
                f"YouTube backfill: Last ran {int(elapsed)}s ago, "
                f"sleeping {int(remaining)}s until next run"
            )
            await asyncio.sleep(remaining)

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

                # FAQ auto-deactivation — mark unused FAQs inactive
                from core.knowledge import auto_deactivate_stale_faqs
                deactivated = auto_deactivate_stale_faqs(days_threshold=30)
                if deactivated:
                    logger.info(f"FAQ cleanup: deactivated {len(deactivated)} unused FAQs")
            else:
                logger.info("Data cleanup skipped — no DB available")

        except Exception as e:
            logger.error(f"Data cleanup error: {e}")

        _mark_run("data_cleanup")
        await asyncio.sleep(DATA_CLEANUP_INTERVAL)


def start_scheduler():
    """Start all background tasks. Call this from FastAPI lifespan."""
    # Load last_run timestamps from DB so intervals survive deploys
    _load_scheduler_timestamps()

    loop = asyncio.get_event_loop()
    loop.create_task(catalog_sync_loop())
    loop.create_task(youtube_check_loop())
    loop.create_task(youtube_backfill_loop())
    loop.create_task(knowledge_refresh_loop())
    loop.create_task(data_cleanup_loop())
    logger.info(
        "Background scheduler started — "
        "Catalog sync every 6h, YouTube check every 24h, "
        f"YouTube backfill every 24h ({YOUTUBE_BACKFILL_BATCH_SIZE} videos/batch), "
        f"YouTube safety: {YOUTUBE_DAILY_QUOTA_LIMIT} fetches/day max, "
        f"{YOUTUBE_PER_VIDEO_DELAY}s delay between videos, "
        f"knowledge refresh every 6h, data cleanup daily ({DATA_RETENTION_DAYS}-day retention)"
    )
