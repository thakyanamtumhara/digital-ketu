import logging

import httpx
from fastapi import APIRouter

from core.config import settings
from core.engine import generate_reply

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/youtube", tags=["youtube"])

# Track replied comment IDs to avoid duplicates
_replied_comments: set[str] = set()
_replied_loaded = False


def _load_replied_comments():
    """Load replied comment IDs from DB on first access."""
    global _replied_comments, _replied_loaded
    if _replied_loaded:
        return
    _replied_loaded = True
    try:
        from core.database import is_db_available, kv_get
        if is_db_available():
            data = kv_get("youtube_replied_comments")
            if data and isinstance(data, dict):
                _replied_comments.update(data.get("ids", []))
    except Exception:
        pass


def _save_replied_comments():
    """Persist replied comment IDs to DB."""
    try:
        from core.database import is_db_available, kv_set
        if is_db_available():
            # Keep last 500 to avoid unbounded growth
            ids = list(_replied_comments)[-500:]
            kv_set("youtube_replied_comments", {"ids": ids})
    except Exception:
        pass

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"


async def get_recent_videos(max_results: int = 5) -> list[dict]:
    """Fetch recent videos from the channel."""
    url = f"{YOUTUBE_API_URL}/search"
    params = {
        "key": settings.youtube_api_key,
        "channelId": settings.youtube_channel_id,
        "part": "snippet",
        "order": "date",
        "type": "video",
        "maxResults": max_results,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])


async def get_video_comments(video_id: str, max_results: int = 20) -> list[dict]:
    """Fetch comments for a video."""
    url = f"{YOUTUBE_API_URL}/commentThreads"
    params = {
        "key": settings.youtube_api_key,
        "videoId": video_id,
        "part": "snippet",
        "order": "time",
        "maxResults": max_results,
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return data.get("items", [])


async def reply_to_comment(comment_id: str, text: str) -> dict | None:
    """Post a reply to a YouTube comment."""
    url = f"{YOUTUBE_API_URL}/comments"
    params = {"part": "snippet", "key": settings.youtube_api_key}
    payload = {
        "snippet": {
            "parentId": comment_id,
            "textOriginal": text,
        }
    }
    # Note: This requires OAuth2 authentication, not just API key
    # For now, log the reply. OAuth setup needed for actual posting.
    logger.info(f"YouTube reply for {comment_id}: {text[:50]}...")
    return {"status": "logged", "comment_id": comment_id}


@router.post("/check-comments")
async def check_and_reply_comments():
    """Check recent video comments and reply to product-related ones."""
    if not settings.youtube_api_key or not settings.youtube_channel_id:
        return {"status": "skipped", "reason": "YouTube API not configured"}

    try:
        videos = await get_recent_videos(max_results=3)
    except Exception as e:
        logger.error(f"Failed to fetch videos: {e}")
        return {"status": "error", "detail": str(e)}

    replied = []
    for video in videos:
        video_id = video["id"]["videoId"]
        video_title = video["snippet"]["title"]

        try:
            comments = await get_video_comments(video_id)
        except Exception as e:
            logger.error(f"Failed to fetch comments for {video_id}: {e}")
            continue

        for comment in comments:
            snippet = comment["snippet"]["topLevelComment"]["snippet"]
            comment_id = comment["snippet"]["topLevelComment"]["id"]
            comment_text = snippet["textDisplay"]
            author = snippet["authorDisplayName"]

            # Skip if already replied
            _load_replied_comments()
            if comment_id in _replied_comments:
                continue

            # Skip very short or non-product comments
            if len(comment_text) < 10:
                continue

            # Generate reply
            message = (
                f"YouTube video '{video_title}' pe comment aaya hai:\n"
                f"User: {author}\n"
                f"Comment: {comment_text}\n\n"
                "Is comment ka reply karo. Short aur helpful rakhna. "
                "Agar product related hai toh pricing bata do. "
                "sale91.com ka mention karo."
            )

            reply = generate_reply(message=message)
            await reply_to_comment(comment_id, reply)

            _replied_comments.add(comment_id)
            _save_replied_comments()
            replied.append({
                "video": video_title,
                "comment": comment_text[:50],
                "reply": reply[:50],
            })

    return {"status": "ok", "replied_count": len(replied), "replies": replied}
