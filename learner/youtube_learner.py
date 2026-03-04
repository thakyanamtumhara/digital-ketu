import json
import logging
import re

import httpx
from anthropic import Anthropic

from core.config import settings, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


def _get_oauth_access_token() -> str | None:
    """Exchange refresh token for a fresh access token (official OAuth 2.0).

    Returns access_token string or None if OAuth not configured.
    """
    if not settings.youtube_client_id or not settings.youtube_client_secret or not settings.youtube_refresh_token:
        return None

    try:
        resp = httpx.post(
            GOOGLE_TOKEN_URL,
            data={
                "client_id": settings.youtube_client_id,
                "client_secret": settings.youtube_client_secret,
                "refresh_token": settings.youtube_refresh_token,
                "grant_type": "refresh_token",
            },
            timeout=15,
        )
        resp.raise_for_status()
        token_data = resp.json()
        return token_data.get("access_token")
    except Exception as e:
        logger.error(f"OAuth token refresh failed: {e}")
        return None


def _get_transcript_official(video_id: str) -> str | None:
    """Get transcript using official YouTube Captions API (OAuth 2.0).

    Steps:
    1. List available caption tracks for the video
    2. Pick Hindi first, then English
    3. Download the caption track in SRT format
    4. Parse SRT → plain text

    This is the official, Google-approved method. No scraping.
    Costs: captions.list = 50 quota units, captions.download = 200 quota units.
    """
    access_token = _get_oauth_access_token()
    if not access_token:
        return None

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        # Step 1: List caption tracks
        resp = httpx.get(
            f"{YOUTUBE_API_URL}/captions",
            params={"videoId": video_id, "part": "snippet"},
            headers=headers,
            timeout=15,
        )

        if resp.status_code == 403:
            logger.warning(f"Captions API forbidden for {video_id} — may need to enable YouTube Data API v3")
            return None
        if resp.status_code == 429:
            logger.warning(f"YouTube API rate limited (official) for {video_id}")
            raise Exception("429 Too Many Requests (official API)")

        resp.raise_for_status()
        captions_data = resp.json()
        tracks = captions_data.get("items", [])

        if not tracks:
            logger.info(f"No caption tracks found for {video_id}")
            return None

        # Step 2: Pick best track — Hindi first, then English
        selected = None
        for lang_pref in ["hi", "hi-IN", "en", "en-US", "en-IN"]:
            for track in tracks:
                if track["snippet"]["language"] == lang_pref:
                    selected = track
                    break
            if selected:
                break

        # Fallback: pick any track
        if not selected:
            selected = tracks[0]

        caption_id = selected["id"]
        lang = selected["snippet"]["language"]
        logger.info(f"Downloading captions for {video_id}: track={caption_id}, lang={lang}")

        # Step 3: Download caption track as SRT
        dl_resp = httpx.get(
            f"{YOUTUBE_API_URL}/captions/{caption_id}",
            params={"tfmt": "srt"},
            headers=headers,
            timeout=30,
        )

        if dl_resp.status_code == 429:
            raise Exception("429 Too Many Requests (official API)")

        dl_resp.raise_for_status()
        srt_text = dl_resp.text

        # Step 4: Parse SRT → plain text (strip timestamps and sequence numbers)
        lines = []
        for line in srt_text.split("\n"):
            line = line.strip()
            # Skip empty lines, sequence numbers, and timestamp lines
            if not line:
                continue
            if line.isdigit():
                continue
            if "-->" in line:
                continue
            lines.append(line)

        transcript = " ".join(lines)
        logger.info(f"Official captions downloaded for {video_id}: {len(transcript)} chars")
        return transcript

    except Exception as e:
        error_str = str(e).lower()
        if "429" in error_str or "too many" in error_str:
            raise  # Re-raise rate limits for scheduler to handle
        logger.error(f"Official captions API error for {video_id}: {e}")
        return None


def get_transcript(video_url: str) -> str | None:
    """Get transcript from a YouTube video URL using official Captions API only.

    Requires OAuth 2.0 credentials (YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN).
    No unofficial scraping — fully compliant with YouTube Terms of Service.
    """
    # Extract video ID from URL
    video_id = None
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'^([a-zA-Z0-9_-]{11})$',
    ]
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            video_id = match.group(1)
            break

    if not video_id:
        logger.error(f"Could not extract video ID from: {video_url}")
        return None

    # Official API only — no scraping
    if not settings.youtube_client_id or not settings.youtube_client_secret or not settings.youtube_refresh_token:
        logger.warning(f"YouTube OAuth not configured — cannot fetch transcript for {video_id}. "
                       "Set YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, YOUTUBE_REFRESH_TOKEN.")
        return None

    logger.info(f"Using official YouTube Captions API for {video_id}")
    return _get_transcript_official(video_id)


def extract_knowledge_from_transcript(transcript: str, video_title: str = "") -> dict:
    """Use Claude to extract business knowledge from a YouTube video transcript."""
    client = Anthropic(api_key=settings.anthropic_api_key)

    prompt = f"""Analyze this YouTube video transcript from Ketu (owner of Sale91.com / Own Knitted Blank Wears — a B2B plain t-shirt manufacturer).

Video title: {video_title}
Transcript:
{transcript[:5000]}

Extract the following (in JSON format):
1. "product_info": Any product details, specifications, new products mentioned
2. "pricing": Any prices mentioned
3. "business_knowledge": Business tips, market info, industry knowledge shared
4. "faqs_covered": Any common questions answered in the video
5. "key_points": 3-5 main takeaways from this video

Return ONLY valid JSON."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text

        # Track API cost
        from core.cost_tracker import track_api_cost
        track_api_cost(
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            source="youtube-analysis",
        )
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"status": "parse_error", "raw": result_text}

    except Exception as e:
        logger.error(f"Transcript analysis error: {e}")
        return {"status": "error", "detail": str(e)}


def process_video(video_url: str, video_title: str = "") -> dict:
    """Full pipeline: get transcript → extract knowledge → suggest updates."""
    transcript = get_transcript(video_url)
    if not transcript:
        return {"status": "no_transcript", "video_url": video_url}

    knowledge = extract_knowledge_from_transcript(transcript, video_title)

    # Save extracted knowledge for review
    learned_dir = KNOWLEDGE_DIR / "learned"
    learned_dir.mkdir(exist_ok=True)

    # Save with video ID as filename
    video_id_match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', video_url)
    filename = video_id_match.group(1) if video_id_match else "unknown"

    file_data = {"video_url": video_url, "title": video_title, "knowledge": knowledge}
    file_content = json.dumps(file_data, indent=2, ensure_ascii=False)

    # Save to DB first (primary — survives deploys)
    from core.database import is_db_available, save_learned_file
    if is_db_available():
        save_learned_file(
            f"yt_{filename}.json",
            file_content,
            metadata={"video_url": video_url, "title": video_title},
        )

    output_path = learned_dir / f"yt_{filename}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_content)

    # Auto-persist YouTube learned file to GitHub (backup)
    from core.git_persist import persist_single_file
    persist_single_file(
        f"knowledge/learned/yt_{filename}.json",
        output_path,
        source=f"youtube-{filename}",
    )

    return {
        "status": "ok",
        "video_url": video_url,
        "knowledge": knowledge,
        "saved_to": str(output_path),
    }
