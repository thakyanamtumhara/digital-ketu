import json
import logging
import re

from anthropic import Anthropic

from core.config import settings, KNOWLEDGE_DIR

logger = logging.getLogger(__name__)


def get_transcript(video_url: str) -> str | None:
    """Get transcript from a YouTube video URL."""
    from youtube_transcript_api import YouTubeTranscriptApi

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

    try:
        # Try Hindi first, then English
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_transcript(["hi", "hi-IN"])
        except Exception:
            try:
                transcript = transcript_list.find_transcript(["en"])
            except Exception:
                # Get auto-generated
                transcript = transcript_list.find_generated_transcript(["hi", "en"])

        entries = transcript.fetch()
        return " ".join(entry.text for entry in entries)

    except Exception as e:
        logger.error(f"Transcript fetch error for {video_url}: {e}")
        return None


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

    output_path = learned_dir / f"yt_{filename}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(file_content)

    # Save to DB (primary — survives deploys)
    from core.database import is_db_available, save_learned_file
    if is_db_available():
        save_learned_file(
            f"yt_{filename}.json",
            file_content,
            metadata={"video_url": video_url, "title": video_title},
        )

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
