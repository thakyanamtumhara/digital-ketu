import hashlib
import hmac
import logging
import time
from collections import OrderedDict

from fastapi import APIRouter, Request, Response, HTTPException

from core.config import settings
from core.engine import generate_reply
from integrations.whatsapp.sender import send_text_message
from learner.audio_transcriber import process_whatsapp_audio
from learner.realtime_learner import buffer_conversation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook/whatsapp", tags=["whatsapp"])

# Duplicate message protection — track last 1000 message IDs with timestamps
_seen_messages: OrderedDict[str, float] = OrderedDict()
_SEEN_MAX = 1000
_SEEN_TTL = 300  # 5 minutes


def _is_duplicate(message_id: str) -> bool:
    """Check if message was already processed. Returns True if duplicate."""
    now = time.time()
    # Clean expired entries
    expired = [k for k, t in _seen_messages.items() if now - t > _SEEN_TTL]
    for k in expired:
        _seen_messages.pop(k, None)

    if message_id in _seen_messages:
        return True

    _seen_messages[message_id] = now
    # Trim to max size
    while len(_seen_messages) > _SEEN_MAX:
        _seen_messages.popitem(last=False)
    return False


def verify_signature(payload: bytes, signature: str) -> bool:
    if not settings.whatsapp_app_secret:
        return True  # Skip verification if no secret configured
    expected = hmac.new(
        settings.whatsapp_app_secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@router.get("")
async def verify_webhook(request: Request):
    """WhatsApp webhook verification (GET request from Meta)."""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == settings.whatsapp_verify_token:
        logger.info("WhatsApp webhook verified")
        return Response(content=challenge, media_type="text/plain")

    raise HTTPException(status_code=403, detail="Verification failed")


@router.post("")
async def receive_message(request: Request):
    """Handle incoming WhatsApp messages."""
    body = await request.body()

    # Verify signature
    signature = request.headers.get("X-Hub-Signature-256", "")
    if settings.whatsapp_app_secret and not verify_signature(body, signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    data = await request.json()

    # Process each entry
    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            messages = value.get("messages", [])

            for msg in messages:
                sender = msg.get("from", "")
                msg_type = msg.get("type", "")
                msg_id = msg.get("id", "")

                # Skip if no sender or message type
                if not sender or not msg_type:
                    continue

                # Duplicate protection — skip already-processed messages
                if msg_id and _is_duplicate(msg_id):
                    logger.debug(f"Duplicate message {msg_id} from {sender}, skipping")
                    continue

                # Extract message text
                if msg_type == "text":
                    text = msg.get("text", {}).get("body", "")
                elif msg_type == "interactive":
                    interactive = msg.get("interactive", {})
                    if interactive.get("type") == "button_reply":
                        text = interactive.get("button_reply", {}).get("title", "")
                    elif interactive.get("type") == "list_reply":
                        text = interactive.get("list_reply", {}).get("title", "")
                    else:
                        text = ""
                elif msg_type == "audio":
                    # Transcribe audio using Whisper
                    audio_info = msg.get("audio", {})
                    media_id = audio_info.get("id", "")
                    if media_id and settings.openai_api_key:
                        text = await process_whatsapp_audio(media_id) or ""
                        if text:
                            logger.info(f"Audio transcribed from {sender}: {text[:60]}...")
                    else:
                        text = ""
                elif msg_type in ("image", "video", "document", "sticker", "location", "contacts"):
                    # Unsupported media — send polite reply
                    if settings.auto_reply_enabled:
                        media_reply = (
                            "Ji sir, abhi main sirf text aur audio messages samajh pata hun. "
                            "Aap text mein bata dijiye, main turant help karunga!"
                        )
                        await send_text_message(to=sender, message=media_reply)
                        logger.info(f"Unsupported media ({msg_type}) from {sender}, sent polite reply")
                    continue
                else:
                    text = ""

                if not text:
                    continue

                if not settings.auto_reply_enabled:
                    logger.info(f"Auto-reply disabled, skipping message from {sender}")
                    continue

                # Get contact name if available
                contacts = value.get("contacts", [])
                name = ""
                if contacts:
                    profile = contacts[0].get("profile", {})
                    name = profile.get("name", "")

                logger.info(f"Message from {sender} ({name}): {text}")

                # Generate AI reply
                reply = generate_reply(
                    message=text,
                    customer_phone=sender,
                    customer_name=name,
                )

                # Send reply
                await send_text_message(to=sender, message=reply)
                logger.info(f"Replied to {sender}: {reply[:50]}...")

                # Buffer for realtime learning (non-blocking)
                buffer_conversation(
                    customer_message=text,
                    ai_reply=reply,
                    customer_name=name,
                    customer_phone=sender,
                )

    return {"status": "ok"}
