import hashlib
import hmac
import logging

from fastapi import APIRouter, Request, Response, HTTPException

from core.config import settings
from core.engine import generate_reply
from integrations.whatsapp.sender import send_text_message

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook/whatsapp", tags=["whatsapp"])


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
                else:
                    # For media messages, acknowledge
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

    return {"status": "ok"}
