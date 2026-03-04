import hashlib
import hmac
import logging
import time
from collections import OrderedDict

from fastapi import APIRouter, Request, Response, HTTPException

from core.config import settings
from core.engine import generate_reply
from core.conversation_log import log_conversation, get_recent_conversations, get_last_ai_reply, mark_corrected
from integrations.whatsapp.sender import send_text_message
from learner.audio_transcriber import process_whatsapp_audio
from learner.realtime_learner import buffer_conversation, learn_from_correction

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

                # --- Admin commands (Ketu's own phone) ---
                if settings.admin_phone and _normalize_phone(sender) == _normalize_phone(settings.admin_phone):
                    await _handle_admin_command(sender, text)
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

                # Log conversation for later review/correction
                log_conversation(
                    customer_phone=sender,
                    customer_name=name,
                    customer_message=text,
                    ai_reply=reply,
                )

                # Buffer for realtime learning (non-blocking)
                buffer_conversation(
                    customer_message=text,
                    ai_reply=reply,
                    customer_name=name,
                    customer_phone=sender,
                )

    return {"status": "ok"}


def _normalize_phone(phone: str) -> str:
    """Normalize phone number — remove +, spaces, leading 0."""
    phone = phone.strip().replace("+", "").replace(" ", "").replace("-", "")
    if phone.startswith("0"):
        phone = "91" + phone[1:]
    return phone


async def _handle_admin_command(sender: str, text: str):
    """Handle commands from Ketu's admin phone on WhatsApp.

    Commands:
    - /review or review → List last 10 AI conversations (numbered)
    - <number>: <correction> → Correct conversation #N and learn
    - /status → Quick status check

    Examples:
    - Ketu sends: "/review"
    - Digital Ketu replies with numbered list of recent AI conversations
    - Ketu sends: "3: Bhai rate 180 hai, 170 nahi"
    - Digital Ketu corrects #3, sends correct reply to customer, learns
    """
    text_lower = text.strip().lower()
    logger.info(f"Admin command from {sender}: {text[:60]}")

    # --- /review command ---
    if text_lower in ("/review", "review", "/check", "check", "/list", "list"):
        conversations = get_recent_conversations(limit=10)
        if not conversations:
            await send_text_message(
                to=sender,
                message="Koi recent AI conversation nahi hai abhi."
            )
            return

        lines = ["*Recent AI Replies:*\n"]
        for i, c in enumerate(conversations, 1):
            phone_last4 = c["customer_phone"][-4:] if c["customer_phone"] else "????"
            corrected_tag = " [CORRECTED]" if c.get("corrected") else ""
            lines.append(
                f"*{i}.* ({c.get('time', '')} | ...{phone_last4} {c.get('customer_name', '')}){corrected_tag}\n"
                f"   Customer: _{c['customer_message'][:60]}_\n"
                f"   AI Reply: {c['ai_reply'][:80]}\n"
            )

        lines.append("\n_Correct karne ke liye:_\n"
                      "_Number bhejo: 3: Sahi answer yahan likho_")

        await send_text_message(to=sender, message="\n".join(lines))
        return

    # --- /status command ---
    if text_lower in ("/status", "status"):
        from learner.realtime_learner import get_realtime_stats
        stats = get_realtime_stats()
        conversations = get_recent_conversations(limit=1)
        last_reply = conversations[0] if conversations else None

        msg = (
            f"*Digital Ketu Status:*\n"
            f"Auto-reply: {'ON' if settings.auto_reply_enabled else 'OFF'}\n"
            f"Buffer: {stats['buffer_size']} conversations\n"
            f"Last reply: {last_reply['time'] + ' to ...' + last_reply['customer_phone'][-4:] if last_reply else 'None'}\n"
        )
        await send_text_message(to=sender, message=msg)
        return

    # --- /on and /off commands ---
    if text_lower in ("/on", "on"):
        settings.auto_reply_enabled = True
        await send_text_message(to=sender, message="Auto-reply ON kar diya hai.")
        return

    if text_lower in ("/off", "off"):
        settings.auto_reply_enabled = False
        await send_text_message(to=sender, message="Auto-reply OFF kar diya hai.")
        return

    # --- Correction: "3: Sahi answer yahan" ---
    import re
    correction_match = re.match(r'^(\d{1,2})\s*[:\.]\s*(.+)', text, re.DOTALL)
    if correction_match:
        conv_num = int(correction_match.group(1))
        correction_text = correction_match.group(2).strip()

        if not correction_text:
            await send_text_message(to=sender, message="Correction text khali hai. Example: 3: Sahi answer likho")
            return

        conversations = get_recent_conversations(limit=10)
        if conv_num < 1 or conv_num > len(conversations):
            await send_text_message(
                to=sender,
                message=f"Number {conv_num} galat hai. /review bhejo pehle, phir 1-{len(conversations)} mein se choose karo."
            )
            return

        conv = conversations[conv_num - 1]

        if conv.get("corrected"):
            await send_text_message(to=sender, message=f"#{conv_num} already corrected hai.")
            return

        # 1. Send correct reply to customer
        await send_text_message(to=conv["customer_phone"], message=correction_text)
        logger.info(f"Correction sent to {conv['customer_phone']}: {correction_text[:50]}...")

        # 2. Mark as corrected
        mark_corrected(conv["customer_phone"])

        # 3. Learn from correction (background)
        import threading
        def _learn():
            result = learn_from_correction(
                customer_message=conv["customer_message"],
                ai_reply=conv["ai_reply"],
                ketu_correction=correction_text,
                customer_phone=conv["customer_phone"],
                customer_name=conv.get("customer_name", ""),
            )
            if result.get("status") == "learned":
                from core.activity_log import log_activity
                log_activity(
                    source="correction-learner",
                    action="learned",
                    details={
                        "customer_message": conv["customer_message"][:80],
                        "ai_reply": conv["ai_reply"][:80],
                        "ketu_correction": correction_text[:80],
                        "what_went_wrong": result.get("what_went_wrong", ""),
                        "updates_applied": result.get("updates_applied", []),
                        "via": "whatsapp-admin",
                    },
                    items_count=result.get("count", 0),
                )
                logger.info(f"Correction learned: {result.get('what_went_wrong', '')}")

        thread = threading.Thread(target=_learn, daemon=True)
        thread.start()

        # 4. Confirm to Ketu
        phone_last4 = conv["customer_phone"][-4:]
        await send_text_message(
            to=sender,
            message=(
                f"Done! #{conv_num} corrected.\n"
                f"Customer (...{phone_last4}) ko sahi reply bhej diya.\n"
                f"AI bhi seekh raha hai is correction se."
            ),
        )
        return

    # --- Unknown command ---
    await send_text_message(
        to=sender,
        message=(
            "*Admin Commands:*\n"
            "/review — Last 10 AI conversations dekho\n"
            "3: Sahi answer — #3 correct karo\n"
            "/status — Quick status\n"
            "/on — Auto-reply ON\n"
            "/off — Auto-reply OFF"
        ),
    )
