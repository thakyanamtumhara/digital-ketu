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

# --- Rate Limiting (reads from config, adjustable via env vars or admin commands) ---
_customer_msg_timestamps: dict[str, list[float]] = {}
_global_reply_timestamps: list[float] = []
_RATE_LIMIT_WINDOW = 3600  # 1 hour window

# Track rate-limited messages for dashboard visibility
_rate_limited_today: dict[str, int] = {}  # phone -> count
_rate_limited_total: int = 0
_rate_limited_date: str = ""


def get_rate_limit_stats() -> dict:
    """Get rate limiting stats for dashboard."""
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d %b %Y")

    # Reset if new day
    global _rate_limited_total, _rate_limited_date, _rate_limited_today
    if _rate_limited_date != today:
        _rate_limited_today = {}
        _rate_limited_total = 0
        _rate_limited_date = today

    return {
        "today_blocked": _rate_limited_total,
        "top_blocked_phones": sorted(
            _rate_limited_today.items(), key=lambda x: x[1], reverse=True
        )[:5],
        "current_limits": {
            "per_customer_per_hour": settings.rate_limit_per_customer,
            "global_per_hour": settings.rate_limit_global,
            "night_per_customer": settings.rate_limit_night_per_customer,
            "night_hours": f"{settings.rate_limit_night_start}:00-{settings.rate_limit_night_end}:00 IST",
        },
    }


def _is_rate_limited(phone: str) -> str | None:
    """Check if this customer or global rate is exceeded.
    Returns reason string if limited, None if OK.
    """
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW

    per_cust = settings.rate_limit_per_customer
    global_limit = settings.rate_limit_global
    night_limit = settings.rate_limit_night_per_customer

    # Per-customer check
    if phone in _customer_msg_timestamps:
        _customer_msg_timestamps[phone] = [
            t for t in _customer_msg_timestamps[phone] if t > cutoff
        ]
        count = len(_customer_msg_timestamps[phone])
        if count >= per_cust:
            _track_rate_limited(phone)
            return f"customer_limit ({count}/{per_cust} per hour)"
    else:
        _customer_msg_timestamps[phone] = []

    # Night mode check (IST)
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    hour_ist = datetime.now(ist).hour
    is_night = hour_ist >= settings.rate_limit_night_start or hour_ist < settings.rate_limit_night_end
    if is_night:
        count = len(_customer_msg_timestamps.get(phone, []))
        if count >= night_limit:
            _track_rate_limited(phone)
            return f"night_limit ({count}/{night_limit} between {settings.rate_limit_night_start}:00-{settings.rate_limit_night_end}:00 IST)"

    # Global check
    _global_reply_timestamps[:] = [t for t in _global_reply_timestamps if t > cutoff]
    if len(_global_reply_timestamps) >= global_limit:
        _track_rate_limited(phone)
        return f"global_limit ({len(_global_reply_timestamps)}/{global_limit} per hour)"

    return None


def _track_rate_limited(phone: str):
    """Track rate-limited messages for dashboard stats."""
    global _rate_limited_total, _rate_limited_date, _rate_limited_today
    from datetime import datetime, timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    today = datetime.now(ist).strftime("%d %b %Y")
    if _rate_limited_date != today:
        _rate_limited_today = {}
        _rate_limited_total = 0
        _rate_limited_date = today
    _rate_limited_total += 1
    _rate_limited_today[phone] = _rate_limited_today.get(phone, 0) + 1


def _record_reply(phone: str):
    """Record that a reply was sent (for rate tracking)."""
    now = time.time()
    _customer_msg_timestamps.setdefault(phone, []).append(now)
    _global_reply_timestamps.append(now)

    # Trim old data to prevent memory growth
    if len(_customer_msg_timestamps) > 500:
        oldest_phones = sorted(
            _customer_msg_timestamps.keys(),
            key=lambda p: _customer_msg_timestamps[p][-1] if _customer_msg_timestamps[p] else 0,
        )[:250]
        for p in oldest_phones:
            del _customer_msg_timestamps[p]


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

                # Rate limit check — prevent spam/abuse/cost burning
                rate_reason = _is_rate_limited(sender)
                if rate_reason:
                    logger.warning(f"Rate limited {sender}: {rate_reason}")
                    # Don't reply — silently skip to avoid engaging spammers
                    # Only send a polite message for the first rate-limit hit
                    if "customer_limit" in rate_reason or "night_limit" in rate_reason:
                        count = len(_customer_msg_timestamps.get(sender, []))
                        if count == settings.rate_limit_per_customer or count == settings.rate_limit_night_per_customer:
                            await send_text_message(
                                to=sender,
                                message="Ji, bahut saare messages aa rahe hain. Thodi der mein reply karta hun. Aap sale91.com pe bhi dekh sakte hain.",
                            )
                    continue

                # Generate AI reply
                reply = generate_reply(
                    message=text,
                    customer_phone=sender,
                    customer_name=name,
                )

                # Send reply
                await send_text_message(to=sender, message=reply)
                _record_reply(sender)
                logger.info(f"Replied to {sender}: {reply[:50]}...")

                # Log conversation for later review/correction
                log_conversation(
                    customer_phone=sender,
                    customer_name=name,
                    customer_message=text,
                    ai_reply=reply,
                )

                # Also log to activity (backup — visible in dashboard)
                from core.activity_log import log_activity
                log_activity(
                    source="whatsapp",
                    action="ai-reply",
                    details={
                        "customer_phone": sender,
                        "customer_name": name,
                        "customer_message": text[:200],
                        "ai_reply": reply[:200],
                    },
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

    # --- /limit command (view/set rate limits) ---
    if text_lower.startswith("/limit"):
        parts = text.strip().split()
        if len(parts) == 1:
            # Show current limits
            rl_stats = get_rate_limit_stats()
            msg = (
                f"*Rate Limits:*\n"
                f"Per customer: {settings.rate_limit_per_customer}/hr\n"
                f"Global: {settings.rate_limit_global}/hr\n"
                f"Night ({settings.rate_limit_night_start}:00-{settings.rate_limit_night_end}:00): {settings.rate_limit_night_per_customer}/customer\n"
                f"Blocked today: {rl_stats['today_blocked']}\n\n"
                f"_Change: /limit customer 15_\n"
                f"_Or: /limit global 200_\n"
                f"_Or: /limit night 5_"
            )
            await send_text_message(to=sender, message=msg)
            return
        elif len(parts) >= 3:
            import re as _re
            target = parts[1].lower()
            try:
                val = int(parts[2])
            except ValueError:
                await send_text_message(to=sender, message="Number dijiye. Example: /limit customer 15")
                return
            if target in ("customer", "per_customer"):
                settings.rate_limit_per_customer = val
                await send_text_message(to=sender, message=f"Per-customer limit: {val}/hr set.")
            elif target == "global":
                settings.rate_limit_global = val
                await send_text_message(to=sender, message=f"Global limit: {val}/hr set.")
            elif target == "night":
                settings.rate_limit_night_per_customer = val
                await send_text_message(to=sender, message=f"Night limit: {val}/customer set.")
            else:
                await send_text_message(to=sender, message="Options: customer, global, night")
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
