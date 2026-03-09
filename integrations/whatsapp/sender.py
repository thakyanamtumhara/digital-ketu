import logging
import time

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

GRAPH_API_URL = "https://graph.facebook.com/v22.0"

# Track last sent message per customer for editing (phone -> {message_id, text, timestamp})
_last_sent_messages: dict[str, dict] = {}
MESSAGE_EDIT_WINDOW = 900  # 15 minutes (WhatsApp limit)


async def send_text_message(to: str, message: str) -> dict | None:
    """Send a text message via WhatsApp Business API.

    Returns the API response (contains message_id for editing).
    Also stores the message_id so it can be edited within 15 minutes.
    """
    url = f"{GRAPH_API_URL}/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": message},
    }

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Message sent to {to}: {result}")

                # Store message_id for potential editing
                msg_id = ""
                messages_list = result.get("messages", [])
                if messages_list:
                    msg_id = messages_list[0].get("id", "")
                if msg_id:
                    _last_sent_messages[to] = {
                        "message_id": msg_id,
                        "text": message,
                        "timestamp": time.time(),
                    }
                    # Clean up old entries (keep last 100)
                    if len(_last_sent_messages) > 100:
                        oldest = sorted(_last_sent_messages.items(), key=lambda x: x[1]["timestamp"])
                        for phone, _ in oldest[:20]:
                            _last_sent_messages.pop(phone, None)

                return result
        except httpx.HTTPStatusError as e:
            logger.error(f"WhatsApp API error (attempt {attempt + 1}): {e.response.status_code} {e.response.text}")
            from core.error_tracker import track_error
            track_error("whatsapp-send", f"HTTP {e.response.status_code}: {e.response.text[:100]}", {"to": to[-4:]})
            if e.response.status_code < 500:
                break  # Don't retry client errors
        except Exception as e:
            logger.error(f"Send message error (attempt {attempt + 1}): {e}")
            from core.error_tracker import track_error
            track_error("whatsapp-send", str(e), {"to": to[-4:], "attempt": attempt + 1})

    return None


async def edit_message(to: str, message_id: str, new_text: str) -> dict | None:
    """Edit a previously sent WhatsApp message.

    WhatsApp allows editing messages within 15 minutes of sending.
    The customer sees the updated message with an "(edited)" label.

    Args:
        to: Customer phone number
        message_id: The wamid of the message to edit
        new_text: The new message text
    """
    url = f"{GRAPH_API_URL}/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {"body": new_text},
        "context": {"message_id": message_id},
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.put(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            logger.info(f"Message edited for {to}: {message_id[:20]}... -> {new_text[:50]}")

            # Update stored message
            if to in _last_sent_messages and _last_sent_messages[to]["message_id"] == message_id:
                _last_sent_messages[to]["text"] = new_text

            return result
    except httpx.HTTPStatusError as e:
        logger.error(f"Edit message API error: {e.response.status_code} {e.response.text}")
        from core.error_tracker import track_error
        track_error("whatsapp-edit", f"HTTP {e.response.status_code}: {e.response.text[:100]}", {"to": to[-4:]})
        return None
    except Exception as e:
        logger.error(f"Edit message error: {e}")
        from core.error_tracker import track_error
        track_error("whatsapp-edit", str(e), {"to": to[-4:]})
        return None


async def edit_last_message(to: str, new_text: str) -> dict | None:
    """Edit the last message sent to a customer.

    Convenience wrapper — finds the last sent message_id automatically.
    Only works within 15 minutes of sending.

    Returns:
        API response dict on success, None if no editable message found.
    """
    last = _last_sent_messages.get(to)
    if not last:
        logger.warning(f"No sent message found for {to} to edit")
        return None

    # Check if within edit window
    elapsed = time.time() - last["timestamp"]
    if elapsed > MESSAGE_EDIT_WINDOW:
        logger.warning(f"Edit window expired for {to}: {elapsed:.0f}s > {MESSAGE_EDIT_WINDOW}s")
        return None

    return await edit_message(to=to, message_id=last["message_id"], new_text=new_text)


def get_last_sent_message(to: str) -> dict | None:
    """Get the last sent message to a customer (for checking if editable)."""
    last = _last_sent_messages.get(to)
    if not last:
        return None
    elapsed = time.time() - last["timestamp"]
    return {
        "message_id": last["message_id"],
        "text": last["text"],
        "seconds_ago": int(elapsed),
        "editable": elapsed < MESSAGE_EDIT_WINDOW,
        "edit_window_remaining": max(0, int(MESSAGE_EDIT_WINDOW - elapsed)),
    }


async def send_image_message(
    to: str,
    image_url: str,
    caption: str = "",
) -> dict | None:
    """Send an image message via WhatsApp Business API.

    Used for sending product images when customer asks about a specific product.
    Zero API cost — just a WhatsApp API call.
    """
    url = f"{GRAPH_API_URL}/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    image_payload = {"link": image_url}
    if caption:
        image_payload["caption"] = caption

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "image",
        "image": image_payload,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            logger.info(f"[Image] Sent to {to}: {image_url[:60]}")
            return result
    except Exception as e:
        logger.error(f"[Image] Send failed to {to}: {e}")
        return None


async def send_template_message(
    to: str,
    template_name: str,
    language: str = "en",
    components: list | None = None,
) -> dict | None:
    """Send a template message via WhatsApp Business API."""
    url = f"{GRAPH_API_URL}/{settings.whatsapp_phone_number_id}/messages"
    headers = {
        "Authorization": f"Bearer {settings.whatsapp_access_token}",
        "Content-Type": "application/json",
    }
    template = {
        "name": template_name,
        "language": {"code": language},
    }
    if components:
        template["components"] = components

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "template",
        "template": template,
    }

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Template send error: {e}")
        return None
