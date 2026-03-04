import logging

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

GRAPH_API_URL = "https://graph.facebook.com/v22.0"


async def send_text_message(to: str, message: str) -> dict | None:
    """Send a text message via WhatsApp Business API."""
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
