import logging

from fastapi import APIRouter, Request

from core.config import settings
from core.engine import generate_reply
from integrations.whatsapp.sender import send_text_message, send_template_message

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/webhook/indiamart", tags=["indiamart"])


def extract_lead_info(lead: dict) -> dict:
    """Extract relevant info from IndiaMART lead data."""
    return {
        "name": lead.get("SENDER_NAME", lead.get("sender_name", "")),
        "phone": lead.get("SENDER_MOBILE", lead.get("sender_mobile", "")),
        "email": lead.get("SENDER_EMAIL", lead.get("sender_email", "")),
        "city": lead.get("SENDER_CITY", lead.get("sender_city", "")),
        "product": lead.get("QUERY_PRODUCT_NAME", lead.get("query_product_name", "")),
        "message": lead.get("QUERY_MESSAGE", lead.get("query_message", "")),
        "quantity": lead.get("QUERY_MCAT_NAME", lead.get("query_mcat_name", "")),
    }


def build_lead_context(lead_info: dict) -> str:
    """Build context string from lead info for AI engine."""
    parts = []
    if lead_info["name"]:
        parts.append(f"Customer name: {lead_info['name']}")
    if lead_info["city"]:
        parts.append(f"City: {lead_info['city']}")
    if lead_info["product"]:
        parts.append(f"Interested in: {lead_info['product']}")
    if lead_info["quantity"]:
        parts.append(f"Category: {lead_info['quantity']}")
    if lead_info["message"]:
        parts.append(f"Their message: {lead_info['message']}")
    return "\n".join(parts)


@router.post("")
async def handle_indiamart_lead(request: Request):
    """Handle incoming IndiaMART lead webhook."""
    data = await request.json()

    # IndiaMART sends leads in various formats
    leads = data if isinstance(data, list) else [data]

    results = []
    for lead in leads:
        lead_info = extract_lead_info(lead)

        if not lead_info["phone"]:
            logger.warning("Lead without phone number, skipping")
            continue

        # Normalize phone number (add 91 prefix if needed)
        phone = lead_info["phone"].replace(" ", "").replace("-", "")
        if not phone.startswith("91") and len(phone) == 10:
            phone = f"91{phone}"

        logger.info(f"IndiaMART lead: {lead_info['name']} ({phone}) - {lead_info['product']}")

        if not settings.auto_reply_enabled:
            logger.info("Auto-reply disabled, skipping")
            results.append({"phone": phone, "status": "skipped"})
            continue

        # Build message context from lead info
        context = build_lead_context(lead_info)
        message = (
            f"IndiaMART se inquiry aayi hai:\n{context}\n\n"
            "Is customer ko reply karo jaise Ketu karta hai. "
            "Pehle greet karo, product info do, price batao, aur next step batao."
        )

        # Generate AI reply
        reply = generate_reply(
            message=message,
            customer_phone=phone,
            customer_name=lead_info["name"],
        )

        # Log conversation for review/correction (same as WhatsApp)
        try:
            from core.conversation_log import log_conversation
            customer_msg = lead_info["message"] or lead_info["product"] or "IndiaMART inquiry"
            log_conversation(phone, lead_info["name"], customer_msg, reply)
        except Exception as e:
            logger.warning(f"Failed to log IndiaMART conversation: {e}")

        # Send via WhatsApp (template first for new contacts, then text)
        await send_text_message(to=phone, message=reply)
        logger.info(f"IndiaMART lead reply sent to {phone}")

        results.append({"phone": phone, "status": "replied", "reply": reply[:100]})

    return {"status": "ok", "results": results}
