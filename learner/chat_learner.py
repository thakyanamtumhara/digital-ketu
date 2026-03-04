import json
import logging
import re
from pathlib import Path

from anthropic import Anthropic

from core.config import settings

logger = logging.getLogger(__name__)
KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"


def parse_whatsapp_export(chat_text: str) -> list[dict]:
    """Parse WhatsApp chat export text into structured messages."""
    messages = []
    # WhatsApp export format: [DD/MM/YY, HH:MM:SS] Name: Message
    pattern = r'\[?(\d{1,2}/\d{1,2}/\d{2,4}),?\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*(?:AM|PM|am|pm)?\]?\s*-?\s*([^:]+):\s*(.*)'

    for line in chat_text.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            date, time_str, sender, text = match.groups()
            messages.append({
                "date": date.strip(),
                "time": time_str.strip(),
                "sender": sender.strip(),
                "text": text.strip(),
            })
        elif messages and line.strip():
            # Continuation of previous message
            messages[-1]["text"] += "\n" + line.strip()

    return messages


def extract_knowledge_from_messages(messages: list[dict], ketu_name: str = "Ketu") -> dict:
    """Use Claude to extract knowledge from Ketu's manual messages."""
    client = Anthropic(api_key=settings.anthropic_api_key)

    # Filter only Ketu's messages (manual typing by the owner)
    ketu_messages = [m for m in messages if ketu_name.lower() in m["sender"].lower()]

    if not ketu_messages:
        return {"status": "no_messages", "updates": []}

    # Build conversation context for Claude to analyze
    chat_sample = "\n".join(
        f"[{m['date']} {m['time']}] {m['sender']}: {m['text']}"
        for m in messages[:200]  # Last 200 messages for context
    )

    prompt = f"""Analyze these WhatsApp chat messages from Ketu (the business owner of Sale91.com / Own Knitted Blank Wears).

IMPORTANT: Only learn from messages sent BY Ketu (not by customers or AI).
Ketu's name in chat: {ketu_name}

Chat messages:
{chat_sample}

Extract the following (in JSON format):
1. "new_products": Any new products mentioned with prices
2. "price_updates": Any price changes mentioned
3. "style_patterns": How Ketu talks — phrases, greetings, closing patterns
4. "new_faqs": New question-answer pairs from customer interactions
5. "business_updates": Any new business info (offers, policies, etc.)

Return ONLY valid JSON. If nothing new found, return empty arrays."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        result_text = response.content[0].text
        # Try to parse JSON from response
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"status": "parse_error", "raw": result_text}

    except Exception as e:
        logger.error(f"Knowledge extraction error: {e}")
        return {"status": "error", "detail": str(e)}


def extract_knowledge_from_wwbun_messages(
    messages: list[dict],
    owner_user_id: str,
) -> dict:
    """Extract knowledge from wwbun database messages.

    CRITICAL: Only learn from messages sent by the owner (manual typing).
    AI-generated messages should be tagged and excluded.

    Args:
        messages: List of message dicts from wwbun database
        owner_user_id: The user ID of Ketu (to identify his manual messages)
    """
    client = Anthropic(api_key=settings.anthropic_api_key)

    # Filter: only messages sent by owner, exclude AI-generated ones
    manual_messages = [
        m for m in messages
        if m.get("sender_id") == owner_user_id
        and not m.get("is_ai_generated", False)
    ]

    if not manual_messages:
        return {"status": "no_manual_messages", "updates": []}

    # Include customer messages for context (to understand what Ketu was replying to)
    # but only learn FROM Ketu's messages
    chat_context = []
    for m in messages[:200]:
        role = "KETU" if m.get("sender_id") == owner_user_id else "CUSTOMER"
        is_ai = " [AI]" if m.get("is_ai_generated") else ""
        chat_context.append(f"{role}{is_ai}: {m.get('content', '')}")

    chat_text = "\n".join(chat_context)

    prompt = f"""Analyze these WhatsApp conversations. Learn ONLY from KETU's messages (NOT from [AI] tagged or CUSTOMER messages).

Messages:
{chat_text}

Extract in JSON format:
1. "style_patterns": How Ketu types — his phrases, greetings, tone, typical replies
2. "price_updates": Any prices Ketu mentioned
3. "new_faqs": Q&A pairs where customer asked and Ketu answered
4. "business_updates": Any new policies, offers, shipping info
5. "product_updates": Any new product info Ketu shared

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
        logger.error(f"Knowledge extraction error: {e}")
        return {"status": "error", "detail": str(e)}


def apply_knowledge_updates(updates: dict) -> dict:
    """Apply extracted knowledge to the JSON files."""
    applied = []

    # Update FAQ
    if updates.get("new_faqs"):
        faq_path = KNOWLEDGE_DIR / "faq.json"
        with open(faq_path, "r", encoding="utf-8") as f:
            faq_data = json.load(f)

        existing_questions = {faq["question"].lower() for faq in faq_data["faqs"]}

        for faq in updates["new_faqs"]:
            q = faq.get("question", "")
            a = faq.get("answer", "")
            if q and a and q.lower() not in existing_questions:
                faq_data["faqs"].append({
                    "question": q,
                    "answer": a,
                    "keywords": faq.get("keywords", []),
                    "source": "auto_learned",
                })
                applied.append(f"New FAQ: {q}")

        with open(faq_path, "w", encoding="utf-8") as f:
            json.dump(faq_data, f, indent=2, ensure_ascii=False)

    # Update style patterns
    if updates.get("style_patterns"):
        style_path = KNOWLEDGE_DIR / "style.json"
        with open(style_path, "r", encoding="utf-8") as f:
            style_data = json.load(f)

        patterns = updates["style_patterns"]
        if isinstance(patterns, list):
            for pattern in patterns:
                if isinstance(pattern, str) and pattern not in style_data.get("tone_words", []):
                    style_data.setdefault("learned_patterns", []).append(pattern)
                    applied.append(f"New pattern: {pattern}")
        elif isinstance(patterns, dict):
            style_data.setdefault("learned_patterns", []).append(patterns)
            applied.append("New style patterns learned")

        with open(style_path, "w", encoding="utf-8") as f:
            json.dump(style_data, f, indent=2, ensure_ascii=False)

    return {"applied": applied, "count": len(applied)}
