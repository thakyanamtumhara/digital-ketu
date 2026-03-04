import time
import logging
from anthropic import Anthropic

from core.config import settings
from core.knowledge import format_context

logger = logging.getLogger(__name__)

# In-memory conversation history per customer (phone -> messages)
_conversations: dict[str, list] = {}
_conversation_timestamps: dict[str, float] = {}
CONVERSATION_TTL = 3600  # 1 hour

SYSTEM_PROMPT = """You are Digital Ketu — the AI twin of Ketu, owner of Own Knitted Blank Wears (Sale91.com).

You reply EXACTLY like Ketu would. You are NOT a chatbot — you ARE Ketu talking to customers.

CRITICAL RULES:
1. Reply in Hinglish (Hindi + English mix) — this is how Ketu talks
2. Keep replies SHORT — 3-5 lines MAX. Never write paragraphs.
3. Be respectful — use "Ji", "Sir", "Bhai" naturally
4. Give DIRECT pricing — never hide prices, never say "DM for price"
5. Always mention: factory rate, Tiruppur se direct, no middleman
6. Always give next step: sale91.com link, or WhatsApp karo
7. Be confident about quality: Bio-washed, Combed Cotton, Pre-shrunk
8. If unsure, say "Ek min check karke batata hun" — never say "I don't know"
9. For large orders, show excitement and offer better rates
10. End with a clear call-to-action

{knowledge_context}

Remember: You ARE Ketu. Reply naturally in his voice. Short, confident, helpful."""


def _build_system_prompt() -> str:
    context = format_context()
    return SYSTEM_PROMPT.format(knowledge_context=context)


def _cleanup_old_conversations():
    now = time.time()
    expired = [
        phone for phone, ts in _conversation_timestamps.items()
        if now - ts > CONVERSATION_TTL
    ]
    for phone in expired:
        _conversations.pop(phone, None)
        _conversation_timestamps.pop(phone, None)


def get_conversation_history(phone: str) -> list:
    _cleanup_old_conversations()
    return _conversations.get(phone, [])


def generate_reply(
    message: str,
    customer_phone: str = "",
    customer_name: str = "",
    conversation_history: list | None = None,
) -> str:
    client = Anthropic(api_key=settings.anthropic_api_key)

    # Use provided history or fetch from in-memory store
    if conversation_history:
        # Map roles from wwbun format to Claude API format
        # wwbun sends "customer"/"owner" but Claude needs "user"/"assistant"
        messages = []
        for msg in conversation_history:
            role = msg.get("role", "user")
            if role in ("customer", "user"):
                role = "user"
            elif role in ("owner", "assistant", "ai"):
                role = "assistant"
            else:
                role = "user"
            messages.append({"role": role, "content": msg.get("content", "")})
    elif customer_phone:
        messages = get_conversation_history(customer_phone)
    else:
        messages = []

    # Add current message
    messages = messages + [{"role": "user", "content": message}]

    # Add customer context if available
    system = _build_system_prompt()
    if customer_name:
        system += f"\n\nCustomer name: {customer_name}"

    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            system=system,
            messages=messages,
        )

        reply = response.content[0].text

        # Store conversation history
        if customer_phone:
            _conversations[customer_phone] = messages + [
                {"role": "assistant", "content": reply}
            ]
            # Keep only last 20 messages
            if len(_conversations[customer_phone]) > 20:
                _conversations[customer_phone] = _conversations[customer_phone][-20:]
            _conversation_timestamps[customer_phone] = time.time()

        return reply

    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return "Ji sir, ek chhota sa technical issue aa gaya. Thodi der mein reply karta hun. Aap sale91.com pe check kar sakte hain."
