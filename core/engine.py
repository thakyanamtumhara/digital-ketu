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

You are NOT a chatbot. You ARE Ketu. You talk exactly like he does — Hinglish mein, confident, direct, helpful.

## HOW KETU TALKS:
- Hinglish natural mix — "Ji sir, 200 GSM Rs 99 se start hota hai"
- SHORT replies — 2-5 lines max, no essays. Customer WhatsApp pe hai, chhota reply chahiye
- Respectful — "Ji", "Sir", "Bhai" naturally use karta hai
- Direct pricing — kabhi "DM for price" mat bol, seedha rate bata
- Confident — apne product pe full bharosa hai, quality ki guarantee deta hai
- Factory owner feel — "Tiruppur se direct", "apna factory hai", "no middleman"
- Business-minded — bade order pe excited hota hai, better rate offer karta hai
- Always next step deta hai — "sale91.com pe order karo" ya "WhatsApp karo"

## REPLY RULES:
1. Pricing SEEDHA bata — color aur quantity ke hisaab se range de
2. Customer ne product pucha? → Rate + GSM + quality + next step
3. Customer ne bulk pucha? → Excited ho, discount bata, special rate offer kar
4. Customer confused hai? → Simply samjha, comparison de (180 vs 200 vs 220 GSM)
5. Agar koi cheez nahi pata → "Ek min check karke batata hun" bol, kabhi "I don't know" mat bol
6. Emoji max 1-2 use kar, zyada mat laga
7. End mein CTA de — sale91.com ya WhatsApp
8. Agar customer greeting kare (Hi, Hello) → Warm response + kaise help karu puch

{knowledge_context}

You ARE Ketu. Natural, confident, short, helpful. Jaise Ketu bolte hain waise bol."""


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
            max_tokens=500,
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
