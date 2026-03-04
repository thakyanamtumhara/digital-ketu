"""Complaint & Escalation Detection for Digital Ketu.

Detects when a customer is angry, frustrated, or complaining about quality/defects.
When detected, flags the conversation for real Ketu to handle personally,
and adjusts the AI reply tone to be empathetic rather than salesy.

Scenarios detected:
- Quality complaints (defect, shrinkage, color bleeding, wrong product)
- Angry/frustrated tone
- Demand for refund/replacement
- Urgency signals (repeated messages, ALL CAPS)
- Profanity or rude language
"""

import logging
import re

logger = logging.getLogger(__name__)

# Escalation levels
LEVEL_NONE = "none"
LEVEL_ATTENTION = "attention"    # Needs careful reply, but AI can handle
LEVEL_ESCALATE = "escalate"     # Must escalate to real Ketu


def detect_escalation(message: str, conversation_history: list | None = None) -> dict:
    """Analyze a customer message for complaints/escalation signals.

    Returns:
        {
            "level": "none" | "attention" | "escalate",
            "reason": str,
            "prompt_modifier": str,  # Extra instruction for AI
        }
    """
    msg = message.strip()
    msg_lower = msg.lower()

    # === ESCALATE immediately — real Ketu must handle ===

    # Quality complaints / defects
    complaint_patterns = [
        r"(defect|kharab|toot|fat|phat|fatt|damage|broken|tear|tore)",
        r"(shrink|sikud|sukhad|chota ho gaya|size change)",
        r"(color\s*(bleed|fade|uttar|nikal)|rang.*nikal|rang.*uttar)",
        r"(wrong\s*(product|item|color|size)|galat\s*(maal|product|size|color))",
        r"(fraud|scam|dhoka|cheat|loot|thug)",
        r"(worst|bakwas|bekar|ghatiya|low quality|cheap quality)",
        r"(refund|paisa wapas|money back|return kar|replace kar)",
        r"(complaint|complain|shikayat|grievance)",
        r"(legal|consumer\s*forum|court|case\s*kar)",
    ]

    for pattern in complaint_patterns:
        if re.search(pattern, msg_lower):
            return {
                "level": LEVEL_ESCALATE,
                "reason": "quality_complaint",
                "prompt_modifier": (
                    "IMPORTANT: Customer is complaining about quality/defect. "
                    "Be VERY empathetic and apologetic. Do NOT try to sell anything. "
                    "Say: 'Sir, bahut sorry for the inconvenience. Ye Ketu sir personally dekhenge. "
                    "Aapko jaldi se jaldi solution milega. Ketu sir aapse seedha baat karenge.' "
                    "Do NOT make promises about refund/replacement — only Ketu can decide that."
                ),
            }

    # Profanity / extreme anger
    anger_patterns = [
        r"(bc|mc|bhenchod|madarchod|chutiya|bhosd|gand|haramkhor|saala|kamina)",
        r"(idiot|stupid|fool|useless|pathetic|disgusting|horrible|terrible)",
    ]
    for pattern in anger_patterns:
        if re.search(pattern, msg_lower):
            return {
                "level": LEVEL_ESCALATE,
                "reason": "angry_customer",
                "prompt_modifier": (
                    "IMPORTANT: Customer is very angry. Stay calm, respectful, and brief. "
                    "Do NOT argue or justify. Just say: 'Sir, samajh sakta hun aap upset hain. "
                    "Ketu sir ko abhi inform kar raha hun, wo personally aapse baat karenge. Sorry for the trouble.' "
                    "Keep reply to 1-2 lines MAX."
                ),
            }

    # === ATTENTION — AI can handle but carefully ===

    # Mild frustration
    frustration_patterns = [
        r"(reply nahi|jawab do|answer do|ignore mat|respond karo)",
        r"(kab tak|kitna time|bahut late|delay|der|slow|waiting|wait kar)",
        r"(not happy|satisfied nahi|disappointed|acha nahi|theek nahi)",
        r"(problem|issue|dikkat|pareshani|mushkil)",
    ]
    for pattern in frustration_patterns:
        if re.search(pattern, msg_lower):
            return {
                "level": LEVEL_ATTENTION,
                "reason": "mild_frustration",
                "prompt_modifier": (
                    "Customer seems slightly frustrated. Be extra helpful and responsive. "
                    "Address their concern directly. Don't be salesy. Show urgency in helping."
                ),
            }

    # Repeated ALL CAPS (shouting)
    caps_words = [w for w in msg.split() if len(w) > 2 and w.isupper()]
    if len(caps_words) >= 3:
        return {
            "level": LEVEL_ATTENTION,
            "reason": "caps_urgency",
            "prompt_modifier": (
                "Customer is using ALL CAPS — they want urgent attention. "
                "Be prompt and direct. Address their concern immediately."
            ),
        }

    return {
        "level": LEVEL_NONE,
        "reason": "",
        "prompt_modifier": "",
    }


def format_escalation_notice(phone: str, name: str, message: str, reason: str) -> str:
    """Format a notice for Ketu about an escalated customer.

    This message is logged and can be sent to Ketu's admin WhatsApp.
    """
    reason_labels = {
        "quality_complaint": "Quality/Defect Complaint",
        "angry_customer": "Angry Customer",
        "mild_frustration": "Frustrated Customer",
        "caps_urgency": "Urgent Message",
    }
    label = reason_labels.get(reason, reason)
    name_str = f" ({name})" if name else ""
    return (
        f"⚠️ ESCALATION: {label}\n"
        f"Customer: {phone}{name_str}\n"
        f"Message: {message[:200]}\n"
        f"Action needed: Please reply personally."
    )
