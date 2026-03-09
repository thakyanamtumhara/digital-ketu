"""Smart Reply Confidence Scoring.

Before sending an AI reply, scores confidence (0-100) based on:
- FAQ match quality (exact keyword matches vs vague)
- Customer buying stage (repeat buyers = more context)
- Topic familiarity (known intents vs unclassified)
- Ketu-only proximity (question close to ketu-only patterns but didn't trigger)

Low-confidence replies (below threshold) get auto-deferred to Ketu
instead of risking fabrication.
"""

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Confidence threshold — below this, defer to Ketu
CONFIDENCE_DEFER_THRESHOLD = 25

# Weights for each scoring component (total = 100)
WEIGHT_FAQ_MATCH = 35       # Does the question match a known FAQ?
WEIGHT_INTENT_CLARITY = 25  # Was the intent clearly classified?
WEIGHT_CUSTOMER_STAGE = 15  # Do we know this customer well?
WEIGHT_SAFETY = 25          # Is the topic safe (not close to ketu-only)?


def score_confidence(
    message: str,
    classification: dict | None,
    customer_phone: str = "",
    faq_data: list | None = None,
    ketu_only_config: dict | None = None,
) -> dict:
    """Score confidence for replying to this message.

    Returns:
        {
            "score": 0-100,
            "should_defer": bool,
            "components": {faq_match, intent_clarity, customer_stage, safety},
            "reason": str  # Why low/high confidence
        }
    """
    msg_lower = message.strip().lower()
    components = {}
    reasons = []

    # --- 1. FAQ Match Score (0-35) ---
    faq_score = _score_faq_match(msg_lower, faq_data or [])
    components["faq_match"] = faq_score
    if faq_score >= 25:
        reasons.append("strong FAQ match")
    elif faq_score < 10:
        reasons.append("no FAQ match")

    # --- 2. Intent Clarity Score (0-25) ---
    intent_score = _score_intent_clarity(classification)
    components["intent_clarity"] = intent_score
    if intent_score >= 20:
        reasons.append("clear intent")
    elif intent_score < 8:
        reasons.append("unclear intent")

    # --- 3. Customer Stage Score (0-15) ---
    stage_score = _score_customer_stage(customer_phone)
    components["customer_stage"] = stage_score

    # --- 4. Safety Score (0-25) ---
    safety_score = _score_safety(msg_lower, ketu_only_config)
    components["safety"] = safety_score
    if safety_score < 10:
        reasons.append("close to ketu-only territory")

    total = sum(components.values())
    should_defer = total < CONFIDENCE_DEFER_THRESHOLD

    if should_defer:
        reasons.append(f"score {total} < threshold {CONFIDENCE_DEFER_THRESHOLD}")

    return {
        "score": total,
        "should_defer": should_defer,
        "components": components,
        "reason": "; ".join(reasons) if reasons else "normal",
    }


def _score_faq_match(msg_lower: str, faqs: list) -> int:
    """Score how well the message matches known FAQs (0-35)."""
    if not faqs:
        return 15  # No FAQs loaded = neutral

    best_keyword_hits = 0
    best_similarity = 0.0

    for faq in faqs:
        if not isinstance(faq, dict):
            continue
        # Skip inactive FAQs
        if faq.get("status") == "inactive":
            continue

        # Keyword match count
        keywords = faq.get("keywords", [])
        hits = sum(1 for kw in keywords if kw.lower() in msg_lower)
        if hits > best_keyword_hits:
            best_keyword_hits = hits

        # Question similarity
        q = faq.get("question", "").lower()
        if q:
            sim = SequenceMatcher(None, msg_lower, q).ratio()
            if sim > best_similarity:
                best_similarity = sim

    # Score: keyword hits (0-20) + similarity (0-15)
    kw_score = min(best_keyword_hits * 7, 20)
    sim_score = int(best_similarity * 15)

    return min(kw_score + sim_score, 35)


def _score_intent_clarity(classification: dict | None) -> int:
    """Score how clearly the message intent was classified (0-25)."""
    if not classification:
        return 5  # No classification = low

    intents = classification.get("intents", [])
    if not intents:
        return 5

    # Complex/unclassified messages
    if classification.get("is_complex"):
        return 8

    # Single clear intent = high confidence
    if len(intents) == 1:
        return 25

    # Multiple intents = moderate (some ambiguity)
    if len(intents) == 2:
        return 18

    # 3+ intents = lower
    return 12


def _score_customer_stage(customer_phone: str) -> int:
    """Score based on how well we know this customer (0-15)."""
    if not customer_phone:
        return 5  # Unknown customer = low

    try:
        from core.customer_memory import get_profile
        profile = get_profile(customer_phone)

        stage = profile.get("stage", "new")
        stage_scores = {
            "new": 5,
            "inquiry": 8,
            "interested": 10,
            "negotiating": 12,
            "ready": 13,
            "bought": 15,
            "repeat": 15,
        }
        return stage_scores.get(stage, 5)
    except Exception:
        return 5


def _score_safety(msg_lower: str, ketu_only_config: dict | None) -> int:
    """Score safety — how far from ketu-only territory (0-25).

    High = safe topic. Low = dangerously close to questions only Ketu can answer.
    """
    if not ketu_only_config:
        return 20  # No config = assume safe

    # Check proximity to ketu-only keywords (partial matches)
    danger_keywords = set()
    for cat in ketu_only_config.get("categories", []):
        for kw in cat.get("keywords", []):
            danger_keywords.add(kw.lower())

    # Count how many danger keywords partially appear
    partial_hits = 0
    for dk in danger_keywords:
        # Partial match: keyword appears as substring
        if dk in msg_lower:
            partial_hits += 2  # Direct hit
        elif any(word in dk for word in msg_lower.split() if len(word) > 3):
            partial_hits += 1  # Partial overlap

    # Check regex patterns proximity
    for cat in ketu_only_config.get("categories", []):
        for pattern in cat.get("patterns", []):
            try:
                if re.search(pattern, msg_lower):
                    partial_hits += 3  # Pattern match = very close
            except re.error:
                continue

    # More hits = less safe
    if partial_hits == 0:
        return 25
    elif partial_hits <= 1:
        return 18
    elif partial_hits <= 3:
        return 10
    else:
        return 3  # Very close to ketu-only territory
