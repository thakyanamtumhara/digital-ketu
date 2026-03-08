"""Smart context selector — classifies customer messages and picks only relevant knowledge.

Uses keyword matching (zero API cost) to determine what the customer is asking about,
then returns only the relevant sections of knowledge in ULTRA-COMPACT format.

TOKEN BUDGET: Max ~1,200 tokens for knowledge context.
Everything is compressed to one-liners. No verbose descriptions.
This keeps cost under ₹0.50/reply even as knowledge base grows.
"""

import logging
import re

from core.token_budget import estimate_tokens, truncate_to_budget, BUDGET_KNOWLEDGE_TOKENS

logger = logging.getLogger(__name__)

# --- Intent categories with keyword patterns ---
INTENT_KEYWORDS = {
    "price_product": {
        "keywords": {
            "price", "rate", "kitna hai", "kitne ka", "kitne mein", "cost", "sasta", "mehnga", "expensive",
            "cheap", "budget", "rs", "rupees", "rupee", "paisa", "amount",
            "bulk", "wholesale", "per piece", "discount", "offer",
        },
        "sections": ["matched_products", "pricing_note", "bulk_discounts", "payment_terms"],
    },
    "product_inquiry": {
        "keywords": {
            "oversize", "oversized", "round neck", "polo", "hoodie", "hoody",
            "sweatshirt", "jacket", "varsity", "shorts", "kids", "boxy",
            "acidwash", "acid wash", "sublimation", "tshirt", "t-shirt", "tee",
            "product", "catalogue", "catalog", "milega",
            "stock", "ready", "color", "colour", "size", "range",
        },
        "sections": ["matched_products", "pricing_note"],
    },
    "gsm_fabric": {
        "keywords": {
            "gsm", "fabric", "cotton", "polyester", "biowash", "bio wash", "bio-wash",
            "supercombed", "loopknit", "brushed", "terry", "thickness", "mota", "patla",
            "weight", "heavy", "light", "quality", "material", "kapda",
        },
        "sections": ["gsm_guide", "matched_products"],
    },
    "printing": {
        "keywords": {
            "print", "printing", "dtg", "dtf", "screen", "sublimation", "embroidery",
            "heat press", "htv", "design", "custom", "logo",
        },
        "sections": ["printing_compatibility", "matched_products"],
    },
    "shipping_delivery": {
        "keywords": {
            "delivery", "shipping", "dispatch", "courier", "transport", "kitne din",
            "kab milega", "track", "tracking", "porter", "rapido", "speed",
        },
        "sections": ["shipping"],
    },
    "payment": {
        "keywords": {
            "payment", "pay", "upi", "bank transfer", "neft", "imps", "cod",
            "cash on delivery", "prepaid", "online payment",
        },
        "sections": ["payment_terms"],
    },
    "order_how": {
        "keywords": {
            "order", "kaise karu", "how to order", "buy", "kharidna", "purchase",
            "website", "link", "checkout", "sample",
        },
        "sections": ["company_basic", "payment_terms", "shipping"],
    },
    "dropshipping": {
        "keywords": {
            "dropship", "dropshipping", "blind", "resell", "reseller",
        },
        "sections": ["dropshipping"],
    },
    "location_visit": {
        "keywords": {
            "factory", "warehouse", "location", "address", "kahan", "where", "visit",
            "office", "showroom", "tiruppur", "delhi", "khanpur",
        },
        "sections": ["company_locations"],
    },
    "return_complaint": {
        "keywords": {
            "return", "refund", "exchange", "defect", "damage", "problem", "issue",
            "complaint", "quality issue", "hole", "tear", "shrink", "color bleed",
        },
        "sections": ["company_returns"],
    },
    "gst_invoice": {
        "keywords": {
            "gst", "invoice", "bill", "tax",
        },
        "sections": ["gst"],
    },
    "greeting": {
        "keywords": {
            "hi", "hello", "hey", "hlo", "namaste", "namaskar",
            "good morning", "good evening", "good afternoon",
        },
        "sections": ["company_basic"],
    },
    "moq": {
        "keywords": {
            "minimum", "moq", "kam se kam", "least", "kitne se",
            "minimum order", "kitna order",
        },
        "sections": ["pricing_note"],
    },
}

# Product matching keywords
PRODUCT_KEYWORDS = {
    "oversize": ["oversize-210gsm", "oversize-240gsm", "oversize-180gsm"],
    "oversized": ["oversize-210gsm", "oversize-240gsm", "oversize-180gsm"],
    "drop shoulder": ["oversize-210gsm", "oversize-240gsm", "oversize-180gsm"],
    "boxy": ["boxy-fit"],
    "acidwash": ["acidwash-oversize"],
    "acid wash": ["acidwash-oversize"],
    "round neck": ["true-biowash-round-neck", "biowash-round-neck", "non-bio-round-neck"],
    "biowash": ["true-biowash-round-neck", "biowash-round-neck"],
    "non bio": ["non-bio-round-neck"],
    "sublimation": ["sublimation-t-shirt"],
    "polo": ["premium-polo", "cotton-polo"],
    "hoodie": ["zip-hoodie", "hoodie-320gsm-black", "hoodie-320gsm", "dropshoulder-hoodie-430gsm", "hoodie-430gsm"],
    "hoody": ["zip-hoodie", "hoodie-320gsm-black", "hoodie-320gsm", "dropshoulder-hoodie-430gsm", "hoodie-430gsm"],
    "zip hoodie": ["zip-hoodie"],
    "sweatshirt": ["sweatshirt", "sweatshirt-2"],
    "varsity": ["varsity-jacket"],
    "jacket": ["varsity-jacket"],
    "kids": ["kids-round-neck"],
    "bacche": ["kids-round-neck"],
    "child": ["kids-round-neck"],
    "shorts": ["shorts"],
    "bottom": ["shorts"],
}

# GSM to product mapping
GSM_PRODUCT_MAP = {
    "180": ["oversize-180gsm", "boxy-fit", "true-biowash-round-neck", "biowash-round-neck", "non-bio-round-neck", "kids-round-neck"],
    "200": ["sublimation-t-shirt"],
    "210": ["oversize-210gsm"],
    "220": ["premium-polo", "cotton-polo"],
    "240": ["oversize-240gsm", "acidwash-oversize", "shorts"],
    "320": ["zip-hoodie", "hoodie-320gsm-black", "hoodie-320gsm", "sweatshirt", "sweatshirt-2", "varsity-jacket"],
    "430": ["dropshoulder-hoodie-430gsm", "hoodie-430gsm"],
}


def classify_message(message: str) -> dict:
    """Classify a customer message into intent categories using keyword matching."""
    msg = message.strip().lower()
    msg_normalized = msg.replace("-", " ").replace("_", " ")
    words = set(re.split(r'[\s,.\-!?]+', msg_normalized))

    detected_intents = []
    sections_needed = set()
    product_ids = set()

    for intent, config in INTENT_KEYWORDS.items():
        keywords = config["keywords"]
        matched = False
        for kw in keywords:
            if " " in kw:
                if kw in msg_normalized:
                    matched = True
                    break
            else:
                if kw in words:
                    matched = True
                    break
        if matched:
            detected_intents.append(intent)
            sections_needed.update(config["sections"])

    for kw, pids in PRODUCT_KEYWORDS.items():
        if " " in kw:
            if kw in msg_normalized:
                product_ids.update(pids)
        else:
            if kw in words:
                product_ids.update(pids)

    gsm_matches = re.findall(r'\b(\d{3})\b', msg)
    for gsm in gsm_matches:
        if gsm in GSM_PRODUCT_MAP:
            product_ids.update(GSM_PRODUCT_MAP[gsm])
            if "gsm_fabric" not in detected_intents:
                detected_intents.append("gsm_fabric")
                sections_needed.update(INTENT_KEYWORDS["gsm_fabric"]["sections"])

    if product_ids:
        sections_needed.add("matched_products")

    is_complex = len(detected_intents) == 0
    if is_complex:
        logger.info(f"[ContextSelector] Could not classify: '{msg[:60]}' — using minimal context")

    result = {
        "intents": detected_intents,
        "product_ids": list(product_ids),
        "sections_needed": sections_needed,
        "is_complex": is_complex,
    }

    if detected_intents:
        logger.info(
            f"[ContextSelector] Classified: '{msg[:40]}' → intents={detected_intents}, "
            f"products={len(product_ids)}, sections={len(sections_needed)}"
        )

    return result


def format_smart_context(knowledge: dict, classification: dict) -> str:
    """Build ULTRA-COMPACT knowledge context within token budget.

    Everything is compressed to one-liners. Max 1,200 tokens total.
    As knowledge grows, this function ensures we NEVER exceed the budget.
    """
    if classification["is_complex"]:
        return _format_minimal_context(knowledge)

    sections_needed = classification["sections_needed"]
    product_ids = set(classification["product_ids"])
    parts = []

    # --- Company basic (one-liner) ---
    if "company_basic" in sections_needed:
        parts.append("Sale91.com | B2B blank wears | Tiruppur factory, Delhi warehouse | MOQ 10 | Catalog: sale91.com/catalog")

    # --- Matched products (one-liner per product) ---
    if "matched_products" in sections_needed:
        products = knowledge.get("products", {})
        catalog = products.get("catalog", [])

        if product_ids:
            matched = [p for p in catalog if p.get("id") in product_ids]
        else:
            matched = catalog

        if matched:
            product_lines = []
            for item in matched[:6]:  # Max 6 products to cap tokens
                if "bulk_price" in item:
                    price_str = f"₹{item['bulk_price']}bulk/₹{item['sample_price']}sample"
                else:
                    price_str = item.get("price_range", "N/A")
                # Include color names (max 5 + count) — customers ask "kaun se color hai"
                all_colors = item.get("colors", [])
                if len(all_colors) > 5:
                    colors_str = ", ".join(all_colors[:5]) + f" +{len(all_colors)-5}more"
                else:
                    colors_str = ", ".join(all_colors) if all_colors else "N/A"
                product_lines.append(
                    f"- {item['name']} | {item['gsm']}GSM | {price_str} | {item.get('fabric', '')} | Colors: {colors_str}"
                )
            parts.append("PRODUCTS:\n" + "\n".join(product_lines))

    # --- Pricing note (one-liner) ---
    if "pricing_note" in sections_needed:
        products = knowledge.get("products", {})
        price_note = products.get("price_note", "")
        if price_note:
            parts.append(f"PRICING: {price_note}")

    # --- Bulk discounts (one-liner) ---
    if "bulk_discounts" in sections_needed:
        products = knowledge.get("products", {})
        discounts = products.get("bulk_discounts", {})
        if discounts:
            d_str = " | ".join(f"{k}: {v}" for k, v in discounts.items())
            parts.append(f"DISCOUNTS: {d_str}")

    # --- GSM guide (compact) ---
    if "gsm_guide" in sections_needed:
        products = knowledge.get("products", {})
        gsm_guide = products.get("gsm_guide", {})
        if gsm_guide:
            gsm_str = " | ".join(f"{gsm}GSM={desc}" for gsm, desc in gsm_guide.items())
            parts.append(f"GSM: {gsm_str}")

    # --- Printing compatibility (compact) ---
    if "printing_compatibility" in sections_needed:
        products = knowledge.get("products", {})
        printing = products.get("printing_compatibility", {})
        if printing:
            p_str = " | ".join(f"{k}: {v}" for k, v in printing.items())
            parts.append(f"PRINTING: {p_str}")

    # --- Payment terms (one-liner) ---
    if "payment_terms" in sections_needed:
        c = knowledge.get("company", {})
        pt = c.get("payment_terms", {})
        if pt:
            modes = ", ".join(pt.get("modes", []))
            parts.append(f"PAYMENT: {pt.get('policy', '100% Prepaid')} | {modes} | Website pe ₹2/pc discount")

    # --- Shipping (one-liner) ---
    if "shipping" in sections_needed:
        c = knowledge.get("company", {})
        s = c.get("shipping", {})
        if s:
            delivery = s.get("delivery_time", {})
            parts.append(f"SHIPPING: Dispatch within minutes | Delhi NCR 1-2hrs | PAN India 1-3 days")

    # --- Dropshipping (one-liner) ---
    if "dropshipping" in sections_needed:
        parts.append("DROPSHIPPING: Zero-contact blind dropshipping, no branding, no setup fee, no monthly charge")

    # --- Locations (one-liner) ---
    if "company_locations" in sections_needed:
        parts.append("LOCATIONS: Factory=Tiruppur,TN (no visit) | Warehouse=Khanpur,South Delhi (pickup Mon-Sat 10-6, Sun 11-4)")

    # --- Returns (one-liner) ---
    if "company_returns" in sections_needed:
        parts.append("RETURNS: Manufacturing defect pe replacement (photo bhejo) | No shrinkage, no color bleeding guarantee")

    # --- GST (one-liner) ---
    if "gst" in sections_needed:
        parts.append("GST: 5% extra on all prices | GST invoice har order ke saath")

    # --- PICK relevant FAQs (max 2, keyword-matched) ---
    # Not ALL FAQs — only the ones matching this customer's question.
    # As FAQs grow to 100+, we still only send 2 most relevant ones.
    faqs = knowledge.get("faq", {}).get("faqs", [])
    if faqs:
        relevant_faqs = _pick_relevant_faqs(faqs, classification)
        if relevant_faqs:
            faq_lines = [f"Q: {f['question']} A: {f['answer']}" for f in relevant_faqs]
            parts.append("FAQ:\n" + "\n".join(faq_lines))

    # --- PICK relevant learned patterns (max 3, keyword-matched) ---
    # As patterns grow to 200+, we still only send the 3 most relevant.
    style = knowledge.get("style", {})
    learned_patterns = style.get("learned_patterns", [])
    if learned_patterns:
        relevant_patterns = _pick_relevant_patterns(learned_patterns, classification)
        if relevant_patterns:
            parts.append("LEARNED: " + " | ".join(relevant_patterns))

    # --- PICK relevant evolved rules (max 2) ---
    prompt_data = knowledge.get("prompt", {})
    evolved_rules = prompt_data.get("evolved_rules", [])
    if evolved_rules:
        relevant_rules = _pick_relevant_patterns(evolved_rules, classification)
        if relevant_rules:
            parts.append("EXTRA RULES: " + " | ".join(relevant_rules))

    context = "\n".join(parts)

    # HARD CAP — truncate if over budget (should rarely happen with compact format)
    estimated = estimate_tokens(context)
    if estimated > BUDGET_KNOWLEDGE_TOKENS:
        context = truncate_to_budget(context, BUDGET_KNOWLEDGE_TOKENS)
        logger.warning(f"[ContextSelector] Knowledge context truncated: {estimated} → ~{BUDGET_KNOWLEDGE_TOKENS} tokens")
    else:
        logger.info(f"[ContextSelector] Knowledge context: ~{estimated} tokens (budget: {BUDGET_KNOWLEDGE_TOKENS})")

    return context


def _format_minimal_context(knowledge: dict) -> str:
    """Ultra-minimal context for unclassified messages.

    Just product price list + company one-liner. ~400 tokens max.
    """
    parts = []

    parts.append("Sale91.com | B2B blank wears manufacturer | Tiruppur factory, Delhi warehouse | MOQ 10 | 100% prepaid | Dispatch within minutes")

    # Product price list — one-liner per product, no details
    products = knowledge.get("products", {})
    catalog = products.get("catalog", [])
    if catalog:
        product_lines = []
        for item in catalog:
            if "bulk_price" in item:
                price_str = f"₹{item['bulk_price']}"
            else:
                price_str = item.get("price_range", "N/A")
            product_lines.append(f"- {item['name']} {item['gsm']}GSM {price_str}")
        parts.append("PRODUCTS:\n" + "\n".join(product_lines))

    context = "\n".join(parts)

    estimated = estimate_tokens(context)
    if estimated > BUDGET_KNOWLEDGE_TOKENS:
        context = truncate_to_budget(context, BUDGET_KNOWLEDGE_TOKENS)

    logger.info(f"[ContextSelector] Minimal context: ~{estimated} tokens")
    return context


def _pick_relevant_faqs(faqs: list, classification: dict) -> list:
    """Pick only FAQs relevant to the customer's message. Max 2.

    Uses keyword overlap between FAQ keywords and the detected intents/products.
    As FAQs grow to 100+, this ensures we only send 2 most relevant.
    """
    # Build a keyword set from classification
    intents = classification.get("intents", [])
    product_ids = classification.get("product_ids", [])

    # Map intents to FAQ-matching keywords
    intent_keywords = {
        "price_product": {"rate", "price", "kitna", "discount", "bulk"},
        "product_inquiry": {"available", "sample", "stock"},
        "gsm_fabric": {"gsm", "biowash", "fabric", "thickness"},
        "printing": {"print", "dtg", "dtf", "screen", "sublimation", "embroidery"},
        "shipping_delivery": {"delivery", "shipping", "dispatch"},
        "payment": {"cod", "payment", "prepaid"},
        "order_how": {"order", "sample", "website"},
        "dropshipping": {"dropship", "blind", "resell"},
        "location_visit": {"factory", "warehouse", "location", "address"},
        "return_complaint": {"return", "refund", "defect"},
        "gst_invoice": {"gst", "invoice", "tax"},
        "moq": {"minimum", "moq"},
    }

    # Also add product names as matching keywords
    product_keywords = set()
    for pid in product_ids:
        # "hoodie-320gsm" → {"hoodie", "320gsm"}
        product_keywords.update(pid.replace("-", " ").split())

    search_keywords = product_keywords.copy()
    for intent in intents:
        search_keywords.update(intent_keywords.get(intent, set()))

    if not search_keywords:
        return []

    # Score FAQs by keyword overlap
    scored = []
    for faq in faqs:
        if faq.get("status") == "inactive":
            continue
        faq_kw = set(kw.lower() for kw in faq.get("keywords", []))
        # Also check FAQ question text for product keywords
        q_lower = faq.get("question", "").lower()
        overlap = len(faq_kw & search_keywords)
        # Bonus if product name appears in FAQ question
        for pk in product_keywords:
            if pk in q_lower:
                overlap += 2
        if overlap > 0:
            scored.append((overlap, faq))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [faq for _, faq in scored[:2]]  # Max 2 FAQs


def _pick_relevant_patterns(patterns: list, classification: dict) -> list:
    """Pick learned patterns relevant to the customer's message. Max 4.

    [Correction]-tagged patterns (from Ketu's direct edits) are ALWAYS included
    because they represent the strongest learning signal.
    Other patterns are keyword-matched against the classification.
    """
    if not patterns:
        return []

    intents = set(classification.get("intents", []))
    product_ids = classification.get("product_ids", [])

    # Separate [Correction] patterns (always include) from others (keyword-match)
    correction_patterns = []
    other_patterns = []
    for p in patterns:
        text = p if isinstance(p, str) else str(p)
        if "[Correction]" in text:
            correction_patterns.append(text[:150])
        else:
            other_patterns.append(text)

    # Always include latest 2 correction patterns — most recent = most relevant
    result = correction_patterns[-2:]

    # Keyword-match remaining patterns
    search_terms = set()
    for pid in product_ids:
        search_terms.update(pid.replace("-", " ").split())
    for intent in intents:
        search_terms.update(intent.replace("_", " ").split())

    if search_terms:
        scored = []
        for text in other_patterns:
            text_lower = text.lower()
            score = sum(1 for term in search_terms if term in text_lower)
            if score > 0:
                scored.append((score, text[:150]))
        scored.sort(key=lambda x: x[0], reverse=True)
        result += [text for _, text in scored[:2]]  # Add 2 keyword-matched

    return result[:4]  # Max 4 total


# Keep for backwards compatibility but should not be used in normal flow
def _format_full_context(knowledge: dict) -> str:
    """Full context fallback — DEPRECATED, kept for backwards compat only."""
    from core.knowledge import format_context
    return format_context()
