"""Smart context selector — classifies customer messages and picks only relevant knowledge.

Uses keyword matching (zero API cost) to determine what the customer is asking about,
then returns only the relevant sections of knowledge to include in the system prompt.

This reduces token count by ~60-70% per API call while maintaining reply quality.
"""

import logging
import re

logger = logging.getLogger(__name__)

# --- Intent categories with keyword patterns ---
# Each category maps to which knowledge sections are needed.

INTENT_KEYWORDS = {
    "price_product": {
        "keywords": {
            "price", "rate", "kitna hai", "kitne ka", "kitne mein", "cost", "sasta", "mehnga", "expensive",
            "cheap", "budget", "rs", "rupees", "rupee", "paisa", "amount",
            "bulk", "wholesale", "per piece", "discount", "offer",
        },
        "sections": ["matched_products", "pricing_note", "bulk_discounts", "payment_terms", "relevant_faqs"],
    },
    "product_inquiry": {
        "keywords": {
            "oversize", "oversized", "round neck", "polo", "hoodie", "hoody",
            "sweatshirt", "jacket", "varsity", "shorts", "kids", "boxy",
            "acidwash", "acid wash", "sublimation", "tshirt", "t-shirt", "tee",
            "product", "catalogue", "catalog", "milega",
            "stock", "ready", "color", "colour", "size", "range",
        },
        "sections": ["matched_products", "pricing_note", "bulk_discounts", "relevant_faqs"],
    },
    "gsm_fabric": {
        "keywords": {
            "gsm", "fabric", "cotton", "polyester", "biowash", "bio wash", "bio-wash",
            "supercombed", "loopknit", "brushed", "terry", "thickness", "mota", "patla",
            "weight", "heavy", "light", "quality", "material", "kapda",
        },
        "sections": ["gsm_guide", "fabric_features", "matched_products", "relevant_faqs"],
    },
    "printing": {
        "keywords": {
            "print", "printing", "dtg", "dtf", "screen", "sublimation", "embroidery",
            "heat press", "htv", "design", "custom", "logo",
        },
        "sections": ["printing_compatibility", "matched_products", "relevant_faqs"],
    },
    "shipping_delivery": {
        "keywords": {
            "delivery", "shipping", "dispatch", "courier", "transport", "kitne din",
            "kab milega", "track", "tracking", "porter", "rapido", "speed",
        },
        "sections": ["shipping", "relevant_faqs"],
    },
    "payment": {
        "keywords": {
            "payment", "pay", "upi", "bank transfer", "neft", "imps", "cod",
            "cash on delivery", "prepaid", "online payment",
        },
        "sections": ["payment_terms", "relevant_faqs"],
    },
    "order_how": {
        "keywords": {
            "order", "kaise karu", "how to order", "buy", "kharidna", "purchase",
            "website", "link", "checkout", "sample",
        },
        "sections": ["company_basic", "payment_terms", "shipping", "relevant_faqs"],
    },
    "dropshipping": {
        "keywords": {
            "dropship", "dropshipping", "blind", "resell", "reseller",
        },
        "sections": ["dropshipping", "relevant_faqs"],
    },
    "location_visit": {
        "keywords": {
            "factory", "warehouse", "location", "address", "kahan", "where", "visit",
            "office", "showroom", "tiruppur", "delhi", "khanpur",
        },
        "sections": ["company_locations", "relevant_faqs"],
    },
    "return_complaint": {
        "keywords": {
            "return", "refund", "exchange", "defect", "damage", "problem", "issue",
            "complaint", "quality issue", "hole", "tear", "shrink", "color bleed",
        },
        "sections": ["company_returns", "relevant_faqs"],
    },
    "gst_invoice": {
        "keywords": {
            "gst", "invoice", "bill", "tax",
        },
        "sections": ["gst", "relevant_faqs"],
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
        },
        "sections": ["pricing_note", "relevant_faqs"],
    },
}

# Product matching keywords — maps keywords to product IDs for targeted product inclusion
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
    """Classify a customer message into intent categories using keyword matching.

    Returns:
        {
            "intents": ["price_product", "product_inquiry"],  # detected intents
            "product_ids": ["oversize-240gsm"],  # specific products mentioned
            "sections_needed": {"matched_products", "pricing_note", ...},  # union of all needed sections
            "is_complex": False,  # True if we can't classify → use full context
        }
    """
    msg = message.strip().lower()
    # Normalize common patterns
    msg_normalized = msg.replace("-", " ").replace("_", " ")
    words = set(re.split(r'[\s,.\-!?]+', msg_normalized))

    detected_intents = []
    sections_needed = set()
    product_ids = set()

    # 1. Match intents by keywords
    for intent, config in INTENT_KEYWORDS.items():
        keywords = config["keywords"]
        # Check if any keyword is in the message (word-level or substring for multi-word keywords)
        matched = False
        for kw in keywords:
            if " " in kw:
                # Multi-word keyword — check as substring
                if kw in msg_normalized:
                    matched = True
                    break
            else:
                # Single word — check in word set
                if kw in words:
                    matched = True
                    break

        if matched:
            detected_intents.append(intent)
            sections_needed.update(config["sections"])

    # 2. Match specific products
    for kw, pids in PRODUCT_KEYWORDS.items():
        if " " in kw:
            if kw in msg_normalized:
                product_ids.update(pids)
        else:
            if kw in words:
                product_ids.update(pids)

    # 3. Match GSM numbers
    gsm_matches = re.findall(r'\b(\d{3})\b', msg)
    for gsm in gsm_matches:
        if gsm in GSM_PRODUCT_MAP:
            product_ids.update(GSM_PRODUCT_MAP[gsm])
            if "gsm_fabric" not in detected_intents:
                detected_intents.append("gsm_fabric")
                sections_needed.update(INTENT_KEYWORDS["gsm_fabric"]["sections"])

    # 4. If products were identified, ensure matched_products is in sections
    if product_ids:
        sections_needed.add("matched_products")

    # 5. Determine if we can classify this
    is_complex = len(detected_intents) == 0

    # If complex (unknown intent), use full context as fallback
    if is_complex:
        logger.info(f"[ContextSelector] Could not classify: '{msg[:60]}' — using full context")

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
    """Build a trimmed knowledge context based on classification results.

    Only includes sections that are relevant to the customer's question.
    Falls back to full context if classification failed (is_complex=True).
    """
    if classification["is_complex"]:
        # Fallback — include everything (same as current behavior)
        return _format_full_context(knowledge)

    sections_needed = classification["sections_needed"]
    product_ids = set(classification["product_ids"])
    parts = []

    # --- Company basic info (always minimal) ---
    if "company_basic" in sections_needed:
        c = knowledge.get("company", {})
        websites = c.get("websites", {})
        parts.append(
            f"## Company: {c.get('name', '')} (Brand: {c.get('brand', '')})\n"
            f"- Website: {websites.get('primary', '')} | Catalog: {websites.get('catalog', '')}\n"
            f"- B2B Wholesale Manufacturer — Factory direct, no middleman\n"
            f"- Dispatch within minutes, MOQ 10 pcs"
        )

    # --- Matched products only ---
    if "matched_products" in sections_needed:
        products = knowledge.get("products", {})
        catalog = products.get("catalog", [])

        if product_ids:
            # Only include specific matched products
            matched = [p for p in catalog if p.get("id") in product_ids]
        else:
            # No specific product — include all (rare, usually product_inquiry without specifics)
            matched = catalog

        if matched:
            product_lines = []
            for item in matched:
                colors = ", ".join(item.get("colors", []))
                sizes = ", ".join(item.get("sizes", []))
                if "bulk_price" in item:
                    price_str = f"Rs {item['bulk_price']}/pc (bulk) | Rs {item['sample_price']}/pc (sample)"
                else:
                    price_str = item.get("price_range", "N/A")
                product_lines.append(
                    f"### {item['name']} ({item['gsm']} GSM)\n"
                    f"  Price: {price_str}\n"
                    f"  Fabric: {item.get('fabric', 'N/A')}\n"
                    f"  MOQ: {item.get('moq', 10)} pcs | Sizes: {sizes}\n"
                    f"  Colors ({len(item.get('colors', []))}): {colors}"
                )
            parts.append(f"## Products ({len(matched)} items)\n" + "\n\n".join(product_lines))

    # --- Pricing note ---
    if "pricing_note" in sections_needed:
        products = knowledge.get("products", {})
        price_note = products.get("price_note", "")
        if price_note:
            parts.append(f"## Pricing Note: {price_note}")

    # --- Bulk discounts ---
    if "bulk_discounts" in sections_needed:
        products = knowledge.get("products", {})
        discounts = products.get("bulk_discounts", {})
        if discounts:
            d_lines = [f"- {k}: {v}" for k, v in discounts.items()]
            parts.append("## Discounts\n" + "\n".join(d_lines))

    # --- GSM guide ---
    if "gsm_guide" in sections_needed:
        products = knowledge.get("products", {})
        gsm_guide = products.get("gsm_guide", {})
        if gsm_guide:
            gsm_lines = [f"- {gsm} GSM: {desc}" for gsm, desc in gsm_guide.items()]
            parts.append("## GSM Guide\n" + "\n".join(gsm_lines))

    # --- Fabric features ---
    if "fabric_features" in sections_needed:
        products = knowledge.get("products", {})
        features = products.get("fabric_features", {})
        if features:
            f_lines = [f"- {k}: {v}" for k, v in features.items()]
            parts.append("## Fabric Features\n" + "\n".join(f_lines))

    # --- Printing compatibility ---
    if "printing_compatibility" in sections_needed:
        products = knowledge.get("products", {})
        printing = products.get("printing_compatibility", {})
        if printing:
            p_lines = [f"- {k}: {v}" for k, v in printing.items()]
            parts.append("## Printing Compatibility\n" + "\n".join(p_lines))

    # --- Payment terms ---
    if "payment_terms" in sections_needed:
        c = knowledge.get("company", {})
        pt = c.get("payment_terms", {})
        if pt:
            parts.append(
                f"## Payment Terms\n"
                f"- Policy: {pt.get('policy', '100% Prepaid')}\n"
                f"- Modes: {', '.join(pt.get('modes', []))}\n"
                f"- Website discount: {pt.get('website_discount', '')}"
            )

    # --- Shipping ---
    if "shipping" in sections_needed:
        c = knowledge.get("company", {})
        s = c.get("shipping", {})
        if s:
            delivery = s.get("delivery_time", {})
            parts.append(
                f"## Shipping\n"
                f"- Speed: {s.get('dispatch_speed', 'Dispatch within minutes')}\n"
                f"- Delhi NCR: {delivery.get('delhi_ncr', '1-2 hours')}\n"
                f"- PAN India: {delivery.get('pan_india', '1-3 days')}"
            )

    # --- Dropshipping ---
    if "dropshipping" in sections_needed:
        c = knowledge.get("company", {})
        ds = c.get("dropshipping", {})
        if ds:
            parts.append(
                f"## Dropshipping\n"
                f"- {ds.get('type', 'Zero-contact blind dropshipping')}\n"
                f"- {ds.get('description', '')}\n"
                f"- Setup fee: {ds.get('setup_fee', 'None')}"
            )

    # --- Company locations ---
    if "company_locations" in sections_needed:
        c = knowledge.get("company", {})
        locs = c.get("locations", {})
        if locs:
            factory = locs.get("factory", {})
            warehouse = locs.get("warehouse", {})
            parts.append(
                f"## Locations\n"
                f"- Factory: {factory.get('city', '')}, {factory.get('state', '')} — {factory.get('description', '')}\n"
                f"- Warehouse: {warehouse.get('city', '')} ({warehouse.get('area', '')}) — {warehouse.get('description', '')}"
            )

    # --- Returns ---
    if "company_returns" in sections_needed:
        c = knowledge.get("company", {})
        returns = c.get("returns", {})
        if returns:
            parts.append(
                f"## Returns\n"
                f"- {returns.get('policy', '')}\n"
                f"- {returns.get('guarantee', '')}\n"
                f"- {returns.get('note', '')}"
            )

    # --- GST ---
    if "gst" in sections_needed:
        c = knowledge.get("company", {})
        gst = c.get("gst", {})
        if gst:
            parts.append(f"## GST: {gst.get('rate', '5%')} — {gst.get('note', '')}")

    # --- Relevant FAQs only ---
    if "relevant_faqs" in sections_needed:
        faqs = knowledge.get("faq", {}).get("faqs", [])
        intents = classification["intents"]
        relevant_faqs = _pick_relevant_faqs(faqs, intents, classification)
        if relevant_faqs:
            faq_lines = [f"Q: {f['question']}\nA: {f['answer']}" for f in relevant_faqs]
            parts.append("## Relevant Q&A\n" + "\n\n".join(faq_lines))

    # --- Style rules (always include — small, important for tone) ---
    if "style" in knowledge:
        s = knowledge["style"]
        parts.append("## Reply Style\n" + "\n".join(f"- {r}" for r in s.get("rules", [])))

        # Include learned patterns (important for evolution)
        learned_patterns = s.get("learned_patterns", [])
        if learned_patterns:
            import json
            parts.append("## Learned Patterns\n" + "\n".join(
                f"- {p}" if isinstance(p, str) else f"- {json.dumps(p, ensure_ascii=False)}"
                for p in learned_patterns
            ))

    # --- Skip: full example conversations, full YouTube knowledge, full avoid list ---
    # These are the biggest token hogs and least useful per-message

    return "\n\n".join(parts)


def _pick_relevant_faqs(faqs: list, intents: list, classification: dict) -> list:
    """Pick only FAQs relevant to the detected intents. Max 5."""
    # Map intents to FAQ keyword groups
    intent_faq_keywords = {
        "price_product": {"rate", "price", "kitna", "oversized", "round neck", "hoodie", "polo", "sweatshirt", "shorts", "kids", "acid", "varsity", "boxy", "sublimation"},
        "product_inquiry": {"oversized", "round neck", "hoodie", "polo", "sweatshirt", "shorts", "kids", "acid", "varsity", "boxy", "sublimation", "available", "sample"},
        "gsm_fabric": {"gsm", "bio-wash", "biowash", "thickness", "fabric"},
        "printing": {"print", "dtg", "dtf", "screen", "sublimation", "embroidery"},
        "shipping_delivery": {"delivery", "shipping", "dispatch"},
        "payment": {"cod", "payment", "pay"},
        "order_how": {"order", "sample", "website", "discount"},
        "dropshipping": {"dropship", "blind", "resell"},
        "location_visit": {"factory", "warehouse", "location", "address"},
        "return_complaint": {"return", "refund", "defect", "quality issue"},
        "gst_invoice": {"gst", "invoice", "bill", "tax"},
        "moq": {"minimum", "moq", "kam se kam"},
    }

    # Collect relevant FAQ keywords from all detected intents
    relevant_kw = set()
    for intent in intents:
        relevant_kw.update(intent_faq_keywords.get(intent, set()))

    # Score FAQs by keyword overlap
    scored = []
    for faq in faqs:
        if faq.get("status") == "inactive":
            continue
        faq_keywords = set(kw.lower() for kw in faq.get("keywords", []))
        overlap = len(faq_keywords & relevant_kw)
        if overlap > 0:
            scored.append((overlap, faq))

    # Sort by relevance, take top 5
    scored.sort(key=lambda x: x[0], reverse=True)
    return [faq for _, faq in scored[:5]]


def _format_full_context(knowledge: dict) -> str:
    """Full context fallback — same as the original format_context().

    Used when we can't classify the message (is_complex=True).
    """
    # Import the original formatter as fallback
    from core.knowledge import format_context
    return format_context()
