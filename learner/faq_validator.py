"""FAQ Validator — detects and deactivates outdated/irrelevant FAQs.

Two detection methods:
1. Catalog cross-check: After every catalog sync, checks if FAQ answers mention
   products that no longer exist in the catalog. Flags them as outdated.
2. Contradiction detection: When Ketu gives a new answer that contradicts an
   existing FAQ (e.g., "varsity band hai" vs FAQ saying "varsity available"),
   replaces the old FAQ with the new one.

Outdated FAQs are marked inactive (not deleted) so the owner can review.
"""

import json
import logging
import re
from datetime import datetime, timezone, timedelta

from core.config import KNOWLEDGE_DIR

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


def _load_faq() -> dict:
    # DB first (survives deploys), file fallback
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            data = load_knowledge_from_db("faq")
            if data:
                return data
    except Exception:
        pass
    faq_path = KNOWLEDGE_DIR / "faq.json"
    with open(faq_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_faq(faq_data: dict):
    # DB first (survives deploys), then file
    from core.database import is_db_available, save_knowledge
    if is_db_available():
        save_knowledge("faq", faq_data)
    faq_path = KNOWLEDGE_DIR / "faq.json"
    with open(faq_path, "w", encoding="utf-8") as f:
        json.dump(faq_data, f, indent=2, ensure_ascii=False)


def _load_catalog() -> list[dict]:
    # DB first (survives deploys), file fallback
    try:
        from core.database import is_db_available, load_knowledge_from_db
        if is_db_available():
            data = load_knowledge_from_db("products")
            if data:
                return data.get("catalog", [])
    except Exception:
        pass
    products_path = KNOWLEDGE_DIR / "products.json"
    with open(products_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("catalog", [])


def _extract_product_names(catalog: list[dict]) -> set[str]:
    """Extract all product name keywords from catalog for matching."""
    names = set()
    for product in catalog:
        # Full name lowercase
        name = product["name"].lower()
        names.add(name)
        # Individual words (skip generic ones)
        skip_words = {
            "gsm", "black", "white", "navy", "grey", "red", "maroon",
            "t-shirt", "tshirt", "premium", "non", "bio", "true",
        }
        for word in name.split():
            cleaned = word.strip("()0123456789")
            if cleaned and len(cleaned) > 2 and cleaned not in skip_words:
                names.add(cleaned)
        # Also add the product ID as a keyword
        names.add(product["id"].lower())
    return names


def _extract_product_identifiers(catalog: list[dict]) -> list[dict]:
    """Build a list of product identifiers for precise matching.

    Each entry has: id, name, category, match_terms (unique terms for this product).
    """
    products = []
    for p in catalog:
        # Build match terms — words unique enough to identify this product
        match_terms = set()
        match_terms.add(p["name"].lower())
        match_terms.add(p["id"].lower())

        # Category-specific keywords
        name_lower = p["name"].lower()
        if "varsity" in name_lower:
            match_terms.add("varsity")
        if "hoodie" in name_lower:
            match_terms.add("hoodie")
        if "sweatshirt" in name_lower:
            match_terms.add("sweatshirt")
        if "polo" in name_lower:
            match_terms.add("polo")
        if "shorts" in name_lower:
            match_terms.add("shorts")
        if "kids" in name_lower:
            match_terms.add("kids")
        if "acid" in name_lower:
            match_terms.add("acid wash")
            match_terms.add("acidwash")
        if "boxy" in name_lower:
            match_terms.add("boxy")
        if "sublimation" in name_lower:
            match_terms.add("sublimation")
        if "zip" in name_lower:
            match_terms.add("zip hoodie")
            match_terms.add("zipper hoodie")
        if "oversize" in name_lower or "oversized" in name_lower:
            match_terms.add("oversized")
            match_terms.add("oversize")
        if "round neck" in name_lower:
            match_terms.add("round neck")
        if "dropshoulder" in name_lower:
            match_terms.add("dropshoulder")
            match_terms.add("drop shoulder")

        products.append({
            "id": p["id"],
            "name": p["name"],
            "category": p.get("category", ""),
            "bulk_price": p.get("bulk_price"),
            "match_terms": match_terms,
        })

    return products


def _faq_mentions_product(faq: dict, product_identifiers: list[dict]) -> list[str]:
    """Check if an FAQ mentions any product. Returns list of matched product IDs."""
    text = (faq.get("question", "") + " " + faq.get("answer", "")).lower()
    keywords = [k.lower() for k in faq.get("keywords", [])]
    matched = []

    for product in product_identifiers:
        for term in product["match_terms"]:
            if term in text or term in keywords:
                matched.append(product["id"])
                break

    return matched


def _check_price_mismatch(faq: dict, catalog: list[dict]) -> list[dict]:
    """Check if FAQ answer has prices that don't match current catalog."""
    answer = faq.get("answer", "")
    mismatches = []

    # Extract "Rs XXX" patterns from FAQ answer
    price_mentions = re.findall(r'rs\s*(\d+)', answer.lower())
    if not price_mentions:
        return []

    faq_prices = {int(p) for p in price_mentions}

    # Check each mentioned product's current price
    text = (faq.get("question", "") + " " + answer).lower()

    for product in catalog:
        name_lower = product["name"].lower()
        # Check if this product is likely mentioned in the FAQ
        product_terms = [name_lower, product["id"].lower()]
        if any(term in text for term in product_terms):
            current_bulk = product.get("bulk_price", 0)
            current_sample = product.get("sample_price", 0)
            current_prices = {current_bulk, current_sample}

            # If FAQ mentions a price that's NOT in current prices
            for faq_price in faq_prices:
                if faq_price not in current_prices and abs(faq_price - current_bulk) > 5:
                    mismatches.append({
                        "product": product["name"],
                        "faq_price": faq_price,
                        "current_bulk_price": current_bulk,
                        "current_sample_price": current_sample,
                    })

    return mismatches


def validate_faqs_against_catalog(catalog: list[dict] | None = None) -> dict:
    """Cross-check all FAQs against current product catalog.

    Detects:
    1. FAQs mentioning products no longer in catalog → mark inactive
    2. FAQs with wrong prices → flag for review

    Returns dict with results and list of flagged FAQs.
    """
    faq_data = _load_faq()
    if catalog is None:
        catalog = _load_catalog()

    catalog_product_ids = {p["id"] for p in catalog}
    product_identifiers = _extract_product_identifiers(catalog)

    # Build set of all product keywords currently in catalog
    active_product_keywords = set()
    for p in product_identifiers:
        active_product_keywords.update(p["match_terms"])

    flagged = []
    price_issues = []
    now = datetime.now(IST).isoformat()

    for i, faq in enumerate(faq_data.get("faqs", [])):
        # Skip already inactive FAQs
        if faq.get("status") == "inactive":
            continue

        # Check which products this FAQ mentions
        mentioned_products = _faq_mentions_product(faq, product_identifiers)

        if mentioned_products:
            # Check if any mentioned product is missing from catalog
            missing = [pid for pid in mentioned_products if pid not in catalog_product_ids]
            if missing:
                faq["status"] = "inactive"
                faq["inactive_reason"] = "product_removed"
                faq["inactive_date"] = now
                faq["missing_products"] = missing
                flagged.append({
                    "index": i,
                    "question": faq["question"],
                    "reason": "product_removed",
                    "missing_products": missing,
                })
                logger.info(
                    f"FAQ deactivated — product removed: '{faq['question']}' "
                    f"(missing: {missing})"
                )

            # Check price mismatches
            mismatches = _check_price_mismatch(faq, catalog)
            if mismatches:
                faq["price_warning"] = True
                faq["price_mismatches"] = mismatches
                faq["price_check_date"] = now
                price_issues.append({
                    "index": i,
                    "question": faq["question"],
                    "mismatches": mismatches,
                })
                logger.info(
                    f"FAQ price mismatch: '{faq['question']}' — "
                    f"{mismatches}"
                )

    # Save updated FAQ data
    if flagged or price_issues:
        _save_faq(faq_data)

        # Auto-persist to GitHub
        from core.git_persist import persist_knowledge_files
        persist_knowledge_files(source="faq-validation")

    result = {
        "total_faqs": len(faq_data.get("faqs", [])),
        "active_faqs": len([f for f in faq_data.get("faqs", []) if f.get("status") != "inactive"]),
        "deactivated": len(flagged),
        "price_warnings": len(price_issues),
        "flagged_faqs": flagged,
        "price_issues": price_issues,
    }

    if flagged:
        logger.info(
            f"FAQ validation: {result['deactivated']} FAQs deactivated, "
            f"{result['price_warnings']} price warnings"
        )

    return result


def detect_contradiction(new_faqs: list[dict]) -> dict:
    """Detect if new FAQs contradict existing ones.

    When Ketu says something different from an existing FAQ,
    the new answer should replace the old one.

    Args:
        new_faqs: List of new FAQ dicts with "question" and "answer" keys.

    Returns dict with replaced FAQs and new FAQs.
    """
    faq_data = _load_faq()
    existing_faqs = faq_data.get("faqs", [])

    replaced = []
    now = datetime.now(IST).isoformat()

    # Contradiction signals in Hindi/English
    negative_signals = {
        "nahi", "nhi", "na", "no", "not", "band", "close", "closed",
        "discontinued", "available nahi", "nahi hai", "nhi hai",
        "band ho gaya", "band kar di", "hataya", "hata diya",
        "out of stock", "stock nahi",
    }
    positive_signals = {
        "haan", "ha", "yes", "available", "hai", "milega", "milta",
        "ready", "stock mein", "in stock",
    }

    for new_faq in new_faqs:
        new_q = new_faq.get("question", "").lower()
        new_a = new_faq.get("answer", "").lower()
        new_keywords = {k.lower() for k in new_faq.get("keywords", [])}

        # Find matching existing FAQ by keyword overlap or similar question
        for i, existing in enumerate(existing_faqs):
            if existing.get("status") == "inactive":
                continue

            old_q = existing.get("question", "").lower()
            old_a = existing.get("answer", "").lower()
            old_keywords = {k.lower() for k in existing.get("keywords", [])}

            # Check keyword overlap
            keyword_overlap = new_keywords & old_keywords
            # Check if questions are about the same topic
            q_words_new = set(new_q.split())
            q_words_old = set(old_q.split())
            question_overlap = q_words_new & q_words_old

            # Need significant overlap to consider them about the same topic
            is_same_topic = (
                len(keyword_overlap) >= 2
                or len(question_overlap) >= 3
                or (len(keyword_overlap) >= 1 and len(question_overlap) >= 2)
            )

            if not is_same_topic:
                continue

            # Check if answers contradict each other
            old_is_positive = any(sig in old_a for sig in positive_signals)
            new_is_negative = any(sig in new_a for sig in negative_signals)
            old_is_negative = any(sig in old_a for sig in negative_signals)
            new_is_positive = any(sig in new_a for sig in positive_signals)

            is_contradiction = (
                (old_is_positive and new_is_negative)
                or (old_is_negative and new_is_positive)
            )

            if is_contradiction:
                # Replace old FAQ with new one
                old_copy = dict(existing)
                existing["status"] = "inactive"
                existing["inactive_reason"] = "contradicted"
                existing["inactive_date"] = now
                existing["replaced_by"] = new_faq.get("question", "")

                replaced.append({
                    "old_question": old_copy["question"],
                    "old_answer": old_copy["answer"],
                    "new_question": new_faq.get("question", ""),
                    "new_answer": new_faq.get("answer", ""),
                    "reason": "Ketu gave contradicting answer",
                })

                logger.info(
                    f"FAQ contradicted and replaced: '{old_copy['question']}' "
                    f"→ new answer from Ketu"
                )

    if replaced:
        _save_faq(faq_data)

        # Auto-persist
        from core.git_persist import persist_knowledge_files
        persist_knowledge_files(source="faq-contradiction-update")

    return {
        "contradictions_found": len(replaced),
        "replaced": replaced,
    }


def get_faq_health_report() -> dict:
    """Get a full health report of all FAQs.

    Returns counts of active, inactive, price-warned FAQs
    plus details of any issues.
    """
    faq_data = _load_faq()
    faqs = faq_data.get("faqs", [])

    active = []
    inactive = []
    price_warned = []

    for faq in faqs:
        if faq.get("status") == "inactive":
            inactive.append({
                "question": faq["question"],
                "reason": faq.get("inactive_reason", "unknown"),
                "date": faq.get("inactive_date", ""),
                "missing_products": faq.get("missing_products", []),
                "replaced_by": faq.get("replaced_by", ""),
            })
        elif faq.get("price_warning"):
            price_warned.append({
                "question": faq["question"],
                "mismatches": faq.get("price_mismatches", []),
                "check_date": faq.get("price_check_date", ""),
            })
        else:
            active.append(faq["question"])

    return {
        "total": len(faqs),
        "active": len(active),
        "inactive": len(inactive),
        "price_warnings": len(price_warned),
        "inactive_faqs": inactive,
        "price_warned_faqs": price_warned,
        "active_faqs": active,
    }


def reactivate_faq(question: str) -> dict:
    """Reactivate a previously deactivated FAQ (owner override).

    Args:
        question: The FAQ question text to reactivate.
    """
    faq_data = _load_faq()

    for faq in faq_data.get("faqs", []):
        if faq.get("question", "").lower() == question.lower():
            if faq.get("status") == "inactive":
                faq.pop("status", None)
                faq.pop("inactive_reason", None)
                faq.pop("inactive_date", None)
                faq.pop("missing_products", None)
                faq.pop("replaced_by", None)
                _save_faq(faq_data)

                from core.git_persist import persist_knowledge_files
                persist_knowledge_files(source="faq-reactivate")

                return {"status": "reactivated", "question": question}

            return {"status": "already_active", "question": question}

    return {"status": "not_found", "question": question}
