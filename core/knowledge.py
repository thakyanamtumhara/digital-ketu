import json
import logging
import time

from core.config import KNOWLEDGE_DIR, LEARNED_DIR

logger = logging.getLogger(__name__)

_cache: dict = {}
_cache_time: float = 0
CACHE_TTL = 60  # seconds


def load_knowledge() -> dict:
    global _cache, _cache_time

    if _cache and (time.time() - _cache_time) < CACHE_TTL:
        return _cache

    knowledge = {}

    # Try DB first, fall back to JSON files
    from core.database import is_db_available, load_all_knowledge_from_db
    if is_db_available():
        db_data = load_all_knowledge_from_db()
        if db_data:
            knowledge = db_data
            logger.debug("Knowledge loaded from DB")

    # Fall back to JSON files (or fill gaps)
    if not knowledge:
        for file in KNOWLEDGE_DIR.glob("*.json"):
            if file.name == "activity_log.json":
                continue
            with open(file, "r", encoding="utf-8") as f:
                knowledge[file.stem] = json.load(f)
        logger.debug("Knowledge loaded from JSON files")

    # Load learned knowledge
    learned_knowledge = _load_learned_knowledge()
    if learned_knowledge:
        knowledge["learned"] = learned_knowledge

    _cache = knowledge
    _cache_time = time.time()
    return knowledge


def _load_learned_knowledge() -> list[dict]:
    """Load learned knowledge from DB first, then JSON files."""
    learned = []

    # Try DB
    from core.database import is_db_available, list_learned_files, load_learned_file
    if is_db_available():
        files = list_learned_files()
        for f in files:
            fname = f["file"]
            if fname.startswith("_") or not fname.endswith(".json"):
                continue
            content = load_learned_file(fname)
            if content:
                try:
                    learned.append(json.loads(content))
                except Exception:
                    pass
        if learned:
            return learned

    # Fall back to JSON files
    if not LEARNED_DIR.exists():
        return []

    for file in LEARNED_DIR.glob("*.json"):
        if file.name.startswith("_"):
            continue
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                learned.append(data)
        except Exception as e:
            logger.warning(f"Failed to load learned file {file}: {e}")

    return learned


def invalidate_cache():
    global _cache, _cache_time
    _cache = {}
    _cache_time = 0


def format_context() -> str:
    knowledge = load_knowledge()
    sections = []

    # Company info
    if "company" in knowledge:
        c = knowledge["company"]
        contact = c.get("contact", {})
        websites = c.get("websites", {})
        sections.append(f"""## Company Info
- Company: {c['name']} (Brand: {c['brand']})
- Type: {c.get('business_type', 'B2B Wholesale Manufacturer')}
- Website: {websites.get('primary', '')} / {websites.get('secondary', '')}
- Catalog: {websites.get('catalog', '')}
- WhatsApp: {contact.get('whatsapp', '')}
- Phone: {contact.get('phone', '')}
- Factory: {c['locations']['factory']['city']}, {c['locations']['factory']['state']} — {c['locations']['factory']['description']}
- Warehouse: {c['locations']['warehouse']['city']} ({c['locations']['warehouse']['area']})
- USP: {', '.join(c['usp'])}""")

    # Products — full catalog with details
    if "products" in knowledge:
        p = knowledge["products"]

        # Price note
        price_note = p.get("price_note", "")
        if price_note:
            sections.append(f"## Pricing Note: {price_note}")

        product_lines = []
        for item in p.get("catalog", []):
            colors = ", ".join(item.get("colors", []))
            sizes = ", ".join(item.get("sizes", []))
            if "bulk_price" in item:
                price_str = f"Rs {item['bulk_price']}/pc (bulk) | Rs {item['sample_price']}/pc (sample)"
            else:
                price_str = item.get("price_range", "N/A")
            category = item.get("category", "")
            cat_str = f" [{category}]" if category else ""
            product_lines.append(
                f"### {item['name']} ({item['gsm']} GSM){cat_str}\n"
                f"  Price: {price_str}\n"
                f"  Fabric: {item.get('fabric', 'N/A')}\n"
                f"  MOQ: {item.get('moq', 10)} pcs | Sizes: {sizes}\n"
                f"  Colors ({len(item.get('colors', []))}): {colors}"
            )
        sections.append("## Products (21 items)\n" + "\n\n".join(product_lines))

        if "bulk_discounts" in p:
            d = p["bulk_discounts"]
            discount_lines = [f"- {k}: {v}" for k, v in d.items()]
            sections.append("## Discounts\n" + "\n".join(discount_lines))

        if "printing_compatibility" in p:
            print_lines = [f"- {k}: {v}" for k, v in p["printing_compatibility"].items()]
            sections.append("## Printing Compatibility\n" + "\n".join(print_lines))

        sections.append(f"## Stock: {p.get('ready_stock', 'N/A')}")
        sections.append(f"## Recent Sales: {p.get('total_sold', 'N/A')}")

        if "gsm_guide" in p:
            gsm_lines = [f"- {gsm} GSM: {desc}" for gsm, desc in p["gsm_guide"].items()]
            sections.append("## GSM Guide\n" + "\n".join(gsm_lines))

        if "fabric_features" in p:
            fabric_lines = [f"- {k}: {v}" for k, v in p["fabric_features"].items()]
            sections.append("## Fabric Features\n" + "\n".join(fabric_lines))

    # Payment & Shipping
    if "company" in knowledge:
        c = knowledge["company"]
        if "payment_terms" in c:
            pt = c["payment_terms"]
            sections.append(f"""## Payment Terms
- Policy: {pt.get('policy', '100% Prepaid')}
- Modes: {', '.join(pt.get('modes', []))}
- Website discount: {pt.get('website_discount', '')}""")

        if "shipping" in c:
            s = c["shipping"]
            delivery = s.get("delivery_time", {})
            sections.append(f"""## Shipping
- Speed: {s.get('dispatch_speed', 'Dispatch within minutes')}
- Delhi NCR: {delivery.get('delhi_ncr', '1-2 hours')}
- PAN India: {delivery.get('pan_india', '1-3 days')}""")

        if "dropshipping" in c:
            ds = c["dropshipping"]
            sections.append(f"""## Dropshipping
- {ds.get('type', 'Zero-contact blind dropshipping')}
- {ds.get('description', '')}
- Setup fee: {ds.get('setup_fee', 'None')}""")

        if "gst" in c:
            sections.append(f"## GST: {c['gst'].get('rate', '5%')} — {c['gst'].get('note', '')}")

    # Active FAQs only (skip inactive/outdated ones)
    if "faq" in knowledge:
        faq_lines = []
        for faq in knowledge["faq"].get("faqs", []):
            if faq.get("status") == "inactive":
                continue  # Skip deactivated FAQs
            faq_lines.append(f"Q: {faq['question']}\nA: {faq['answer']}")
        sections.append("## Common Q&A\n" + "\n\n".join(faq_lines))

    # Style rules
    if "style" in knowledge:
        s = knowledge["style"]
        sections.append("## Reply Style Rules\n" + "\n".join(f"- {r}" for r in s.get("rules", [])))
        sections.append("## Avoid\n" + "\n".join(f"- {a}" for a in s.get("avoid", [])))

        # ALL example conversations
        examples = s.get("example_conversations", [])
        if examples:
            ex_lines = []
            for ex in examples:
                ex_lines.append(f"Customer: {ex['customer']}\nKetu: {ex['reply']}")
            sections.append("## Example Conversations\n" + "\n\n".join(ex_lines))

        # Include learned patterns if any
        learned_patterns = s.get("learned_patterns", [])
        if learned_patterns:
            sections.append("## Learned Style Patterns\n" + "\n".join(
                f"- {p}" if isinstance(p, str) else f"- {json.dumps(p, ensure_ascii=False)}"
                for p in learned_patterns
            ))

    # Learned knowledge from YouTube videos
    learned = knowledge.get("learned", [])
    if learned:
        learned_lines = []
        for item in learned:
            title = item.get("title", "Unknown video")
            k = item.get("knowledge", {})

            key_points = k.get("key_points", [])
            pricing = k.get("pricing", [])

            parts = [f"**{title}**:"]
            if key_points:
                for point in key_points[:3]:
                    parts.append(f"  - {point}" if isinstance(point, str) else f"  - {json.dumps(point, ensure_ascii=False)}")
            if pricing:
                for price in (pricing if isinstance(pricing, list) else [pricing]):
                    parts.append(f"  - Pricing: {price}" if isinstance(price, str) else f"  - Pricing: {json.dumps(price, ensure_ascii=False)}")

            learned_lines.append("\n".join(parts))

        if learned_lines:
            sections.append("## Knowledge from YouTube Videos\n" + "\n\n".join(learned_lines))

    return "\n\n".join(sections)
