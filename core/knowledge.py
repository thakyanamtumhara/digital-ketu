import json
import os
import time
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

_cache: dict = {}
_cache_time: float = 0
CACHE_TTL = 60  # seconds


def load_knowledge() -> dict:
    global _cache, _cache_time

    if _cache and (time.time() - _cache_time) < CACHE_TTL:
        return _cache

    knowledge = {}
    for file in KNOWLEDGE_DIR.glob("*.json"):
        with open(file, "r", encoding="utf-8") as f:
            knowledge[file.stem] = json.load(f)

    _cache = knowledge
    _cache_time = time.time()
    return knowledge


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
        sections.append(f"""## Company Info
- Company: {c['name']} (Brand: {c['brand']})
- Tagline: {c['tagline']}
- Website: {c['websites']['primary']} / {c['websites']['secondary']}
- Email: {c['email']}
- Factory: {c['locations']['factory']['city']}, {c['locations']['factory']['state']} — {c['locations']['factory']['description']}
- Warehouse: {c['locations']['warehouse']['city']} ({c['locations']['warehouse']['area']})
- USP: {', '.join(c['usp'])}""")

    # Products
    if "products" in knowledge:
        p = knowledge["products"]
        product_lines = []
        for item in p.get("catalog", []):
            product_lines.append(
                f"- {item['name']} ({item['gsm']} GSM): {item['price_range']}, "
                f"MOQ {item['moq']} pcs, {len(item['colors'])} colors available"
            )
        sections.append("## Products\n" + "\n".join(product_lines))

        if "bulk_discounts" in p:
            d = p["bulk_discounts"]
            sections.append(f"""## Bulk Discounts
- 500+ pieces: {d['500_plus']}
- Online purchase: {d['online_purchase']}
- 1000+ pieces: {d['1000_plus']}""")

        sections.append(f"## Stock: {p.get('ready_stock', 'N/A')}")
        sections.append(f"## Recent Sales: {p.get('total_sold', 'N/A')}")

        if "gsm_guide" in p:
            gsm_lines = [f"- {gsm} GSM: {desc}" for gsm, desc in p["gsm_guide"].items()]
            sections.append("## GSM Guide\n" + "\n".join(gsm_lines))

    # Payment & Shipping
    if "company" in knowledge:
        c = knowledge["company"]
        if "payment_terms" in c:
            pt = c["payment_terms"]
            sections.append(f"""## Payment Terms
- First order: {pt['first_order']}
- Repeat orders: {pt['repeat_orders']}
- Online discount: {pt['online_discount']}""")

        if "shipping" in c:
            s = c["shipping"]
            delivery = s.get("delivery_time", {})
            sections.append(f"""## Shipping
- Dispatch: {s.get('dispatch_time', 'N/A')}
- Delhi NCR: {delivery.get('delhi_ncr', 'N/A')}
- North India: {delivery.get('north_india', 'N/A')}
- South India: {delivery.get('south_india', 'N/A')}
- Export: Available via courier and sea transport""")

    # FAQ
    if "faq" in knowledge:
        faq_lines = []
        for faq in knowledge["faq"].get("faqs", [])[:10]:
            faq_lines.append(f"Q: {faq['question']}\nA: {faq['answer']}")
        sections.append("## Common Q&A\n" + "\n\n".join(faq_lines))

    # Style rules
    if "style" in knowledge:
        s = knowledge["style"]
        sections.append("## Reply Style Rules\n" + "\n".join(f"- {r}" for r in s.get("rules", [])))
        sections.append("## Avoid\n" + "\n".join(f"- {a}" for a in s.get("avoid", [])))

        examples = s.get("example_conversations", [])
        if examples:
            ex_lines = []
            for ex in examples[:4]:
                ex_lines.append(f"Customer: {ex['customer']}\nKetu: {ex['reply']}")
            sections.append("## Example Conversations\n" + "\n\n".join(ex_lines))

    return "\n\n".join(sections)
