"""Catalog repo syncer — fetches latest product data from GitHub catalog repo.

The catalog repo (github.com/thakyanamtumhara/catalog) is the single source of truth
for product pricing, colors, sizes, and descriptions. It also has llms-full.txt
with comprehensive business info, FAQs, size charts, etc.

This module fetches products.json from the repo and updates Digital Ketu's
knowledge/products.json with the latest data.
"""

import json
import logging
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"

# Raw GitHub URLs for catalog data
CATALOG_PRODUCTS_URL = (
    "https://raw.githubusercontent.com/thakyanamtumhara/catalog/master/products.json"
)
CATALOG_LLMS_URL = (
    "https://raw.githubusercontent.com/thakyanamtumhara/catalog/master/llms-full.txt"
)


async def fetch_catalog_products() -> dict | None:
    """Fetch products.json from the catalog GitHub repo."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(CATALOG_PRODUCTS_URL)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"Failed to fetch catalog products: {e}")
        return None


async def fetch_catalog_llms_text() -> str | None:
    """Fetch llms-full.txt from the catalog GitHub repo."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(CATALOG_LLMS_URL)
            response.raise_for_status()
            return response.text
    except Exception as e:
        logger.error(f"Failed to fetch catalog llms text: {e}")
        return None


def convert_catalog_to_knowledge(catalog_data: dict) -> dict:
    """Convert catalog repo products.json format to Digital Ketu knowledge format."""
    products = []

    for category in catalog_data.get("categories", []):
        for product in category.get("products", []):
            products.append({
                "id": product["slug"],
                "name": product["name"],
                "category": category["name"],
                "gsm": product["gsm"],
                "description": product["description"],
                "fabric": product["description"].split(",", 1)[-1].strip() if "," in product["description"] else product["description"],
                "sizes": product["sizes"],
                "colors": product["colors"],
                "bulk_price": product["bulkPrice"],
                "sample_price": product["samplePrice"],
                "price_range": f"Rs {product['bulkPrice']}/pc (bulk) | Rs {product['samplePrice']}/pc (sample)",
                "moq": catalog_data.get("moq", 10),
                "weight_kg": product.get("weightKg"),
                "catalog_url": product.get("catalogUrl", ""),
                "image_url": product.get("imageBaseUrl", ""),
            })

    return {
        "catalog": products,
        "last_updated": catalog_data.get("lastUpdated", ""),
        "currency": catalog_data.get("currency", "INR"),
        "gst_rate": catalog_data.get("gstRate", 5),
        "moq": catalog_data.get("moq", 10),
        "website_discount": f"Extra Rs {catalog_data.get('websiteDiscount', 2)}/pc discount on bulkplaintshirt.com",
        "payment_terms": catalog_data.get("paymentTerms", "100% Prepaid"),
        "contact": catalog_data.get("contact", {}),
        "bulk_discounts": {
            "website": f"Rs {catalog_data.get('websiteDiscount', 2)}/pc discount on website orders",
            "1000_plus": "Special rates available — contact on WhatsApp",
        },
        "ready_stock": "Massive ready-to-dispatch inventory — dispatch within minutes",
        "total_sold": "Selling lakhs of units monthly",
        "gsm_guide": {
            "180": "Standard weight — regular t-shirts, everyday wear, breathable",
            "200": "Standard+ — sublimation printing specifically",
            "210": "Mid-weight — premium oversized tees, like H&M/Zara quality",
            "220": "Mid-weight — structured polo fabric, honeycomb texture",
            "240": "Heavyweight — thick, structured drape, gold standard for streetwear",
            "320": "Heavy fleece — warm, soft brushed inside, hoodies/sweatshirts",
            "430": "Ultra-heavy — thickest blank hoodie in India, extreme cold",
        },
        "fabric_features": {
            "biowash": "Enzyme-washed for softness and pre-shrunk finish",
            "supercombed": "Premium combed cotton — smoother, stronger yarn",
            "drop_shoulder": "Oversized relaxed fit style popular in streetwear",
            "loopknit": "Knitting method that creates loops inside for softness",
            "brushed": "Fabric brushed on inside for fleece-like warmth",
        },
    }


async def sync_catalog() -> dict:
    """Full sync: fetch catalog from GitHub, update local knowledge files.

    Returns dict with status and details of what was updated.
    """
    catalog_data = await fetch_catalog_products()
    if not catalog_data:
        return {"status": "error", "detail": "Could not fetch catalog from GitHub"}

    # Convert to knowledge format
    knowledge_products = convert_catalog_to_knowledge(catalog_data)

    # Count products
    product_count = len(knowledge_products["catalog"])

    # Save to products.json
    products_path = KNOWLEDGE_DIR / "products.json"
    with open(products_path, "w", encoding="utf-8") as f:
        json.dump(knowledge_products, f, indent=2, ensure_ascii=False)

    # Also fetch and save llms-full.txt for reference
    llms_text = await fetch_catalog_llms_text()
    if llms_text:
        llms_path = KNOWLEDGE_DIR / "learned" / "catalog_llms_full.txt"
        llms_path.parent.mkdir(exist_ok=True)
        with open(llms_path, "w", encoding="utf-8") as f:
            f.write(llms_text)

    logger.info(f"Catalog synced: {product_count} products updated from GitHub repo")

    return {
        "status": "ok",
        "products_synced": product_count,
        "last_updated": catalog_data.get("lastUpdated", "unknown"),
        "categories": [cat["name"] for cat in catalog_data.get("categories", [])],
    }
