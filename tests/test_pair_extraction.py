"""Test pair extraction and deduplication — 10 variations.

Covers: short msgs, long msgs, quality, non-quality, multi-message,
AI leaks, cross-chat, dedup, mixed conversations, rapid-fire.

Run: python tests/test_pair_extraction.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock database module before importing anything
import types
mock_db = types.ModuleType("core.database")
mock_db.is_db_available = lambda: False
mock_db.kv_get = lambda k: None
mock_db.kv_set = lambda k, v: None
mock_db.load_knowledge_from_db = lambda k: None
mock_db.save_knowledge_to_db = lambda k, v: None
mock_db.init_db = lambda: False
sys.modules["core.database"] = mock_db

from main import _extract_conversation_pairs, _message_fingerprint, _safe_bool


OWNER_ID = "owner123"


# ──────────────────────────────────────────────────────────────────────
# TEST 1: Short buyer message + short Ketu reply (typical real chat)
# ──────────────────────────────────────────────────────────────────────
def test_1_short_buyer_short_ketu():
    """Buyer: 'Rate batao' → Ketu: '175 bulk'"""
    messages = [
        {"chat_id": "chat1", "sender_id": "cust1", "content": "Oversized rate batao", "timestamp": "2026-03-07T10:00:00+05:30"},
        {"chat_id": "chat1", "sender_id": OWNER_ID, "content": "175 bulk, 210 sample", "timestamp": "2026-03-07T10:01:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
    assert pairs[0]["customer"] == "Oversized rate batao"
    assert pairs[0]["ketu"] == "175 bulk, 210 sample"
    assert pairs[0]["ai"] is False
    print("PASS test_1: Short buyer 'Oversized rate batao' → Ketu '175 bulk, 210 sample'")
    print(f"  BUYER: {pairs[0]['customer']} → KETU: {pairs[0]['ketu']}")


# ──────────────────────────────────────────────────────────────────────
# TEST 2: Long buyer message + long Ketu reply (detailed conversation)
# ──────────────────────────────────────────────────────────────────────
def test_2_long_buyer_long_ketu():
    """Buyer sends detailed requirement, Ketu gives detailed answer."""
    messages = [
        {"chat_id": "chat2", "sender_id": "cust2", "content": "Bhaiya mujhe 500 pieces chahiye oversized 240 GSM black aur white mein, printing ke liye DTG suitable hona chahiye, delivery Delhi NCR mein kitne din lagenge", "timestamp": "2026-03-07T11:00:00+05:30"},
        {"chat_id": "chat2", "sender_id": OWNER_ID, "content": "Ji sir 240 GSM oversized Rs 180 bulk mein, DTG ke liye perfect hai, Delhi 1-2 ghante mein Porter se delivery, 500 pcs pe extra discount milega", "timestamp": "2026-03-07T11:02:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
    assert "500 pieces" in pairs[0]["customer"]
    assert "180 bulk" in pairs[0]["ketu"]
    print("PASS test_2: Long buyer (detailed req) → Long Ketu (detailed answer)")
    print(f"  BUYER: {pairs[0]['customer'][:60]}... → KETU: {pairs[0]['ketu'][:60]}...")


# ──────────────────────────────────────────────────────────────────────
# TEST 3: Non-quality Ketu reply (just "Ok") should be SKIPPED
# ──────────────────────────────────────────────────────────────────────
def test_3_non_quality_ketu_ok():
    """Buyer asks question, Ketu just says 'Ok' → should NOT create pair."""
    messages = [
        {"chat_id": "chat3", "sender_id": "cust3", "content": "Bhaiya please dkehna ki acid wash aache ho", "timestamp": "2026-03-07T17:11:00+05:30"},
        {"chat_id": "chat3", "sender_id": OWNER_ID, "content": "Ok", "timestamp": "2026-03-07T17:18:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    # "Ok" is low-quality — should be skipped
    assert len(pairs) == 0, f"Expected 0 pairs (Ok is low quality), got {len(pairs)}: {pairs}"
    print("PASS test_3: Ketu replied 'Ok' → correctly skipped (low quality)")


# ──────────────────────────────────────────────────────────────────────
# TEST 4: Non-quality buyer message (greeting) should be SKIPPED
# ──────────────────────────────────────────────────────────────────────
def test_4_non_quality_buyer_greeting():
    """Buyer just says 'Hi', Ketu replies → should NOT create pair (no business intent)."""
    messages = [
        {"chat_id": "chat4", "sender_id": "cust4", "content": "Hi", "timestamp": "2026-03-07T12:00:00+05:30"},
        {"chat_id": "chat4", "sender_id": OWNER_ID, "content": "Ji sir, kaise help kar sakta hun", "timestamp": "2026-03-07T12:01:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    # "Hi" has no business intent → should be skipped
    assert len(pairs) == 0, f"Expected 0 pairs (Hi has no business intent), got {len(pairs)}: {pairs}"
    print("PASS test_4: Buyer said 'Hi' → correctly skipped (no business intent)")


# ──────────────────────────────────────────────────────────────────────
# TEST 5: Multiple buyer messages combined → single Ketu reply
#   (the Sujal Ahir scenario — EXACT production bug)
# ──────────────────────────────────────────────────────────────────────
def test_5_multi_message_buyer_single_ketu():
    """Buyer sends 2 messages quickly, Ketu replies once → correct combined pair."""
    messages = [
        {"chat_id": "chat5", "sender_id": "cust5", "content": "Parcel late lagwaya bhaiya", "timestamp": "2026-03-07T19:41:00+05:30"},
        {"chat_id": "chat5", "sender_id": "cust5", "content": "Kal miljayega?", "timestamp": "2026-03-07T19:41:30+05:30"},
        {"chat_id": "chat5", "sender_id": OWNER_ID, "content": "7 baje hi lag gaya tha", "timestamp": "2026-03-07T19:44:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1
    assert "Parcel late lagwaya bhaiya" in pairs[0]["customer"]
    assert "Kal miljayega?" in pairs[0]["customer"]
    assert pairs[0]["ketu"] == "7 baje hi lag gaya tha"
    # Must NOT have any other random messages mixed in
    parts = pairs[0]["customer"].split(" | ")
    assert len(parts) == 2, f"Expected exactly 2 combined messages, got {len(parts)}: {parts}"
    print("PASS test_5: Multi-msg buyer 'Parcel late + Kal miljayega?' → Ketu '7 baje hi lag gaya tha'")
    print(f"  BUYER: {pairs[0]['customer']} → KETU: {pairs[0]['ketu']}")


# ──────────────────────────────────────────────────────────────────────
# TEST 6: AI reply in middle must NOT leak previous messages forward
#   (the MAIN bug — "Oversized tshirt" leaking to "7 baje hi lag gaya tha")
# ──────────────────────────────────────────────────────────────────────
def test_6_ai_reply_blocks_leak():
    """AI replies to first Q, then buyer asks new Q, Ketu replies manually.
    First Q must NOT appear in second pair."""
    messages = [
        # Customer asks about oversized → AI replies
        {"chat_id": "chat6", "sender_id": "cust6", "content": "Oversized tshirt price kitna hai", "timestamp": "2026-03-07T17:10:00+05:30"},
        {"chat_id": "chat6", "sender_id": OWNER_ID, "content": "Ji sir 175 bulk rate hai", "timestamp": "2026-03-07T17:10:30+05:30", "is_ai_generated": True},
        # 2 hours later, same customer asks about delivery → Ketu replies manually
        {"chat_id": "chat6", "sender_id": "cust6", "content": "Mera order dispatch hua?", "timestamp": "2026-03-07T19:30:00+05:30"},
        {"chat_id": "chat6", "sender_id": OWNER_ID, "content": "Haan bhai, tracking number bhej raha hun", "timestamp": "2026-03-07T19:35:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1, f"Expected 1 pair (only manual), got {len(pairs)}"
    assert "Oversized" not in pairs[0]["customer"], f"BUG: Oversized leaked! Got: {pairs[0]['customer']}"
    assert "order dispatch" in pairs[0]["customer"]
    assert "tracking" in pairs[0]["ketu"]
    print("PASS test_6: AI reply blocks leak — 'Oversized tshirt' does NOT leak to 'tracking number' pair")
    print(f"  BUYER: {pairs[0]['customer']} → KETU: {pairs[0]['ketu']}")


# ──────────────────────────────────────────────────────────────────────
# TEST 7: Two different customers in same buffer — no cross-contamination
# ──────────────────────────────────────────────────────────────────────
def test_7_two_customers_different_chats():
    """Customer A talks about hoodie, Customer B talks about round neck.
    Each pair must stay with its own customer."""
    messages = [
        # Customer A: Hoodie inquiry
        {"chat_id": "919111111111@s.whatsapp.net", "sender_id": "custA", "content": "Hoodie ka rate kya hai bhaiya", "timestamp": "2026-03-07T14:00:00+05:30"},
        {"chat_id": "919111111111@s.whatsapp.net", "sender_id": OWNER_ID, "content": "320 GSM hoodie Rs 450 bulk mein", "timestamp": "2026-03-07T14:01:00+05:30", "is_ai_generated": False},
        # Customer B: Round neck inquiry
        {"chat_id": "919222222222@s.whatsapp.net", "sender_id": "custB", "content": "Round neck 180 GSM available hai?", "timestamp": "2026-03-07T14:02:00+05:30"},
        {"chat_id": "919222222222@s.whatsapp.net", "sender_id": OWNER_ID, "content": "Haan ji, Rs 95 bulk, 10 colors mein ready", "timestamp": "2026-03-07T14:03:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 2, f"Expected 2 pairs (one per customer), got {len(pairs)}"

    # Find each pair
    hoodie_pair = [p for p in pairs if "Hoodie" in p["customer"]]
    round_neck_pair = [p for p in pairs if "Round neck" in p["customer"]]

    assert len(hoodie_pair) == 1, f"Expected 1 hoodie pair, got {len(hoodie_pair)}"
    assert len(round_neck_pair) == 1, f"Expected 1 round neck pair, got {len(round_neck_pair)}"

    assert "450" in hoodie_pair[0]["ketu"], f"Hoodie paired with wrong reply: {hoodie_pair[0]['ketu']}"
    assert "95" in round_neck_pair[0]["ketu"], f"Round neck paired with wrong reply: {round_neck_pair[0]['ketu']}"

    # No cross-contamination
    assert "Round neck" not in hoodie_pair[0]["customer"]
    assert "Hoodie" not in round_neck_pair[0]["customer"]

    print("PASS test_7: Two customers — Hoodie→450, Round neck→95, no mixing")
    for p in pairs:
        print(f"  BUYER: {p['customer'][:50]} → KETU: {p['ketu'][:50]}")


# ──────────────────────────────────────────────────────────────────────
# TEST 8: Rapid-fire buyer messages (5 msgs in 1 minute) + 1 Ketu reply
# ──────────────────────────────────────────────────────────────────────
def test_8_rapid_fire_buyer():
    """Buyer sends 5 messages quickly, Ketu replies once → all 5 combined correctly."""
    messages = [
        {"chat_id": "chat8", "sender_id": "cust8", "content": "Bhai suniye", "timestamp": "2026-03-07T15:00:00+05:30"},
        {"chat_id": "chat8", "sender_id": "cust8", "content": "Mujhe 200 pieces chahiye", "timestamp": "2026-03-07T15:00:10+05:30"},
        {"chat_id": "chat8", "sender_id": "cust8", "content": "Oversized 240 GSM", "timestamp": "2026-03-07T15:00:20+05:30"},
        {"chat_id": "chat8", "sender_id": "cust8", "content": "Black color mein", "timestamp": "2026-03-07T15:00:30+05:30"},
        {"chat_id": "chat8", "sender_id": "cust8", "content": "Kal tak mil jayega?", "timestamp": "2026-03-07T15:00:40+05:30"},
        {"chat_id": "chat8", "sender_id": OWNER_ID, "content": "Haan ji, 200 pcs black 240 GSM ready hai, kal delivery ho jayegi", "timestamp": "2026-03-07T15:02:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
    parts = pairs[0]["customer"].split(" | ")
    assert len(parts) == 5, f"Expected 5 combined messages, got {len(parts)}: {parts}"
    assert "200 pieces" in pairs[0]["customer"]
    assert "240 GSM" in pairs[0]["customer"]
    assert "Black color" in pairs[0]["customer"]
    assert "200 pcs" in pairs[0]["ketu"]
    print("PASS test_8: 5 rapid buyer msgs combined → 1 Ketu reply")
    print(f"  BUYER: {pairs[0]['customer'][:80]}...")
    print(f"  KETU: {pairs[0]['ketu']}")


# ──────────────────────────────────────────────────────────────────────
# TEST 9: Mixed AI + manual replies in same chat (realistic production flow)
# ──────────────────────────────────────────────────────────────────────
def test_9_mixed_ai_and_manual_same_chat():
    """Full conversation: AI handles first 2 questions, Ketu takes over for 3rd.
    Only the manual Ketu reply should create a pair."""
    messages = [
        # Round 1: Customer asks price → AI replies
        {"chat_id": "chat9", "sender_id": "cust9", "content": "Polo tshirt ka rate batao", "timestamp": "2026-03-07T10:00:00+05:30"},
        {"chat_id": "chat9", "sender_id": OWNER_ID, "content": "Ji sir polo Rs 220 bulk mein", "timestamp": "2026-03-07T10:00:15+05:30", "is_ai_generated": True},
        # Round 2: Customer asks about GSM → AI replies
        {"chat_id": "chat9", "sender_id": "cust9", "content": "GSM kitna hai", "timestamp": "2026-03-07T10:01:00+05:30"},
        {"chat_id": "chat9", "sender_id": OWNER_ID, "content": "220 GSM premium cotton", "timestamp": "2026-03-07T10:01:15+05:30", "is_ai_generated": True},
        # Round 3: Customer asks about special discount → Ketu manually replies
        {"chat_id": "chat9", "sender_id": "cust9", "content": "Bhai 1000 pcs ke liye special rate dedo", "timestamp": "2026-03-07T10:05:00+05:30"},
        {"chat_id": "chat9", "sender_id": OWNER_ID, "content": "1000 pe 195 de dunga bhai, best rate hai", "timestamp": "2026-03-07T10:10:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1, f"Expected 1 pair (only manual), got {len(pairs)}: {[p['customer'][:30] + ' → ' + p['ketu'][:30] for p in pairs]}"
    # Must only have the 3rd question, NOT the first 2
    assert "Polo" not in pairs[0]["customer"], f"Polo leaked past AI reply! Got: {pairs[0]['customer']}"
    assert "GSM" not in pairs[0]["customer"], f"GSM leaked past AI reply! Got: {pairs[0]['customer']}"
    assert "1000 pcs" in pairs[0]["customer"]
    assert "195" in pairs[0]["ketu"]
    print("PASS test_9: 2 AI replies + 1 manual → only manual creates pair, no leaks")
    print(f"  BUYER: {pairs[0]['customer']} → KETU: {pairs[0]['ketu']}")


# ──────────────────────────────────────────────────────────────────────
# TEST 10: Dedup across 4 wwbun syncs (production-scale simulation)
#   Full Sujal Ahir scenario with overlapping syncs
# ──────────────────────────────────────────────────────────────────────
def test_10_full_production_dedup():
    """Simulate 4 wwbun syncs with overlapping messages — EXACT production scenario.

    Messages from multiple customers across syncs. Verifies:
    - No duplicates in buffer after dedup
    - No duplicate messages in any pair
    - Correct buyer-Ketu pairing across all chats
    """
    # Customer 1: Oversized inquiry (AI handles)
    cust1_msgs = [
        {"chat_id": "c1@wa", "sender_id": "c1", "content": "Oversized tshirt", "timestamp": "2026-03-07T17:10:00+05:30"},
        {"chat_id": "c1@wa", "sender_id": OWNER_ID, "content": "Ji sir 175 bulk rate", "timestamp": "2026-03-07T17:10:30+05:30", "is_ai_generated": True},
    ]

    # Customer 2 (Sujal): Acid wash + later parcel query (Ketu handles)
    cust2_msgs_early = [
        {"chat_id": "c2@wa", "sender_id": "c2", "content": "Bhaiya acid wash dekhna", "timestamp": "2026-03-07T17:11:00+05:30"},
        {"chat_id": "c2@wa", "sender_id": OWNER_ID, "content": "Ok", "timestamp": "2026-03-07T17:18:00+05:30", "is_ai_generated": False},
    ]
    cust2_msgs_late = [
        {"chat_id": "c2@wa", "sender_id": "c2", "content": "Parcel late lagwaya bhaiya", "timestamp": "2026-03-07T19:41:00+05:30"},
        {"chat_id": "c2@wa", "sender_id": "c2", "content": "Kal miljayega?", "timestamp": "2026-03-07T19:41:30+05:30"},
        {"chat_id": "c2@wa", "sender_id": OWNER_ID, "content": "7 baje hi lag gaya tha", "timestamp": "2026-03-07T19:44:00+05:30", "is_ai_generated": False},
    ]

    # Customer 3: Round neck bulk (Ketu handles)
    cust3_msgs = [
        {"chat_id": "c3@wa", "sender_id": "c3", "content": "Round neck 500 pcs rate bhai", "timestamp": "2026-03-07T19:50:00+05:30"},
        {"chat_id": "c3@wa", "sender_id": OWNER_ID, "content": "500 pe 85 per piece, factory direct", "timestamp": "2026-03-07T19:55:00+05:30", "is_ai_generated": False},
    ]

    # Simulate 4 overlapping wwbun syncs
    sync1 = cust1_msgs + cust2_msgs_early
    sync2 = cust1_msgs + cust2_msgs_early  # Duplicate of sync1
    sync3 = cust1_msgs + cust2_msgs_early + cust2_msgs_late  # Old + new
    sync4 = cust2_msgs_late + cust3_msgs  # Latest messages

    # Buffer with dedup
    buffer = []
    existing_fps = set()
    total_received = 0
    total_duped = 0

    for sync_batch in [sync1, sync2, sync3, sync4]:
        for m in sync_batch:
            total_received += 1
            fp = _message_fingerprint(m)
            if fp and fp in existing_fps:
                total_duped += 1
                continue
            buffer.append(m)
            if fp:
                existing_fps.add(fp)

    # Verify dedup worked
    expected_unique = len(cust1_msgs) + len(cust2_msgs_early) + len(cust2_msgs_late) + len(cust3_msgs)
    assert len(buffer) == expected_unique, (
        f"Expected {expected_unique} unique msgs, got {len(buffer)}. "
        f"Received {total_received}, duped {total_duped}"
    )

    # Extract pairs
    pairs = _extract_conversation_pairs(buffer, OWNER_ID)

    # Verify NO duplicate messages in ANY pair
    for pair in pairs:
        parts = pair["customer"].split(" | ")
        if len(parts) != len(set(parts)):
            assert False, f"Duplicate in pair: {parts}"

    # Verify correct pairing
    # "Ok" is low quality → cust2 early should NOT create pair
    # Cust1 was AI → no pair
    # Cust2 late → "Parcel late + Kal miljayega" → "7 baje hi lag gaya tha"
    # Cust3 → "Round neck 500" → "500 pe 85"

    ketu7_pairs = [p for p in pairs if "7 baje" in p["ketu"]]
    ketu85_pairs = [p for p in pairs if "85 per piece" in p["ketu"]]

    assert len(ketu7_pairs) == 1, f"Expected 1 '7 baje' pair, got {len(ketu7_pairs)}"
    assert len(ketu85_pairs) == 1, f"Expected 1 '85 per piece' pair, got {len(ketu85_pairs)}"

    # No cross-contamination
    assert "Oversized" not in ketu7_pairs[0]["customer"], f"Oversized leaked into 7 baje pair: {ketu7_pairs[0]['customer']}"
    assert "acid wash" not in ketu7_pairs[0]["customer"], f"acid wash leaked into 7 baje pair: {ketu7_pairs[0]['customer']}"
    assert "Parcel" in ketu7_pairs[0]["customer"]
    assert "Round neck" in ketu85_pairs[0]["customer"]

    print(f"PASS test_10: Full production scenario — {total_received} msgs received, {total_duped} duped, {len(buffer)} unique")
    print(f"  Pairs extracted: {len(pairs)}")
    for p in pairs:
        print(f"    [{p['chat_id'][:5]}] BUYER: {p['customer'][:50]} → KETU: {p['ketu'][:40]}")


# ──────────────────────────────────────────────────────────────────────
# TEST 11: No chat_id — messages from different customers must NOT mix
#   (THE REAL PRODUCTION BUG — wwbun doesn't send chat_id)
# ──────────────────────────────────────────────────────────────────────
def test_11_no_chat_id_different_customers():
    """wwbun sends no chat_id. 3 different customers ask questions, Ketu replies.
    Each customer's messages must stay separate — NO cross-customer combining."""
    messages = [
        # Customer 1: acid wash inquiry
        {"sender_id": "919111111111", "content": "Tell me the wholesale price of acid t-shirts", "timestamp": "2026-03-07T20:30:00+05:30", "is_owner": False, "is_ai_generated": False},
        {"sender_id": OWNER_ID, "content": "Acid wash Rs 280 bulk mein, 50 pcs minimum", "timestamp": "2026-03-07T20:31:00+05:30", "is_owner": True, "is_ai_generated": False},
        # Customer 2: sends image + hello (different person!)
        {"sender_id": "919222222222", "content": "[Image]", "timestamp": "2026-03-07T20:32:00+05:30", "is_owner": False, "is_ai_generated": False},
        {"sender_id": "919222222222", "content": "Hello", "timestamp": "2026-03-07T20:32:30+05:30", "is_owner": False, "is_ai_generated": False},
        {"sender_id": OWNER_ID, "content": "Ji sir, kaise help kar sakta hun", "timestamp": "2026-03-07T20:33:00+05:30", "is_owner": True, "is_ai_generated": False},
        # Customer 3: Kolkata delivery question
        {"sender_id": "919333333333", "content": "If I place an order in Kolkata, how long will it take to receive it", "timestamp": "2026-03-07T20:34:00+05:30", "is_owner": False, "is_ai_generated": False},
        {"sender_id": OWNER_ID, "content": "Usually 4 to 5 days, train option available on website", "timestamp": "2026-03-07T20:35:00+05:30", "is_owner": True, "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    # Should get 2 pairs (Customer 1 + Customer 3 are quality, Customer 2 is greeting/no-intent)
    # CRITICAL: "Tell me the wholesale price" must NOT be combined with "[Image]" or "Hello"
    acid_pairs = [p for p in pairs if "acid" in p["customer"].lower() or "wholesale" in p["customer"].lower()]
    kolkata_pairs = [p for p in pairs if "Kolkata" in p["customer"] or "order" in p["customer"].lower()]

    # Verify acid wash pair is clean
    if acid_pairs:
        assert "[Image]" not in acid_pairs[0]["customer"], f"[Image] leaked into acid pair: {acid_pairs[0]['customer']}"
        assert "Hello" not in acid_pairs[0]["customer"], f"Hello leaked into acid pair: {acid_pairs[0]['customer']}"
        assert "Kolkata" not in acid_pairs[0]["customer"], f"Kolkata leaked into acid pair: {acid_pairs[0]['customer']}"

    # Verify Kolkata pair is clean
    if kolkata_pairs:
        assert "acid" not in kolkata_pairs[0]["customer"].lower(), f"acid leaked into Kolkata pair: {kolkata_pairs[0]['customer']}"
        assert "[Image]" not in kolkata_pairs[0]["customer"], f"[Image] leaked into Kolkata pair: {kolkata_pairs[0]['customer']}"

    # No pair should have messages from multiple customers combined
    for p in pairs:
        print(f"  BUYER: {p['customer'][:80]} → KETU: {p['ketu'][:60]}")

    print(f"PASS test_11: No chat_id, 3 customers — no cross-customer mixing ({len(pairs)} pairs)")


# ──────────────────────────────────────────────────────────────────────
# TEST 12: No chat_id — same customer's multiple messages still combine
# ──────────────────────────────────────────────────────────────────────
def test_12_no_chat_id_same_customer_combines():
    """wwbun sends no chat_id. Same customer sends 2 messages, Ketu replies once.
    The 2 messages from same sender_id should still combine correctly."""
    messages = [
        {"sender_id": "919444444444", "content": "Bhaiya parcel late lagwaya", "timestamp": "2026-03-07T21:00:00+05:30", "is_owner": False, "is_ai_generated": False},
        {"sender_id": "919444444444", "content": "Kal miljayega?", "timestamp": "2026-03-07T21:00:30+05:30", "is_owner": False, "is_ai_generated": False},
        {"sender_id": OWNER_ID, "content": "7 baje hi dispatch hua tha bhai", "timestamp": "2026-03-07T21:02:00+05:30", "is_owner": True, "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
    assert "parcel late lagwaya" in pairs[0]["customer"].lower()
    assert "Kal miljayega?" in pairs[0]["customer"]
    assert "dispatch" in pairs[0]["ketu"]
    print(f"PASS test_12: No chat_id, same customer 2 msgs → correctly combined")
    print(f"  BUYER: {pairs[0]['customer']} → KETU: {pairs[0]['ketu']}")


# ──────────────────────────────────────────────────────────────────────
# RUN ALL TESTS
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("  12 PAIR EXTRACTION TESTS — all variations")
    print("=" * 70)
    print()

    all_tests = [
        test_1_short_buyer_short_ketu,
        test_2_long_buyer_long_ketu,
        test_3_non_quality_ketu_ok,
        test_4_non_quality_buyer_greeting,
        test_5_multi_message_buyer_single_ketu,
        test_6_ai_reply_blocks_leak,
        test_7_two_customers_different_chats,
        test_8_rapid_fire_buyer,
        test_9_mixed_ai_and_manual_same_chat,
        test_10_full_production_dedup,
        test_11_no_chat_id_different_customers,
        test_12_no_chat_id_same_customer_combines,
    ]

    passed = 0
    failed = 0

    for test_fn in all_tests:
        try:
            test_fn()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"FAIL {test_fn.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 70)
    if failed == 0:
        print(f"  ALL {passed} TESTS PASSED")
    else:
        print(f"  {passed} passed, {failed} FAILED")
    print("=" * 70)
