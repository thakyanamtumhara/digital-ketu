"""10-buyer simulation test — run 5 times to verify no cross-chat mixing.

Simulates 10 different buyers messaging Ketu, each with unique questions.
wwbun format: NO chat_id (just sender_id, content, is_owner, is_ai_generated).
Verifies: no buyer's message appears in another buyer's pair.

Run: python tests/test_10_buyers_simulation.py
"""

import sys
import os
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock database module
import types
mock_db = types.ModuleType("core.database")
mock_db.is_db_available = lambda: False
mock_db.kv_get = lambda k: None
mock_db.kv_set = lambda k, v: None
mock_db.load_knowledge_from_db = lambda k: None
mock_db.save_knowledge_to_db = lambda k, v: None
mock_db.init_db = lambda: False
sys.modules["core.database"] = mock_db

from main import _extract_conversation_pairs, _message_fingerprint

OWNER_ID = "owner_ketu_123"

# 10 different buyers with unique conversations (wwbun format — NO chat_id)
BUYER_CONVERSATIONS = [
    {
        "phone": "919100000001",
        "buyer_msgs": ["Trackpant ka bulk rate kya hai bhaiya"],
        "ketu_reply": "Trackpant Rs 220 bulk, 8 colors ready stock",
        "keyword": "Trackpant",
    },
    {
        "phone": "919100000002",
        "buyer_msgs": ["Hoodie 320 GSM black color available?"],
        "ketu_reply": "Haan ji 320 GSM hoodie Rs 450, black ready stock hai",
        "keyword": "Hoodie",
    },
    {
        "phone": "919100000003",
        "buyer_msgs": ["Round neck 180 GSM 500 pieces chahiye", "White aur grey dono mein"],
        "ketu_reply": "500 pcs pe Rs 85 per piece, dono color ready",
        "keyword": "Round neck",
    },
    {
        "phone": "919100000004",
        "buyer_msgs": ["Polo tshirt DTG printing ke liye konsa best hai"],
        "ketu_reply": "DTG ke liye premium cotton polo best hai Rs 220",
        "keyword": "Polo",
    },
    {
        "phone": "919100000005",
        "buyer_msgs": ["Acid wash tshirt ka sample bhej sakte ho?", "Jaipur delivery kitne din?"],
        "ketu_reply": "Sample Rs 280, Jaipur 2-3 din mein aa jayega",
        "keyword": "Acid wash",
    },
    {
        "phone": "919100000006",
        "buyer_msgs": ["Joggers pant 1000 pieces ka rate batao", "Bulk discount milega?"],
        "ketu_reply": "Joggers 1000 pe special rate Rs 195, extra discount bhi",
        "keyword": "Joggers",
    },
    {
        "phone": "919100000007",
        "buyer_msgs": ["Kolkata mein order deliver hoga kitne din mein"],
        "ketu_reply": "Kolkata 4-5 din, train option se next day possible",
        "keyword": "Kolkata",
    },
    {
        "phone": "919100000008",
        "buyer_msgs": ["Crop top available hai ladies ke liye?"],
        "ketu_reply": "Ladies crop top Rs 145 bulk, 6 colors mein ready",
        "keyword": "Crop top",
    },
    {
        "phone": "919100000009",
        "buyer_msgs": ["Sublimation printing support karta hai fabric?", "Polyester blend chahiye"],
        "ketu_reply": "Polyester blend Rs 120, sublimation perfect chalega",
        "keyword": "Sublimation",
    },
    {
        "phone": "919100000010",
        "buyer_msgs": ["Franchise model hai kya aapka?", "Meri city mein outlet kholna hai"],
        "ketu_reply": "Franchise model available hai, city details bhejo discuss karenge",
        "keyword": "Franchise",
    },
]


def build_messages_wwbun_format(conversations, shuffle_order=False, base_offset_min=0):
    """Build messages in exact wwbun format (no chat_id!) with interleaved timing."""
    messages = []
    base_hour = 10
    base_min = base_offset_min

    # Each conversation happens at slightly different times
    for i, conv in enumerate(conversations):
        ts_min = base_min + i * 3  # 3 min apart
        ts_hour = base_hour + ts_min // 60
        ts_min = ts_min % 60

        # Buyer messages
        for j, buyer_msg in enumerate(conv["buyer_msgs"]):
            messages.append({
                "sender_id": conv["phone"],
                "content": buyer_msg,
                "timestamp": f"2026-03-07T{ts_hour:02d}:{ts_min:02d}:{j * 15:02d}+05:30",
                "is_owner": False,
                "is_ai_generated": False,
            })

        # Ketu reply (1-2 min after buyer)
        reply_min = ts_min + 1
        reply_hour = ts_hour + reply_min // 60
        reply_min = reply_min % 60
        messages.append({
            "sender_id": OWNER_ID,
            "content": conv["ketu_reply"],
            "timestamp": f"2026-03-07T{reply_hour:02d}:{reply_min:02d}:00+05:30",
            "is_owner": True,
            "is_ai_generated": False,
        })

    if shuffle_order:
        # Shuffle but keep timestamps for sorting (tests robustness)
        random.shuffle(messages)

    return messages


def verify_no_cross_contamination(pairs, conversations, run_label):
    """Verify that NO buyer's keyword appears in another buyer's pair.
    Match by Ketu reply keyword (more reliable than customer text which can overlap)."""
    errors = []

    for pair in pairs:
        # Find which conversation this pair belongs to by matching Ketu's reply
        matched_conv = None
        for conv in conversations:
            # Check if Ketu reply matches this conversation's expected reply
            ketu_keywords = conv["ketu_reply"].lower().split()[:3]  # First 3 words
            if all(kw in pair["ketu"].lower() for kw in ketu_keywords):
                matched_conv = conv
                break

        if not matched_conv:
            # Try matching by keyword in customer text as fallback
            for conv in conversations:
                if conv["keyword"].lower() in pair["customer"].lower():
                    matched_conv = conv
                    break

        if not matched_conv:
            continue

        # Check that NO OTHER buyer's keyword appears in this pair's customer text
        for other_conv in conversations:
            if other_conv["phone"] == matched_conv["phone"]:
                continue
            if other_conv["keyword"].lower() in pair["customer"].lower():
                errors.append(
                    f"CROSS-CONTAMINATION: '{other_conv['keyword']}' (buyer {other_conv['phone']}) "
                    f"leaked into pair of buyer {matched_conv['phone']}: {pair['customer'][:80]}"
                )

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
        return False
    return True


def run_simulation(run_num, shuffle=False):
    """Run one full 10-buyer simulation."""
    label = f"Run {run_num}" + (" (shuffled)" if shuffle else "")
    messages = build_messages_wwbun_format(BUYER_CONVERSATIONS, shuffle_order=shuffle)

    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    # Verify no cross-contamination
    clean = verify_no_cross_contamination(pairs, BUYER_CONVERSATIONS, label)

    # Print summary
    print(f"\n{'─' * 60}")
    print(f"  {label}: {len(messages)} messages → {len(pairs)} pairs")
    print(f"{'─' * 60}")
    for i, p in enumerate(pairs):
        print(f"  [{i+1}] BUYER: {p['customer'][:70]}")
        print(f"      KETU:  {p['ketu'][:70]}")

    if clean:
        print(f"\n  RESULT: PASS — No cross-customer mixing detected")
    else:
        print(f"\n  RESULT: FAIL — Cross-customer contamination found!")

    return clean


if __name__ == "__main__":
    print("=" * 70)
    print("  10-BUYER SIMULATION TEST — 5 runs")
    print("  (wwbun format: NO chat_id, only sender_id)")
    print("=" * 70)

    all_pass = True

    # Run 1: Normal order
    all_pass &= run_simulation(1, shuffle=False)

    # Run 2: Shuffled message order
    all_pass &= run_simulation(2, shuffle=True)

    # Run 3: Normal again (consistency check)
    all_pass &= run_simulation(3, shuffle=False)

    # Run 4: Shuffled again (different random seed)
    all_pass &= run_simulation(4, shuffle=True)

    # Run 5: Simulate dedup (same messages sent twice via wwbun overlapping syncs)
    print(f"\n{'─' * 60}")
    print(f"  Run 5: DEDUP TEST — 2 overlapping wwbun syncs")
    print(f"{'─' * 60}")
    msgs_sync1 = build_messages_wwbun_format(BUYER_CONVERSATIONS[:5])
    msgs_sync2 = build_messages_wwbun_format(BUYER_CONVERSATIONS[:5])  # Duplicate
    msgs_sync3 = build_messages_wwbun_format(BUYER_CONVERSATIONS[5:], base_offset_min=15)  # Later timestamps

    # Dedup like production
    buffer = []
    existing_fps = set()
    total_received = 0
    total_duped = 0
    for batch in [msgs_sync1, msgs_sync2, msgs_sync3]:
        for m in batch:
            total_received += 1
            fp = _message_fingerprint(m)
            if fp and fp in existing_fps:
                total_duped += 1
                continue
            buffer.append(m)
            if fp:
                existing_fps.add(fp)

    pairs = _extract_conversation_pairs(buffer, OWNER_ID)
    clean = verify_no_cross_contamination(pairs, BUYER_CONVERSATIONS, "Run 5")

    print(f"  Received: {total_received}, Duped: {total_duped}, Unique: {len(buffer)}")
    print(f"  Pairs: {len(pairs)}")
    for i, p in enumerate(pairs):
        print(f"  [{i+1}] BUYER: {p['customer'][:70]}")
        print(f"      KETU:  {p['ketu'][:70]}")

    if clean:
        print(f"\n  RESULT: PASS — Dedup works, no cross-customer mixing")
    else:
        print(f"\n  RESULT: FAIL — Cross-customer contamination after dedup!")
    all_pass &= clean

    # Final verdict
    print()
    print("=" * 70)
    if all_pass:
        print("  ALL 5 RUNS PASSED — NO CROSS-CUSTOMER MIXING DETECTED")
    else:
        print("  SOME RUNS FAILED — CROSS-CUSTOMER MIXING EXISTS")
    print("=" * 70)
