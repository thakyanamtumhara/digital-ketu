"""Test pair extraction and deduplication fixes.

Tests the two bugs:
1. Duplicate messages from repeated wwbun syncs causing wrong combining
2. AI-generated owner replies not clearing customer buffer (messages leak across)

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


def test_1_basic_correct_pairing():
    """Test: Buyer asks question, Ketu replies → correct pair."""
    messages = [
        {"chat_id": "chat1", "sender_id": "cust1", "content": "Parcel late lagwaya bhaiya", "timestamp": "2026-03-07T19:41:00+05:30"},
        {"chat_id": "chat1", "sender_id": "cust1", "content": "Kal miljayega?", "timestamp": "2026-03-07T19:41:30+05:30"},
        {"chat_id": "chat1", "sender_id": OWNER_ID, "content": "7 baje hi lag gaya tha", "timestamp": "2026-03-07T19:44:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
    assert "Parcel late lagwaya bhaiya" in pairs[0]["customer"], f"Wrong customer msg: {pairs[0]['customer']}"
    assert "Kal miljayega" in pairs[0]["customer"], f"Missing second customer msg: {pairs[0]['customer']}"
    assert pairs[0]["ketu"] == "7 baje hi lag gaya tha", f"Wrong Ketu reply: {pairs[0]['ketu']}"
    assert "Oversized" not in pairs[0]["customer"], f"Oversized should NOT be in this pair!"
    print("PASS test_1: Basic correct pairing - Buyer: 'Parcel late lagwaya + Kal miljayega' → Ketu: '7 baje hi lag gaya tha'")


def test_2_ai_reply_clears_buffer():
    """Test: AI reply between customer messages should reset buffer.

    Scenario: Customer A asks, AI replies, then Customer A asks again.
    The second question should NOT be combined with the first.
    """
    messages = [
        {"chat_id": "chat1", "sender_id": "cust1", "content": "Oversized tshirt price batao", "timestamp": "2026-03-07T17:10:00+05:30"},
        {"chat_id": "chat1", "sender_id": OWNER_ID, "content": "Ji sir, 175 bulk rate hai", "timestamp": "2026-03-07T17:10:05+05:30", "is_ai_generated": True},
        {"chat_id": "chat1", "sender_id": "cust1", "content": "Parcel late lagwaya bhaiya", "timestamp": "2026-03-07T19:41:00+05:30"},
        {"chat_id": "chat1", "sender_id": "cust1", "content": "Kal miljayega?", "timestamp": "2026-03-07T19:41:30+05:30"},
        {"chat_id": "chat1", "sender_id": OWNER_ID, "content": "7 baje hi lag gaya tha", "timestamp": "2026-03-07T19:44:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    # Should get exactly 1 pair (the manual Ketu reply, not the AI one)
    assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}: {pairs}"
    # The pair should NOT contain "Oversized tshirt" — that was answered by AI already
    assert "Oversized" not in pairs[0]["customer"], (
        f"BUG: 'Oversized tshirt' leaked past AI reply into next pair! Got: {pairs[0]['customer']}"
    )
    assert "Parcel late" in pairs[0]["customer"], f"Expected 'Parcel late' in pair: {pairs[0]['customer']}"
    assert pairs[0]["ketu"] == "7 baje hi lag gaya tha"
    print("PASS test_2: AI reply clears buffer - 'Oversized tshirt' does NOT leak into '7 baje hi lag gaya tha' pair")


def test_3_multi_chat_no_cross_contamination():
    """Test: Messages from different chats should never mix."""
    messages = [
        # Chat 1: Customer asking about oversized
        {"chat_id": "919876543210@s.whatsapp.net", "sender_id": "cust1", "content": "Oversized tshirt dikhao", "timestamp": "2026-03-07T17:10:00+05:30"},
        {"chat_id": "919876543210@s.whatsapp.net", "sender_id": OWNER_ID, "content": "Ji sir, sale91.com pe dekho", "timestamp": "2026-03-07T17:11:00+05:30", "is_ai_generated": True},
        # Chat 2: Sujal Ahir asking about parcel (DIFFERENT customer)
        {"chat_id": "919928258289@s.whatsapp.net", "sender_id": "cust2", "content": "Parcel late lagwaya bhaiya", "timestamp": "2026-03-07T19:41:00+05:30"},
        {"chat_id": "919928258289@s.whatsapp.net", "sender_id": "cust2", "content": "Kal miljayega?", "timestamp": "2026-03-07T19:41:30+05:30"},
        {"chat_id": "919928258289@s.whatsapp.net", "sender_id": OWNER_ID, "content": "7 baje hi lag gaya tha", "timestamp": "2026-03-07T19:44:00+05:30", "is_ai_generated": False},
    ]
    pairs = _extract_conversation_pairs(messages, OWNER_ID)

    assert len(pairs) == 1, f"Expected 1 pair (only manual Ketu reply), got {len(pairs)}"
    assert "Oversized" not in pairs[0]["customer"], f"Cross-chat contamination! Got: {pairs[0]['customer']}"
    assert "Parcel late" in pairs[0]["customer"]
    print("PASS test_3: Multi-chat isolation - Oversized (chat1) stays separate from Parcel (chat2)")


def test_4_dedup_fingerprint():
    """Test: Same message sent across multiple wwbun syncs should be deduplicated."""
    msg1 = {"chat_id": "chat1", "sender_id": "cust1", "content": "Oversized tshirt", "timestamp": "2026-03-07T17:10:00+05:30"}
    msg2 = {"chat_id": "chat1", "sender_id": "cust1", "content": "Oversized tshirt", "timestamp": "2026-03-07T17:10:00+05:30"}
    msg3 = {"chat_id": "chat1", "sender_id": "cust1", "content": "Different message", "timestamp": "2026-03-07T17:15:00+05:30"}

    fp1 = _message_fingerprint(msg1)
    fp2 = _message_fingerprint(msg2)
    fp3 = _message_fingerprint(msg3)

    assert fp1 == fp2, f"Same message should have same fingerprint: {fp1} != {fp2}"
    assert fp1 != fp3, f"Different messages should have different fingerprints"
    assert fp1 != "", "Fingerprint should not be empty for messages with content"

    # Test empty content
    empty = {"chat_id": "chat1", "sender_id": "cust1", "content": "", "timestamp": "ts"}
    assert _message_fingerprint(empty) == "", "Empty content should return empty fingerprint"

    print("PASS test_4: Dedup fingerprinting - same messages matched, different messages distinguished")


def test_5_realistic_wwbun_scenario():
    """Test: Realistic scenario — wwbun sends same batch 3 times + Ketu replies once.

    This is EXACTLY the bug from production:
    - wwbun syncs last 20 messages every few minutes
    - Same messages appear in buffer 3-4 times
    - Ketu's manual reply "7 baje hi lag gaya tha" gets paired with
      combined duplicates "Oversized tshirt | Oversized tshirt | Oversized tshirt"
    """
    # Simulate 3 wwbun syncs sending overlapping messages
    sync1 = [
        {"chat_id": "chat_sujal", "sender_id": "cust_sujal", "content": "Bhaiya please dkehna ki acid wash aache ho", "timestamp": "2026-03-07T17:11:00+05:30"},
        {"chat_id": "chat_sujal", "sender_id": OWNER_ID, "content": "Ok", "timestamp": "2026-03-07T17:18:00+05:30", "is_ai_generated": False},
        {"chat_id": "chat_sujal", "sender_id": "cust_sujal", "content": "Hanji thank u", "timestamp": "2026-03-07T17:18:10+05:30"},
        {"chat_id": "chat_sujal", "sender_id": "cust_sujal", "content": "And jaldi nikalwafena thora", "timestamp": "2026-03-07T17:18:20+05:30"},
        {"chat_id": "chat_sujal", "sender_id": "cust_sujal", "content": "Kal chahiye", "timestamp": "2026-03-07T17:18:30+05:30"},
    ]

    sync2 = sync1 + [  # Same messages + new ones
        {"chat_id": "chat_sujal", "sender_id": "cust_sujal", "content": "Parcel late lagwaya bhaiya", "timestamp": "2026-03-07T19:41:00+05:30"},
        {"chat_id": "chat_sujal", "sender_id": "cust_sujal", "content": "Kal miljayega?", "timestamp": "2026-03-07T19:41:30+05:30"},
    ]

    sync3 = sync2 + [  # Same messages + Ketu's reply
        {"chat_id": "chat_sujal", "sender_id": OWNER_ID, "content": "7 baje hi lag gaya tha", "timestamp": "2026-03-07T19:44:00+05:30", "is_ai_generated": False},
    ]

    # Simulate buffer accumulation WITH dedup
    buffer = []
    existing_fps = set()

    for sync_batch in [sync1, sync2, sync3]:
        for m in sync_batch:
            fp = _message_fingerprint(m)
            if fp and fp in existing_fps:
                continue  # Dedup!
            buffer.append(m)
            if fp:
                existing_fps.add(fp)

    # Total unique messages should be 8 (5 from sync1 + 2 new from sync2 + 1 new from sync3)
    assert len(buffer) == 8, f"Expected 8 unique messages after dedup, got {len(buffer)}"

    # Now extract pairs from deduplicated buffer
    pairs = _extract_conversation_pairs(buffer, OWNER_ID)

    # Check that NO pair has duplicate messages in it
    for pair in pairs:
        parts = pair["customer"].split(" | ")
        # No duplicates within a single pair
        if len(parts) != len(set(parts)):
            print(f"FAIL: Duplicate messages in pair: {pair['customer']}")
            assert False, f"Duplicate in pair: {parts}"

    # The final pair should be Parcel+Kal → 7 baje hi lag gaya tha
    ketu_pairs = [p for p in pairs if p["ketu"] == "7 baje hi lag gaya tha"]
    assert len(ketu_pairs) == 1, f"Expected exactly 1 '7 baje' pair, got {len(ketu_pairs)}: {ketu_pairs}"
    assert "Oversized" not in ketu_pairs[0]["customer"], f"Oversized should not be here: {ketu_pairs[0]['customer']}"

    print(f"PASS test_5: Realistic wwbun scenario - {len(buffer)} unique msgs from 3 syncs, pairs correct")
    print(f"  Pairs found: {len(pairs)}")
    for p in pairs:
        print(f"    BUYER: {p['customer'][:60]} → KETU: {p['ketu'][:40]}")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing pair extraction & deduplication fixes")
    print("=" * 70)
    print()

    passed = 0
    failed = 0

    for test_fn in [test_1_basic_correct_pairing, test_2_ai_reply_clears_buffer,
                     test_3_multi_chat_no_cross_contamination, test_4_dedup_fingerprint,
                     test_5_realistic_wwbun_scenario]:
        try:
            test_fn()
            passed += 1
        except (AssertionError, Exception) as e:
            print(f"FAIL {test_fn.__name__}: {e}")
            failed += 1
        print()

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
