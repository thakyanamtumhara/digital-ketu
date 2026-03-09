"""Test quality pair detection + batch learning threshold.

Tests:
  1-5: Quality pair classification (quality vs non-quality)
  6-10: Batch learning trigger (20-pair threshold, 30-min cooldown, force flush)

Run: python tests/test_quality_and_batch.py
"""

import sys
import os
import time
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

from main import (
    _extract_conversation_pairs,
    _count_quality_owner_messages,
    _message_fingerprint,
    _safe_bool,
    _LEARNING_BUFFER_MIN_PAIRS,
    _LEARNING_FLUSH_COOLDOWN,
    _LEARNING_FORCE_FLUSH_PAIRS,
)

OWNER_ID = "owner123"


def _make_msg(chat_id, sender_id, content, timestamp, is_ai=False):
    """Helper to build a message dict."""
    return {
        "chat_id": chat_id,
        "sender_id": sender_id,
        "content": content,
        "timestamp": timestamp,
        "is_ai_generated": is_ai,
    }


def _make_quality_pair(pair_num, chat_prefix="chat"):
    """Generate one buyer+Ketu quality pair (business question + informative answer)."""
    chat_id = f"{chat_prefix}_{pair_num}@wa"
    base_min = pair_num * 2
    buyer = _make_msg(
        chat_id, f"cust{pair_num}",
        f"Bhaiya oversized tshirt {pair_num * 100} pcs ka rate batao 240 GSM",
        f"2026-03-07T10:{base_min:02d}:00+05:30",
    )
    ketu = _make_msg(
        chat_id, OWNER_ID,
        f"Ji sir {pair_num * 100} pcs ke liye Rs {180 - pair_num} per piece bulk rate hai",
        f"2026-03-07T10:{base_min + 1:02d}:00+05:30",
        is_ai=False,
    )
    return [buyer, ketu]


# ══════════════════════════════════════════════════════════════════════
# QUALITY PAIR DETECTION TESTS (1-5)
# ══════════════════════════════════════════════════════════════════════

def test_q1_quality_business_pair_counted():
    """A real business Q+A should be counted as 1 quality pair."""
    msgs = [
        _make_msg("c1@wa", "cust1", "Hoodie 320 GSM rate bhaiya bulk mein", "2026-03-07T10:00:00+05:30"),
        _make_msg("c1@wa", OWNER_ID, "320 GSM hoodie Rs 450 bulk, 10 colors available", "2026-03-07T10:01:00+05:30"),
    ]
    count = _count_quality_owner_messages(msgs, OWNER_ID)
    assert count == 1, f"Expected 1 quality pair, got {count}"
    print("PASS test_q1: Business pair (Hoodie rate → Rs 450) counted as 1 quality pair")


def test_q2_junk_ketu_reply_not_counted():
    """Ketu replying 'Ok' or 'Ji' should NOT count as quality pair."""
    msgs = [
        _make_msg("c2@wa", "cust2", "Bhaiya parcel dispatch karo", "2026-03-07T11:00:00+05:30"),
        _make_msg("c2@wa", OWNER_ID, "Ok", "2026-03-07T11:01:00+05:30"),
        _make_msg("c3@wa", "cust3", "Sample bhejiye na", "2026-03-07T11:02:00+05:30"),
        _make_msg("c3@wa", OWNER_ID, "Ji", "2026-03-07T11:03:00+05:30"),
    ]
    count = _count_quality_owner_messages(msgs, OWNER_ID)
    assert count == 0, f"Expected 0 quality pairs (Ok/Ji are junk), got {count}"
    print("PASS test_q2: Junk replies ('Ok', 'Ji') NOT counted as quality pairs")


def test_q3_ai_reply_not_counted():
    """AI-generated replies should NOT count toward quality pair threshold."""
    msgs = [
        _make_msg("c4@wa", "cust4", "Polo tshirt available hai?", "2026-03-07T12:00:00+05:30"),
        _make_msg("c4@wa", OWNER_ID, "Ji sir, polo Rs 220 bulk mein, 5 colors available", "2026-03-07T12:00:30+05:30", is_ai=True),
    ]
    count = _count_quality_owner_messages(msgs, OWNER_ID)
    assert count == 0, f"Expected 0 quality pairs (AI reply), got {count}"
    print("PASS test_q3: AI-generated reply NOT counted as quality pair")


def test_q4_mixed_quality_and_junk():
    """Mix of quality pairs, junk replies, and AI replies — only quality manual pairs counted."""
    msgs = []
    # Quality pair 1: real business Q+A
    msgs += [
        _make_msg("c5@wa", "cust5", "Round neck 180 GSM 500 pcs rate?", "2026-03-07T13:00:00+05:30"),
        _make_msg("c5@wa", OWNER_ID, "500 pe Rs 85 per piece, factory direct", "2026-03-07T13:01:00+05:30"),
    ]
    # Junk pair: "Theek hai"
    msgs += [
        _make_msg("c6@wa", "cust6", "Order ready hai?", "2026-03-07T13:02:00+05:30"),
        _make_msg("c6@wa", OWNER_ID, "Theek hai", "2026-03-07T13:03:00+05:30"),
    ]
    # AI pair
    msgs += [
        _make_msg("c7@wa", "cust7", "Printing ke liye konsa tshirt best hai?", "2026-03-07T13:04:00+05:30"),
        _make_msg("c7@wa", OWNER_ID, "DTG ke liye 240 GSM oversized best hai sir", "2026-03-07T13:05:00+05:30", is_ai=True),
    ]
    # Quality pair 2: real business Q+A
    msgs += [
        _make_msg("c8@wa", "cust8", "Bhaiya acid wash tshirt rate batao bulk", "2026-03-07T13:06:00+05:30"),
        _make_msg("c8@wa", OWNER_ID, "Acid wash Rs 280 bulk, 50 pcs minimum order", "2026-03-07T13:07:00+05:30"),
    ]

    count = _count_quality_owner_messages(msgs, OWNER_ID)
    assert count == 2, f"Expected 2 quality pairs (skipping junk+AI), got {count}"
    print("PASS test_q4: Mixed buffer — 2 quality, 1 junk, 1 AI → correctly counted 2")


def test_q5_greeting_only_buyer_not_counted():
    """Buyer 'Hi' + substantive Ketu reply should NOT be quality (no business content from buyer)."""
    msgs = [
        _make_msg("c9@wa", "cust9", "Hi", "2026-03-07T14:00:00+05:30"),
        _make_msg("c9@wa", OWNER_ID, "Ji sir kaise help kar sakta hun, oversized polo sab available hai", "2026-03-07T14:01:00+05:30"),
    ]
    count = _count_quality_owner_messages(msgs, OWNER_ID)
    assert count == 0, f"Expected 0 quality pairs (buyer just said Hi), got {count}"
    print("PASS test_q5: Buyer said 'Hi' → NOT counted as quality pair (no business intent)")


# ══════════════════════════════════════════════════════════════════════
# BATCH LEARNING THRESHOLD TESTS (6-10)
# ══════════════════════════════════════════════════════════════════════

def test_b6_under_threshold_no_flush():
    """19 quality pairs → should NOT trigger flush (need 20)."""
    msgs = []
    for i in range(19):
        msgs += _make_quality_pair(i)
    count = _count_quality_owner_messages(msgs, OWNER_ID)
    should_flush = count >= _LEARNING_BUFFER_MIN_PAIRS
    assert count == 19, f"Expected 19 quality pairs, got {count}"
    assert not should_flush, f"19 pairs should NOT trigger flush"
    print(f"PASS test_b6: 19 quality pairs → count={count}, flush={should_flush} (correctly no flush)")


def test_b7_at_threshold_triggers_flush():
    """Exactly 20 quality pairs → should trigger flush."""
    msgs = []
    for i in range(20):
        msgs += _make_quality_pair(i)
    count = _count_quality_owner_messages(msgs, OWNER_ID)
    # Simulate no cooldown (first ever flush)
    cooldown_active = False
    force_flush = count >= _LEARNING_FORCE_FLUSH_PAIRS
    should_flush = count >= _LEARNING_BUFFER_MIN_PAIRS and (not cooldown_active or force_flush)
    assert count == 20, f"Expected 20 quality pairs, got {count}"
    assert should_flush, f"20 pairs with no cooldown should trigger flush"
    print(f"PASS test_b7: 20 quality pairs, no cooldown → flush={should_flush} (correctly triggers)")


def test_b8_cooldown_blocks_second_flush():
    """20 quality pairs but cooldown active → should NOT flush (wait 30 min)."""
    msgs = []
    for i in range(25):
        msgs += _make_quality_pair(i)
    count = _count_quality_owner_messages(msgs, OWNER_ID)

    # Simulate: last flush was 5 minutes ago (cooldown still active)
    last_flush_time = time.time() - 300  # 5 min ago
    time_since_flush = time.time() - last_flush_time
    cooldown_active = time_since_flush < _LEARNING_FLUSH_COOLDOWN  # 300 < 1800 = True
    force_flush = count >= _LEARNING_FORCE_FLUSH_PAIRS  # 25 < 50 = False
    should_flush = count >= _LEARNING_BUFFER_MIN_PAIRS and (not cooldown_active or force_flush)

    assert count == 25, f"Expected 25 quality pairs, got {count}"
    assert cooldown_active, f"Cooldown should be active (5 min ago)"
    assert not force_flush, f"25 < 50, no force flush"
    assert not should_flush, f"25 pairs + cooldown active → should NOT flush"
    print(f"PASS test_b8: 25 quality pairs, cooldown active (5 min ago) → flush={should_flush} (correctly blocked)")


def test_b9_cooldown_expired_allows_flush():
    """20 pairs + cooldown expired (35 min since last flush) → should flush."""
    msgs = []
    for i in range(22):
        msgs += _make_quality_pair(i)
    count = _count_quality_owner_messages(msgs, OWNER_ID)

    # Simulate: last flush was 35 minutes ago (cooldown expired)
    last_flush_time = time.time() - 2100  # 35 min ago
    time_since_flush = time.time() - last_flush_time
    cooldown_active = time_since_flush < _LEARNING_FLUSH_COOLDOWN  # 2100 < 1800 = False
    force_flush = count >= _LEARNING_FORCE_FLUSH_PAIRS
    should_flush = count >= _LEARNING_BUFFER_MIN_PAIRS and (not cooldown_active or force_flush)

    assert count == 22, f"Expected 22 quality pairs, got {count}"
    assert not cooldown_active, f"Cooldown should be expired (35 min ago)"
    assert should_flush, f"22 pairs + cooldown expired → should flush"
    print(f"PASS test_b9: 22 quality pairs, cooldown expired (35 min) → flush={should_flush} (correctly triggers)")


def test_b10_force_flush_overrides_cooldown():
    """50+ quality pairs + cooldown active → should force flush anyway."""
    msgs = []
    for i in range(55):
        msgs += _make_quality_pair(i)
    count = _count_quality_owner_messages(msgs, OWNER_ID)

    # Simulate: last flush was 10 minutes ago (cooldown still active)
    last_flush_time = time.time() - 600  # 10 min ago
    time_since_flush = time.time() - last_flush_time
    cooldown_active = time_since_flush < _LEARNING_FLUSH_COOLDOWN  # 600 < 1800 = True
    force_flush = count >= _LEARNING_FORCE_FLUSH_PAIRS  # 55 >= 50 = True
    should_flush = count >= _LEARNING_BUFFER_MIN_PAIRS and (not cooldown_active or force_flush)

    assert count == 55, f"Expected 55 quality pairs, got {count}"
    assert cooldown_active, f"Cooldown should be active (10 min ago)"
    assert force_flush, f"55 >= 50 → force flush should be True"
    assert should_flush, f"55 pairs + force flush → should flush even with cooldown"
    print(f"PASS test_b10: 55 quality pairs, cooldown active BUT force flush → flush={should_flush} (correctly overrides)")


# ══════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("  QUALITY PAIR DETECTION + BATCH LEARNING TESTS")
    print("=" * 70)
    print()
    print(f"  Config: MIN_PAIRS={_LEARNING_BUFFER_MIN_PAIRS}, "
          f"COOLDOWN={_LEARNING_FLUSH_COOLDOWN}s, "
          f"FORCE_FLUSH={_LEARNING_FORCE_FLUSH_PAIRS}")
    print()

    all_tests = [
        # Quality pair detection
        test_q1_quality_business_pair_counted,
        test_q2_junk_ketu_reply_not_counted,
        test_q3_ai_reply_not_counted,
        test_q4_mixed_quality_and_junk,
        test_q5_greeting_only_buyer_not_counted,
        # Batch learning threshold
        test_b6_under_threshold_no_flush,
        test_b7_at_threshold_triggers_flush,
        test_b8_cooldown_blocks_second_flush,
        test_b9_cooldown_expired_allows_flush,
        test_b10_force_flush_overrides_cooldown,
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
