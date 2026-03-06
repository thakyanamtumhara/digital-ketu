"""5 End-to-End Test Scenarios: Edit → Learn Flow

Tests the complete flow:
1. Customer sends message
2. AI replies (possibly wrong)
3. Ketu edits the AI reply
4. Digital Ketu receives the edited reply
5. Digital Ketu learns from the correction

These tests mock the WhatsApp API and Anthropic API to run locally
without external dependencies.
"""

import asyncio
import json
import time
import threading
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pytest_asyncio

# Mock environment before imports
import os
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-123")
os.environ.setdefault("WHATSAPP_ACCESS_TOKEN", "test-token")
os.environ.setdefault("WHATSAPP_PHONE_NUMBER_ID", "123456")
os.environ.setdefault("WHATSAPP_VERIFY_TOKEN", "test-verify")
os.environ.setdefault("AUTO_REPLY_ENABLED", "true")

from httpx import AsyncClient, ASGITransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_anthropic_response(text: str):
    """Create a mock Anthropic API response."""
    mock_resp = MagicMock()
    mock_resp.content = [MagicMock(text=text)]
    mock_resp.model = "claude-haiku-4-5-20251001"
    mock_resp.usage = MagicMock(input_tokens=100, output_tokens=50)
    return mock_resp


def _correction_analysis_response(
    what_went_wrong: str,
    new_faq_q: str = "",
    new_faq_a: str = "",
    new_rule: str = "",
    style_lesson: str = "",
):
    """Build a mock correction analysis JSON response."""
    faq = {"question": new_faq_q, "answer": new_faq_a} if new_faq_q else None
    return json.dumps({
        "new_faq": faq,
        "style_lesson": style_lesson or None,
        "new_rule": new_rule or None,
        "example_conversation": {"customer": new_faq_q or "test", "reply": new_faq_a or "test"},
        "what_went_wrong": what_went_wrong,
    })


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_state():
    """Reset in-memory state before each test."""
    from integrations.whatsapp.sender import _last_sent_messages
    from core.conversation_log import _conversation_log
    _last_sent_messages.clear()
    _conversation_log.clear()
    yield


@pytest.fixture
def app():
    """Import the FastAPI app."""
    from main import app as fastapi_app
    return fastapi_app


@pytest_asyncio.fixture
async def client(app):
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _simulate_ai_sent_message(phone: str, message: str, msg_id: str = "wamid_test_123"):
    """Simulate that Digital Ketu sent a message (populate sender tracking)."""
    from integrations.whatsapp.sender import _last_sent_messages
    _last_sent_messages[phone] = {
        "message_id": msg_id,
        "text": message,
        "timestamp": time.time(),
    }


def _simulate_conversation_log(phone: str, name: str, customer_msg: str, ai_reply: str):
    """Simulate a logged conversation (AI replied to customer)."""
    from core.conversation_log import log_conversation
    log_conversation(
        customer_phone=phone,
        customer_name=name,
        customer_message=customer_msg,
        ai_reply=ai_reply,
    )


# ===========================================================================
# TEST 1: Wrong Price — AI says ₹170, Ketu corrects to ₹190
# ===========================================================================

@pytest.mark.asyncio
async def test_1_wrong_price_correction(client):
    """Scenario: Customer asks t-shirt rate, AI gives wrong price.
    Ketu edits with correct price. Digital Ketu should learn.
    """
    phone = "919876543210"
    customer_msg = "Bhai round neck t-shirt ka rate kya hai?"
    ai_reply = "Ji bhai, round neck t-shirt ₹170/piece hai. MOQ 50 pieces."
    ketu_correction = "Bhai rate ₹190/piece hai, MOQ 50 pieces."

    # Step 1: Simulate AI replied
    _simulate_ai_sent_message(phone, ai_reply)
    _simulate_conversation_log(phone, "Ravi", customer_msg, ai_reply)

    # Step 2: Ketu edits the message
    learned_data = {}

    def mock_learn_from_correction(**kwargs):
        learned_data.update(kwargs)
        return {
            "status": "learned",
            "what_went_wrong": "AI quoted wrong price ₹170 instead of ₹190",
            "updates_applied": ["faq_updated"],
            "count": 1,
        }

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_edit, \
         patch("learner.realtime_learner.learn_from_correction", side_effect=mock_learn_from_correction):

        mock_edit.return_value = {"messages": [{"id": "wamid_edited_1"}]}

        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ketu_correction,
            "message_id": "wamid_test_123",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "edited"
    assert data["learning"] == "correction sent to learner"
    assert "₹170" in data["original_ai_reply"]
    assert "round neck" in data["customer_question"].lower() or "rate" in data["customer_question"].lower()

    # Wait for background thread
    time.sleep(0.5)

    # Verify learn_from_correction was called with correct data
    assert learned_data.get("customer_message") == customer_msg
    assert learned_data.get("ai_reply") == ai_reply
    assert learned_data.get("ketu_correction") == ketu_correction
    assert learned_data.get("customer_phone") == phone

    print("TEST 1 PASSED: Wrong price → Ketu edited → Digital Ketu learned correct price")


# ===========================================================================
# TEST 2: Wrong Product Info — AI gives wrong GSM
# ===========================================================================

@pytest.mark.asyncio
async def test_2_wrong_product_info_correction(client):
    """Scenario: Customer asks about hoodie GSM, AI says 300 GSM.
    Ketu corrects to 380 GSM. Digital Ketu should learn.
    """
    phone = "919988776655"
    customer_msg = "Hoodie ka GSM kitna hai?"
    ai_reply = "Ji sir, hoodie 300 GSM hai, fleece material."
    ketu_correction = "Sir hoodie 380 GSM hai, premium fleece with brushed inside."

    _simulate_ai_sent_message(phone, ai_reply)
    _simulate_conversation_log(phone, "Aakash", customer_msg, ai_reply)

    learned_data = {}

    def mock_learn(**kwargs):
        learned_data.update(kwargs)
        return {
            "status": "learned",
            "what_went_wrong": "AI quoted wrong GSM (300 instead of 380)",
            "updates_applied": ["product_info_updated"],
            "count": 1,
        }

    with patch("integrations.whatsapp.sender.edit_last_message", new_callable=AsyncMock) as mock_edit, \
         patch("learner.realtime_learner.learn_from_correction", side_effect=mock_learn):

        mock_edit.return_value = {"messages": [{"id": "wamid_edited_2"}]}

        # Edit without message_id → uses edit_last_message
        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ketu_correction,
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "edited"
    assert data["learning"] == "correction sent to learner"

    time.sleep(0.5)

    assert learned_data.get("ai_reply") == ai_reply
    assert learned_data.get("ketu_correction") == ketu_correction
    assert "Hoodie" in learned_data.get("customer_message", "")

    print("TEST 2 PASSED: Wrong GSM → Ketu edited → Digital Ketu learned correct GSM")


# ===========================================================================
# TEST 3: Wrong Tone — AI too formal, Ketu wants casual Hinglish
# ===========================================================================

@pytest.mark.asyncio
async def test_3_wrong_tone_correction(client):
    """Scenario: AI replies too formally. Ketu edits with casual Hinglish tone.
    Digital Ketu should learn the style correction.
    """
    phone = "919112233445"
    customer_msg = "Shipping kitne din mein hoga?"
    ai_reply = "Dear customer, shipping will take approximately 3-5 business days to your location."
    ketu_correction = "Bhai 3-4 din mein aa jayega, courier se bhejte hai."

    _simulate_ai_sent_message(phone, ai_reply)
    _simulate_conversation_log(phone, "Suresh", customer_msg, ai_reply)

    learned_data = {}

    def mock_learn(**kwargs):
        learned_data.update(kwargs)
        return {
            "status": "learned",
            "what_went_wrong": "AI replied too formally in English instead of casual Hinglish",
            "updates_applied": ["style_updated", "rule_added"],
            "count": 2,
        }

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_edit, \
         patch("learner.realtime_learner.learn_from_correction", side_effect=mock_learn):

        mock_edit.return_value = {"messages": [{"id": "wamid_edited_3"}]}

        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ketu_correction,
            "message_id": "wamid_test_123",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "edited"
    assert data["learning"] == "correction sent to learner"

    time.sleep(0.5)

    assert learned_data.get("ai_reply") == ai_reply
    assert learned_data.get("ketu_correction") == ketu_correction

    print("TEST 3 PASSED: Wrong tone → Ketu edited casual → Digital Ketu learned style")


# ===========================================================================
# TEST 4: Missing Info — AI didn't mention MOQ/colors
# ===========================================================================

@pytest.mark.asyncio
async def test_4_missing_info_correction(client):
    """Scenario: Customer asks about polo t-shirts. AI gives price but
    forgets MOQ and available colors. Ketu edits with complete info.
    """
    phone = "919555666777"
    customer_msg = "Polo t-shirt ke baare mein batao"
    ai_reply = "Ji bhai, polo t-shirt ₹220/piece hai."
    ketu_correction = "Bhai polo t-shirt ₹220/piece, MOQ 100 pieces. Colors: white, black, navy, grey, maroon. Sizes S to XXL available hai."

    _simulate_ai_sent_message(phone, ai_reply)
    _simulate_conversation_log(phone, "Manoj", customer_msg, ai_reply)

    learned_data = {}

    def mock_learn(**kwargs):
        learned_data.update(kwargs)
        return {
            "status": "learned",
            "what_went_wrong": "AI gave incomplete info — missing MOQ, colors, and sizes",
            "updates_applied": ["faq_added", "product_info_updated"],
            "count": 2,
        }

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_edit, \
         patch("learner.realtime_learner.learn_from_correction", side_effect=mock_learn):

        mock_edit.return_value = {"messages": [{"id": "wamid_edited_4"}]}

        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ketu_correction,
            "message_id": "wamid_test_123",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "edited"
    assert data["learning"] == "correction sent to learner"

    time.sleep(0.5)

    assert learned_data.get("ketu_correction") == ketu_correction
    assert "polo" in learned_data.get("customer_message", "").lower()

    print("TEST 4 PASSED: Missing info → Ketu added MOQ/colors → Digital Ketu learned complete answer")


# ===========================================================================
# TEST 5: No Learning Needed — Edit same text or no conversation found
# ===========================================================================

@pytest.mark.asyncio
async def test_5_no_learning_when_same_text(client):
    """Scenario: Ketu edits the message but the new text is the same as AI reply.
    OR no conversation log found. Digital Ketu should NOT trigger learning.
    """
    phone = "919444333222"
    ai_reply = "Ji bhai, available hai."

    _simulate_ai_sent_message(phone, ai_reply)
    # Simulate a conversation log
    _simulate_conversation_log(phone, "Test", "Available hai kya?", ai_reply)

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_edit, \
         patch("learner.realtime_learner.learn_from_correction") as mock_learn:

        mock_edit.return_value = {"messages": [{"id": "wamid_edited_5"}]}

        # Edit with SAME text as AI reply — no learning should happen
        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ai_reply,  # Same text!
            "message_id": "wamid_test_123",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "edited"
    # Should NOT trigger learning since text is the same
    assert data.get("learning") != "correction sent to learner"

    time.sleep(0.3)
    mock_learn.assert_not_called()

    # --- Part B: No conversation log for this phone ---
    phone2 = "919111000999"
    _simulate_ai_sent_message(phone2, "Some reply")
    # NO conversation log for phone2

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_edit2, \
         patch("learner.realtime_learner.learn_from_correction") as mock_learn2:

        mock_edit2.return_value = {"messages": [{"id": "wamid_edited_5b"}]}

        resp2 = await client.post("/api/whatsapp/edit", json={
            "phone": phone2,
            "new_text": "Corrected reply",
            "message_id": "wamid_test_123",
        })

    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["status"] == "edited"
    assert data2.get("learning") == "no matching conversation found"

    time.sleep(0.3)
    mock_learn2.assert_not_called()

    print("TEST 5 PASSED: Same text / no conversation → Digital Ketu skipped learning (correct)")
