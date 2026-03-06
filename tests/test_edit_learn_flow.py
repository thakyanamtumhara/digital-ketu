"""5 End-to-End Test Scenarios: Edit → Save → Learn (NO send to buyer)

Tests the complete flow:
1. Customer sends message
2. AI replies (possibly wrong)
3. Ketu edits the AI reply
4. Correction is SAVED locally — NOT sent/edited to buyer on WhatsApp
5. Digital Ketu learns from the correction

KEY BEHAVIOR: The buyer never receives a second message or edit.
Digital Ketu learns silently and gives better replies next time.
"""

import json
import time
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
    from main import app as fastapi_app
    return fastapi_app


@pytest_asyncio.fixture
async def client(app):
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
# TEST 1: Wrong Price — Ketu corrects, buyer gets NOTHING, DK learns
# ===========================================================================

@pytest.mark.asyncio
async def test_1_wrong_price_no_send_to_buyer(client):
    """Customer asks rate, AI gives wrong price ₹170.
    Ketu edits to ₹190. Buyer should NOT receive any message.
    Digital Ketu should learn the correct price.
    """
    phone = "919876543210"
    customer_msg = "Bhai round neck t-shirt ka rate kya hai?"
    ai_reply = "Ji bhai, round neck t-shirt ₹170/piece hai. MOQ 50 pieces."
    ketu_correction = "Bhai rate ₹190/piece hai, MOQ 50 pieces."

    _simulate_ai_sent_message(phone, ai_reply)
    _simulate_conversation_log(phone, "Ravi", customer_msg, ai_reply)

    learned_data = {}

    def mock_learn(**kwargs):
        learned_data.update(kwargs)
        return {
            "status": "learned",
            "what_went_wrong": "AI quoted wrong price ₹170 instead of ₹190",
            "updates_applied": ["faq_updated"],
            "count": 1,
        }

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_wa_edit, \
         patch("integrations.whatsapp.sender.edit_last_message", new_callable=AsyncMock) as mock_wa_edit_last, \
         patch("integrations.whatsapp.sender.send_text_message", new_callable=AsyncMock) as mock_wa_send, \
         patch("learner.realtime_learner.learn_from_correction", side_effect=mock_learn):

        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ketu_correction,
        })

        # CRITICAL: No WhatsApp API calls should be made (no send, no edit)
        mock_wa_edit.assert_not_called()
        mock_wa_edit_last.assert_not_called()
        mock_wa_send.assert_not_called()

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "saved"
    assert data["sent_to_customer"] is False
    assert data["learning"] == "correction sent to learner"

    time.sleep(0.5)

    # Digital Ketu learned the correction
    assert learned_data.get("customer_message") == customer_msg
    assert learned_data.get("ketu_correction") == ketu_correction
    assert learned_data.get("customer_phone") == phone

    print("TEST 1 PASSED: Wrong price → saved + learned → buyer got NOTHING")


# ===========================================================================
# TEST 2: Wrong GSM — Save only, no WhatsApp edit, DK learns
# ===========================================================================

@pytest.mark.asyncio
async def test_2_wrong_gsm_no_send_to_buyer(client):
    """Customer asks hoodie GSM, AI says 300. Ketu corrects to 380.
    NO message/edit to buyer. Digital Ketu learns correct GSM.
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
            "what_went_wrong": "Wrong GSM (300 instead of 380)",
            "updates_applied": ["product_info_updated"],
            "count": 1,
        }

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_wa_edit, \
         patch("integrations.whatsapp.sender.edit_last_message", new_callable=AsyncMock) as mock_wa_edit_last, \
         patch("integrations.whatsapp.sender.send_text_message", new_callable=AsyncMock) as mock_wa_send, \
         patch("learner.realtime_learner.learn_from_correction", side_effect=mock_learn):

        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ketu_correction,
        })

        # NO WhatsApp calls
        mock_wa_edit.assert_not_called()
        mock_wa_edit_last.assert_not_called()
        mock_wa_send.assert_not_called()

    data = resp.json()
    assert data["status"] == "saved"
    assert data["sent_to_customer"] is False
    assert data["learning"] == "correction sent to learner"

    time.sleep(0.5)
    assert learned_data.get("ai_reply") == ai_reply
    assert learned_data.get("ketu_correction") == ketu_correction

    print("TEST 2 PASSED: Wrong GSM → saved + learned → buyer got NOTHING")


# ===========================================================================
# TEST 3: Wrong Tone — Save correction, no send, DK learns style
# ===========================================================================

@pytest.mark.asyncio
async def test_3_wrong_tone_no_send_to_buyer(client):
    """AI replies too formal English. Ketu corrects to casual Hinglish.
    Buyer should NOT get a second message. DK learns the style.
    """
    phone = "919112233445"
    customer_msg = "Shipping kitne din mein hoga?"
    ai_reply = "Dear customer, shipping will take approximately 3-5 business days."
    ketu_correction = "Bhai 3-4 din mein aa jayega, courier se bhejte hai."

    _simulate_ai_sent_message(phone, ai_reply)
    _simulate_conversation_log(phone, "Suresh", customer_msg, ai_reply)

    learned_data = {}

    def mock_learn(**kwargs):
        learned_data.update(kwargs)
        return {
            "status": "learned",
            "what_went_wrong": "Too formal English, should be casual Hinglish",
            "updates_applied": ["style_updated", "rule_added"],
            "count": 2,
        }

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_wa_edit, \
         patch("integrations.whatsapp.sender.edit_last_message", new_callable=AsyncMock) as mock_wa_edit_last, \
         patch("integrations.whatsapp.sender.send_text_message", new_callable=AsyncMock) as mock_wa_send, \
         patch("learner.realtime_learner.learn_from_correction", side_effect=mock_learn):

        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ketu_correction,
        })

        # NO WhatsApp calls — buyer not bothered
        mock_wa_edit.assert_not_called()
        mock_wa_edit_last.assert_not_called()
        mock_wa_send.assert_not_called()

    data = resp.json()
    assert data["status"] == "saved"
    assert data["sent_to_customer"] is False

    time.sleep(0.5)
    assert learned_data.get("ketu_correction") == ketu_correction

    print("TEST 3 PASSED: Wrong tone → saved + learned → buyer got NOTHING")


# ===========================================================================
# TEST 4: Missing Info — DK learns, buyer gets nothing
# ===========================================================================

@pytest.mark.asyncio
async def test_4_missing_info_no_send_to_buyer(client):
    """AI gives only price, forgets MOQ/colors/sizes.
    Ketu corrects with full info. Buyer NOT messaged again.
    """
    phone = "919555666777"
    customer_msg = "Polo t-shirt ke baare mein batao"
    ai_reply = "Ji bhai, polo t-shirt ₹220/piece hai."
    ketu_correction = "Bhai polo ₹220/piece, MOQ 100. Colors: white, black, navy, grey, maroon. S to XXL."

    _simulate_ai_sent_message(phone, ai_reply)
    _simulate_conversation_log(phone, "Manoj", customer_msg, ai_reply)

    learned_data = {}

    def mock_learn(**kwargs):
        learned_data.update(kwargs)
        return {
            "status": "learned",
            "what_went_wrong": "Missing MOQ, colors, sizes",
            "updates_applied": ["faq_added", "product_info_updated"],
            "count": 2,
        }

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_wa_edit, \
         patch("integrations.whatsapp.sender.edit_last_message", new_callable=AsyncMock) as mock_wa_edit_last, \
         patch("integrations.whatsapp.sender.send_text_message", new_callable=AsyncMock) as mock_wa_send, \
         patch("learner.realtime_learner.learn_from_correction", side_effect=mock_learn):

        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ketu_correction,
        })

        # NO WhatsApp calls
        mock_wa_edit.assert_not_called()
        mock_wa_edit_last.assert_not_called()
        mock_wa_send.assert_not_called()

    data = resp.json()
    assert data["status"] == "saved"
    assert data["sent_to_customer"] is False
    assert data["learning"] == "correction sent to learner"

    time.sleep(0.5)
    assert "polo" in learned_data.get("customer_message", "").lower()

    print("TEST 4 PASSED: Missing info → saved + learned → buyer got NOTHING")


# ===========================================================================
# TEST 5: Same text / no conversation — no learning, no send
# ===========================================================================

@pytest.mark.asyncio
async def test_5_no_learning_no_send_when_same_text(client):
    """Part A: Ketu submits same text as AI reply — no learning, no send.
    Part B: No conversation log found — no learning, no send.
    Both cases: buyer gets NOTHING.
    """
    # --- Part A: Same text ---
    phone = "919444333222"
    ai_reply = "Ji bhai, available hai."

    _simulate_ai_sent_message(phone, ai_reply)
    _simulate_conversation_log(phone, "Test", "Available hai kya?", ai_reply)

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_wa_edit, \
         patch("integrations.whatsapp.sender.edit_last_message", new_callable=AsyncMock) as mock_wa_edit_last, \
         patch("integrations.whatsapp.sender.send_text_message", new_callable=AsyncMock) as mock_wa_send, \
         patch("learner.realtime_learner.learn_from_correction") as mock_learn:

        resp = await client.post("/api/whatsapp/edit", json={
            "phone": phone,
            "new_text": ai_reply,  # SAME text
        })

        mock_wa_edit.assert_not_called()
        mock_wa_edit_last.assert_not_called()
        mock_wa_send.assert_not_called()

    data = resp.json()
    assert data["status"] == "saved"
    assert data["sent_to_customer"] is False
    assert "same text" in data["learning"].lower() or "no change" in data["learning"].lower()

    time.sleep(0.3)
    mock_learn.assert_not_called()

    # --- Part B: No conversation log ---
    phone2 = "919111000999"
    _simulate_ai_sent_message(phone2, "Some reply")

    with patch("integrations.whatsapp.sender.edit_message", new_callable=AsyncMock) as mock_wa_edit2, \
         patch("integrations.whatsapp.sender.edit_last_message", new_callable=AsyncMock) as mock_wa_edit_last2, \
         patch("integrations.whatsapp.sender.send_text_message", new_callable=AsyncMock) as mock_wa_send2, \
         patch("learner.realtime_learner.learn_from_correction") as mock_learn2:

        resp2 = await client.post("/api/whatsapp/edit", json={
            "phone": phone2,
            "new_text": "Corrected reply",
        })

        mock_wa_edit2.assert_not_called()
        mock_wa_edit_last2.assert_not_called()
        mock_wa_send2.assert_not_called()

    data2 = resp2.json()
    assert data2["status"] == "saved"
    assert data2["sent_to_customer"] is False
    assert "no matching conversation" in data2["learning"].lower()

    time.sleep(0.3)
    mock_learn2.assert_not_called()

    print("TEST 5 PASSED: Same text / no convo → no learning, no send → buyer got NOTHING")
