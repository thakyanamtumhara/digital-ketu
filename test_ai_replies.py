"""
Quick diagnostic test — calls generate_reply() directly with 5 different message types.
Tests if the AI engine is working without needing WhatsApp webhook.
"""
import sys
import os
import time
import logging

# Setup logging so we can see what's happening
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

# Load env
from dotenv import load_dotenv
load_dotenv()

from core.config import settings
from core.engine import generate_reply

# Check API key first
if not settings.anthropic_api_key:
    print("\n❌ PROBLEM FOUND: anthropic_api_key is EMPTY!")
    print("   This is why Digital Ketu is not replying.")
    print("   Set ANTHROPIC_API_KEY in your .env file or Railway environment variables.")
    sys.exit(1)
else:
    print(f"✅ Anthropic API key found: {settings.anthropic_api_key[:8]}...{settings.anthropic_api_key[-4:]}")

print(f"✅ Auto-reply enabled: {settings.auto_reply_enabled}")
print()

# 5 Test messages with different variations
tests = [
    {
        "name": "Test 1: SHORT message (Hi)",
        "message": "Hi",
        "phone": "919999900001",
    },
    {
        "name": "Test 2: PRODUCT INQUIRY (price question)",
        "message": "100 GSM tissue paper ka rate kya hai?",
        "phone": "919999900002",
    },
    {
        "name": "Test 3: SAMPLE REQUEST (quality question)",
        "message": "Kya aap pehle sample bhej sakte ho? Hume quality check karni hai before bulk order",
        "phone": "919999900003",
    },
    {
        "name": "Test 4: ENGLISH formal message",
        "message": "I am interested in purchasing tissue paper in bulk for my hotel chain. What are your minimum order quantities and pricing?",
        "phone": "919999900004",
    },
    {
        "name": "Test 5: HINGLISH casual (complaint tone)",
        "message": "Bhai last order mein quality acchi nahi thi, tissue bahut patla tha",
        "phone": "919999900005",
    },
]

results = []
for test in tests:
    print(f"{'='*60}")
    print(f"🔄 {test['name']}")
    print(f"   Message: \"{test['message']}\"")
    print()

    start = time.time()
    try:
        reply = generate_reply(
            message=test["message"],
            customer_phone=test["phone"],
            customer_name="Test Customer",
        )
        elapsed = time.time() - start

        if reply:
            print(f"   ✅ REPLY: \"{reply}\"")
            print(f"   ⏱️  Time: {elapsed:.1f}s")
            results.append(("PASS", test["name"], reply))
        else:
            print(f"   ⚠️  EMPTY REPLY (no response generated)")
            print(f"   ⏱️  Time: {elapsed:.1f}s")
            results.append(("EMPTY", test["name"], ""))
    except Exception as e:
        elapsed = time.time() - start
        print(f"   ❌ ERROR: {e}")
        print(f"   ⏱️  Time: {elapsed:.1f}s")
        results.append(("ERROR", test["name"], str(e)))
    print()

# Summary
print(f"{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
passed = sum(1 for r in results if r[0] == "PASS")
empty = sum(1 for r in results if r[0] == "EMPTY")
errors = sum(1 for r in results if r[0] == "ERROR")

for status, name, reply in results:
    icon = "✅" if status == "PASS" else "⚠️" if status == "EMPTY" else "❌"
    print(f"  {icon} {name}")
    if reply and status == "PASS":
        print(f"     → \"{reply}\"")
    elif status == "ERROR":
        print(f"     → Error: {reply[:80]}")

print()
print(f"  Passed: {passed}/5 | Empty: {empty}/5 | Errors: {errors}/5")

if errors > 0:
    print("\n⚠️  DIAGNOSIS: API errors detected. Check:")
    print("   1. Is ANTHROPIC_API_KEY valid and has credits?")
    print("   2. Is the API reachable from Railway?")
    print("   3. Check Railway logs for detailed error messages")
elif empty > 0:
    print("\n⚠️  DIAGNOSIS: Some replies were empty. Check:")
    print("   1. Conversation ender detection may be too aggressive")
    print("   2. Shutup cooldown may be active")
    print("   3. Rate limiting may be blocking")
else:
    print("\n✅ All 5 tests passed! AI engine is working correctly.")
    print("   If WhatsApp still isn't replying, check:")
    print("   1. Webhook URL is correct in Meta dashboard")
    print("   2. WHATSAPP_ACCESS_TOKEN is valid")
    print("   3. auto_reply_enabled is ON (send /status from your phone)")
