# wwbun WhatsApp Features — Summary for Implementation

## Context

Digital Ketu is a **reply bot only** — it generates AI replies and learns from Ketu's manual chats. All WhatsApp-specific features (editing, reactions, read receipts, replying to specific messages) should be implemented in **wwbun**, not Digital Ketu.

This document lists WhatsApp Business API features that wwbun should support.

---

## 1. Message Editing (PUT Request)

WhatsApp Business API allows editing sent messages within **15 minutes** of sending.

**API Details:**
```
PUT https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages
Authorization: Bearer {ACCESS_TOKEN}
Content-Type: application/json

{
  "messaging_product": "whatsapp",
  "recipient_type": "individual",
  "to": "CUSTOMER_PHONE",
  "type": "text",
  "text": { "body": "Updated message text" },
  "context": { "message_id": "wamid.ORIGINAL_MESSAGE_ID" }
}
```

**Key points:**
- Uses **PUT** method (not POST)
- Requires `context.message_id` — the wamid of the original message to edit
- Customer sees the updated message with an **(edited)** label
- **15-minute window** — after that, edit fails
- Only text messages can be edited (not images, templates, etc.)

**wwbun implementation:**
- Store `wamid` (message_id) returned from every sent message
- Add "Edit" button in the chat UI for messages sent within last 15 minutes
- Show remaining edit time (countdown from 15 min)
- Call PUT with the original wamid + new text
- Update the message in wwbun's local DB too

**Digital Ketu integration (optional):**
- Digital Ketu already has edit endpoints if wwbun wants to route edits through it:
  - `POST /api/whatsapp/edit` — edit a specific message by wamid
  - `GET /api/whatsapp/last-sent/{phone}` — check if last AI reply is still editable
- Or wwbun can call WhatsApp API directly (recommended — wwbun already handles message sending)

---

## 2. Mark as Read (Read Receipts)

Mark incoming customer messages as "read" (blue ticks).

**API Details:**
```
POST https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages
Authorization: Bearer {ACCESS_TOKEN}
Content-Type: application/json

{
  "messaging_product": "whatsapp",
  "status": "read",
  "message_id": "wamid.INCOMING_MESSAGE_ID"
}
```

**Key points:**
- Send this when Ketu opens/reads a message in wwbun
- Customer sees blue ticks (double blue check marks)
- Can also be sent automatically when AI processes the message
- message_id is the wamid from the incoming webhook payload

**wwbun implementation:**
- When a conversation is opened in wwbun UI → mark all unread messages as read
- Optionally: auto-mark as read when Digital Ketu processes and replies

---

## 3. Reactions (Emoji Reactions)

React to customer messages with emojis (like regular WhatsApp).

**API Details:**
```
POST https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages
Authorization: Bearer {ACCESS_TOKEN}
Content-Type: application/json

{
  "messaging_product": "whatsapp",
  "recipient_type": "individual",
  "to": "CUSTOMER_PHONE",
  "type": "reaction",
  "reaction": {
    "message_id": "wamid.MESSAGE_TO_REACT_TO",
    "emoji": "👍"
  }
}
```

**To remove a reaction:**
```json
{
  "messaging_product": "whatsapp",
  "recipient_type": "individual",
  "to": "CUSTOMER_PHONE",
  "type": "reaction",
  "reaction": {
    "message_id": "wamid.MESSAGE_TO_REACT_TO",
    "emoji": ""
  }
}
```

**wwbun implementation:**
- Add emoji reaction picker on each message in chat UI
- Store reactions in DB
- Handle incoming reaction webhooks (customers can react to your messages too)

---

## 4. Reply to Specific Message (Quoted Reply)

Reply to a specific message in the conversation (shows the quoted original).

**API Details:**
```
POST https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages
Authorization: Bearer {ACCESS_TOKEN}
Content-Type: application/json

{
  "messaging_product": "whatsapp",
  "recipient_type": "individual",
  "to": "CUSTOMER_PHONE",
  "type": "text",
  "context": { "message_id": "wamid.MESSAGE_TO_REPLY_TO" },
  "text": { "body": "Your reply text" }
}
```

**Key points:**
- Add `context.message_id` to any outgoing message to make it a quoted reply
- Works with text, image, document — any message type
- Customer sees the original message quoted above the reply (like swipe-to-reply in regular WhatsApp)

**wwbun implementation:**
- Add "swipe to reply" or reply button on each message
- When replying to a specific message, include `context.message_id` in the API call
- Display quoted replies in the chat UI

---

## 5. Interactive Messages (Buttons & Lists)

Send messages with clickable buttons or list menus.

**Reply Buttons (max 3):**
```json
{
  "messaging_product": "whatsapp",
  "to": "CUSTOMER_PHONE",
  "type": "interactive",
  "interactive": {
    "type": "button",
    "body": { "text": "Which size do you want?" },
    "action": {
      "buttons": [
        { "type": "reply", "reply": { "id": "size_s", "title": "S" } },
        { "type": "reply", "reply": { "id": "size_m", "title": "M" } },
        { "type": "reply", "reply": { "id": "size_l", "title": "L" } }
      ]
    }
  }
}
```

**List Message (max 10 items per section):**
```json
{
  "messaging_product": "whatsapp",
  "to": "CUSTOMER_PHONE",
  "type": "interactive",
  "interactive": {
    "type": "list",
    "body": { "text": "Choose a product category:" },
    "action": {
      "button": "View Options",
      "sections": [
        {
          "title": "T-Shirts",
          "rows": [
            { "id": "oversized", "title": "Oversized", "description": "240 GSM" },
            { "id": "regular", "title": "Regular Fit", "description": "180 GSM" }
          ]
        }
      ]
    }
  }
}
```

**wwbun implementation:**
- Support sending button and list messages from the chat UI
- Handle interactive reply webhooks (`interactive.button_reply` and `interactive.list_reply`)
- Digital Ketu can suggest interactive messages in its reply (future feature)

---

## 6. Delete for Everyone — NOT Available

WhatsApp Business API does **NOT** support "Delete for Everyone". Once a message is sent, it cannot be deleted — only edited (within 15 minutes). This is an API limitation.

**Workaround:** Use message editing to replace wrong content with a correction.

---

## 7. Media Messages

Send images, documents, audio, video, stickers.

**Image example:**
```json
{
  "messaging_product": "whatsapp",
  "to": "CUSTOMER_PHONE",
  "type": "image",
  "image": {
    "link": "https://your-cdn.com/product-image.jpg",
    "caption": "Check out our new oversized collection!"
  }
}
```

**Document example:**
```json
{
  "messaging_product": "whatsapp",
  "to": "CUSTOMER_PHONE",
  "type": "document",
  "document": {
    "link": "https://your-cdn.com/catalog.pdf",
    "caption": "Product Catalog 2026",
    "filename": "catalog.pdf"
  }
}
```

---

## 8. "Ketu Only" Deferred Questions — wwbun Integration

Digital Ketu now has a **"Ketu Only"** system. When a customer asks something only the real Ketu can answer (stock timelines, order status, custom pricing), the AI sends a polite defer message like:

> "Bhai, ye Ketu sir khud confirm karenge — thodi der mein reply aayega."

**What wwbun should do:**

1. **Show deferred questions prominently** — When Digital Ketu returns `should_reply: true` with a defer message, wwbun should flag that conversation as "Needs Ketu's Reply" in the UI
2. **When Ketu manually replies** to a deferred conversation, sync that to Digital Ketu via `POST /api/learn/wwbun-sync` so the AI learns from it
3. **Optional: Mark as resolved** — Call `POST /api/ketu-only/resolve/{index}` when Ketu replies to a deferred question

**Digital Ketu API endpoints for this:**
- `GET /api/ketu-only/queue` — Get list of deferred questions + stats
- `POST /api/ketu-only/resolve/{index}` — Mark a deferred question as resolved
- `GET /api/ketu-only/categories` — Get ketu-only categories and learned patterns count

---

## Priority Order for Implementation

1. **Message Editing** — Ketu specifically requested this. Most useful.
2. **Mark as Read** — Simple to implement, good UX.
3. **Reply to Specific Message** — Important for context in busy conversations.
4. **Reactions** — Nice to have, quick acknowledgment.
5. **Interactive Buttons/Lists** — Future enhancement for product catalog browsing.
6. **Media Messages** — If not already supported.

---

## WhatsApp Business API Reference

- Base URL: `https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages`
- Auth: `Authorization: Bearer {ACCESS_TOKEN}`
- Docs: https://developers.facebook.com/docs/whatsapp/cloud-api/messages
