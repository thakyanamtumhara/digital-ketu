# wwbun — Add Message Editing Feature

## What's needed

Add **message editing** in wwbun. WhatsApp Business API supports editing sent messages within **15 minutes**.

Everything else (Ketu-only categorization, learning, deferred questions) is handled by Digital Ketu — wwbun just sends data to Digital Ketu and Digital Ketu does the rest.

---

## Message Editing — API Details

**Method:** PUT (not POST)

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

## Key Points

- Uses **PUT** method (not POST like sending)
- Requires `context.message_id` — the `wamid` of the original sent message
- Customer sees the updated text with an **(edited)** label
- **15-minute window only** — after that, the API rejects the edit
- Only **text messages** can be edited (not images, templates, etc.)
- **Delete for Everyone is NOT available** in WhatsApp Business API — editing is the only way to fix a wrong message

## What wwbun needs to do

1. **Store `wamid`** — When any message is sent (via POST), the API returns a `wamid` in the response. Store this ID in the message record in DB.
2. **Add "Edit" button** — In the chat UI, show an edit option on sent messages that are less than 15 minutes old.
3. **Call PUT API** — When Ketu edits a message, call the PUT endpoint above with the original `wamid` + new text.
4. **Update local DB** — After successful edit, update the message content in wwbun's database too.
5. **Show edit time remaining** (optional) — Countdown from 15 min so Ketu knows how much time is left.

## Response format (from original send)

When you send a message via POST, the response contains the wamid you need to store:

```json
{
  "messaging_product": "whatsapp",
  "contacts": [{ "input": "91XXXXXXXXXX", "wa_id": "91XXXXXXXXXX" }],
  "messages": [{ "id": "wamid.HBgMOTE5ODcxxxxxx" }]
}
```

Store `messages[0].id` — that's the `wamid` needed for editing.
