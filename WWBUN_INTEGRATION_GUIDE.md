# wwbun + Digital Ketu Integration Instructions

## For Claude Code (paste this in wwbun Cowork session)

---

Mujhe wwbun app mein Digital Ketu AI auto-reply feature add karni hai. Digital Ketu ek alag service hai jo Railway pe deploy hogi (`DIGITAL_KETU_URL` env var mein URL hoga). Jab AI mode ON ho, incoming WhatsApp messages ka reply Digital Ketu se aaye.

**IMPORTANT**: Voice notes (audio messages) bhi handle karne hain — audio file read karke base64 mein Digital Ketu ko bhejni hai, woh Whisper se transcribe karke reply dega.

## Kya karna hai:

### 1. `server/index.js` mein changes

**A) Top of file mein (imports ke baad, around line 22):**
Add Digital Ketu config:

```javascript
// Digital Ketu AI Auto-Reply
const DIGITAL_KETU_URL = process.env.DIGITAL_KETU_URL || ''
let digitalKetuEnabled = process.env.DIGITAL_KETU_ENABLED === 'true'
```

**B) New functions add karo (anywhere before `processIncomingMessage` function, around line 3300):**

```javascript
// ============ DIGITAL KETU AI REPLY ============

/**
 * Send message to Digital Ketu and get AI reply.
 * For audio messages, include audio_base64 so Digital Ketu can transcribe.
 */
async function getDigitalKetuReply(payload) {
  if (!DIGITAL_KETU_URL || !digitalKetuEnabled) return null

  try {
    const response = await axios.post(`${DIGITAL_KETU_URL}/api/reply`, payload, {
      timeout: 45000,  // 45s — Whisper transcription takes a few seconds for audio
      maxContentLength: 50 * 1024 * 1024,  // 50MB for audio base64
    })

    // Check should_reply flag — false means skip (conversation ender like "ok", "thanks")
    if (response.data?.should_reply === false) return null
    return response.data?.reply || null
  } catch (error) {
    console.error('[Digital Ketu] Error:', error.message)
    return null
  }
}

/**
 * Read audio file from local /uploads dir and convert to base64.
 * Returns base64 string or null if file not found.
 */
async function getAudioBase64(mediaUrl) {
  if (!mediaUrl) return null
  try {
    // mediaUrl is like "/uploads/filename.mp3" — resolve to absolute path
    const filePath = mediaUrl.startsWith('/') ? `.${mediaUrl}` : mediaUrl
    const fs = require('fs').promises
    const fileBuffer = await fs.readFile(filePath)
    console.log(`[Digital Ketu] Read audio file: ${filePath} (${fileBuffer.length} bytes)`)
    return fileBuffer.toString('base64')
  } catch (error) {
    console.error(`[Digital Ketu] Could not read audio file ${mediaUrl}:`, error.message)
    return null
  }
}
```

**C) `processIncomingMessage` function mein — Digital Ketu auto-reply block:**

Find the place in `processIncomingMessage` where the incoming message has been saved to the database and the conversation/contact objects are available. Add this block AFTER the message is saved:

```javascript
  // ========== DIGITAL KETU AI AUTO-REPLY ==========
  // Handle TEXT messages and AUDIO (voice notes) — transcribe audio via Whisper
  const msgType = message.type  // 'text', 'audio', 'image', etc.
  const isTextOrAudio = (msgType === 'text' || msgType === 'audio')

  if (digitalKetuEnabled && DIGITAL_KETU_URL && isTextOrAudio) {
    try {
      // Get recent conversation history for context
      const recentMessages = await db.message.findMany({
        where: { conversationId: conversation.id },
        orderBy: { createdAt: 'desc' },
        take: 10,
        select: { content: true, contactId: true, createdAt: true }
      })

      // Format history for Digital Ketu (oldest first)
      const history = recentMessages.reverse().map(m => ({
        role: m.contactId === contact.id ? 'user' : 'assistant',
        content: m.content
      }))

      // Build request payload
      const payload = {
        message: msgType === 'audio' ? '[audio]' : (message.text?.body || content || ''),
        customer_phone: contact.whatsappNumber || message.from,
        customer_name: contact.name || '',
        conversation_history: history
      }

      // For AUDIO messages: read the saved audio file and send as base64
      // Digital Ketu will transcribe using OpenAI Whisper and reply
      if (msgType === 'audio') {
        // The message was just saved to DB — find it to get mediaUrl
        const savedMsg = await db.message.findFirst({
          where: { conversationId: conversation.id },
          orderBy: { createdAt: 'desc' },
          select: { mediaUrl: true }
        })

        if (savedMsg?.mediaUrl) {
          const audioBase64 = await getAudioBase64(savedMsg.mediaUrl)
          if (audioBase64) {
            payload.audio_base64 = audioBase64
            console.log(`[Digital Ketu] Sending voice note for transcription (${audioBase64.length} chars base64)`)
          } else {
            console.log('[Digital Ketu] Could not read audio file, sending without audio data')
          }
        }

        // Also try sending media_id in case Digital Ketu has WhatsApp access token
        if (message.audio?.id) {
          payload.media_id = message.audio.id
        }
      }

      console.log(`[Digital Ketu] Getting AI reply for ${contact.whatsappNumber || message.from}... (type: ${msgType})`)
      const aiReply = await getDigitalKetuReply(payload)

      if (aiReply) {
        // Send reply via WhatsApp
        await sendOutboundMessage({
          contactId: contact.id,
          content: aiReply,
          type: 'text',
          metadata: { ai_generated: true, source: 'digital-ketu' }
        })
        console.log(`[Digital Ketu] AI reply sent to ${contact.whatsappNumber}: ${aiReply.substring(0, 50)}...`)
      }
    } catch (ketuError) {
      console.error('[Digital Ketu] Auto-reply error:', ketuError.message)
    }
  }
```

**IMPORTANT NOTES about the code above:**
- `message.type` uses LOWERCASE ('text', 'audio') — this is what WhatsApp API sends
- `sendOutboundMessage` — check the actual function name and signature in your codebase. It might be different. The metadata `{ ai_generated: true, source: 'digital-ketu' }` prevents learning loops.
- `conversation` and `contact` — these should already be available from earlier in processIncomingMessage
- The audio file is read from the `/uploads` directory where wwbun already saves it

**D) API endpoints add karo — settings section mein (jahan indiamart settings hain uske paas):**

```javascript
// ============ DIGITAL KETU SETTINGS ============
app.get('/api/settings/digital-ketu', requireAuth, async (c) => {
  return c.json({
    enabled: digitalKetuEnabled,
    url: DIGITAL_KETU_URL ? '***configured***' : 'not configured'
  })
})

app.post('/api/settings/digital-ketu/toggle', requireAuth, async (c) => {
  const { enabled } = await c.req.json()
  digitalKetuEnabled = !!enabled
  console.log(`[Digital Ketu] ${digitalKetuEnabled ? 'ENABLED' : 'DISABLED'}`)
  return c.json({ enabled: digitalKetuEnabled })
})
```

### 2. Environment Variables

Railway mein yeh env vars add karo:
```
DIGITAL_KETU_URL=https://digital-ketu-production.up.railway.app
DIGITAL_KETU_ENABLED=true
```

### 3. Frontend mein toggle (Optional — baad mein)

Settings page mein ek toggle button add karna chahiye "Digital Ketu AI Reply" ke liye jo `/api/settings/digital-ketu/toggle` ko call kare.

---

## How it works:

### Text Messages:
1. Customer WhatsApp pe text message bhejta hai
2. wwbun message receive karta hai, save karta hai (as usual)
3. **NEW**: Agar Digital Ketu enabled hai → message Digital Ketu ko bhejta hai
4. Digital Ketu Claude AI se reply generate karta hai (Ketu ki style mein)
5. Reply wwbun ko wapas aata hai
6. wwbun customer ko WhatsApp pe reply bhejta hai

### Voice Notes (Audio):
1. Customer voice note bhejta hai
2. wwbun audio download karta hai, `/uploads` mein save karta hai (as usual)
3. **NEW**: wwbun audio file read karta hai, base64 mein convert karta hai
4. Digital Ketu ko bhejta hai: `{message: "[audio]", audio_base64: "..."}`
5. Digital Ketu Whisper API se transcribe karta hai (Hindi/English)
6. Transcribed text se reply generate karta hai
7. wwbun customer ko reply bhejta hai

### Digital Ketu `/api/reply` Endpoint:
```
POST /api/reply
{
  "message": "text message" or "[audio]",
  "customer_phone": "91XXXXXXXXXX",
  "customer_name": "Customer Name",
  "conversation_history": [...],
  "audio_base64": "base64...",     // Optional: for voice notes
  "audio_url": "https://...",      // Optional: direct URL to audio
  "media_id": "whatsapp_media_id"  // Optional: WhatsApp media ID
}
```

## Important:
- Jab aap manually type karo → woh normal kaam karega (Digital Ketu interfere nahi karega)
- Digital Ketu sirf incoming customer messages ka reply karega
- Voice notes ko Whisper se transcribe karke reply dega
- Toggle se ON/OFF kar sakte ho
- `DIGITAL_KETU_URL` env var mein Railway URL dalna hai
- AI-generated replies have metadata `{ source: 'digital-ketu' }` to prevent learning loops

---

## Part 2: Learning Sync Hook (CRITICAL — Ketu ki style seekhne ke liye)

Jab Ketu (owner) manually WhatsApp pe reply karta hai, woh messages Digital Ketu ko bhejne hain
taaki AI seekh sake — pricing, negotiation style, customer handling, naye products, etc.

### Kya karna hai:

**E) New function add karo — `getDigitalKetuReply` ke neeche:**

```javascript
// ============ DIGITAL KETU LEARNING SYNC ============
async function syncManualMessageToDigitalKetu(conversationId, ownerUserId) {
  if (!DIGITAL_KETU_URL) return

  try {
    // Get last 20 messages from this conversation for context
    const recentMessages = await db.message.findMany({
      where: { conversationId },
      orderBy: { createdAt: 'desc' },
      take: 20,
      include: { contact: { select: { whatsappNumber: true } } }
    })

    if (!recentMessages.length) return

    // Format messages for Digital Ketu learning
    // is_owner: true when Ketu sent the message (no contactId = owner message)
    const formattedMessages = recentMessages.reverse().map(m => ({
      sender_id: m.contactId ? m.contact?.whatsappNumber : ownerUserId,
      content: m.content || '',
      is_ai_generated: m.metadata?.ai_generated === true || m.metadata?.source === 'digital-ketu',
      is_owner: !m.contactId,  // Owner messages have no contactId
    }))

    await axios.post(`${DIGITAL_KETU_URL}/api/learn/wwbun-sync`, {
      messages: formattedMessages,
      owner_user_id: ownerUserId
    }, { timeout: 10000 })

    console.log('[Digital Ketu] Learning sync sent — owner manual message learned')
  } catch (error) {
    // Silent fail — learning sync is non-critical, don't break message flow
    console.error('[Digital Ketu] Learning sync error:', error.message)
  }
}
```

**F) Hook into outbound message sending — jab owner manually message bheje:**

Find the place in `server/index.js` where outbound/sent messages are saved to database
(typically in `sendOutboundMessage` or the API endpoint that handles owner sending messages).

After the message is saved to DB, add this:

```javascript
// After owner's manual message is saved to DB:
// Sync to Digital Ketu for learning (non-blocking)
if (DIGITAL_KETU_URL && !metadata?.ai_generated && !metadata?.source?.includes('digital-ketu')) {
  // This is a manual message from owner — sync for learning
  syncManualMessageToDigitalKetu(conversationId, ownerUserId).catch(() => {})
}
```

**KEY POINTS:**
- `.catch(() => {})` ensures it never blocks the main message flow
- Only syncs manual messages (NOT AI-generated ones) — prevents learning loop
- Sends last 20 messages for conversation context
- Digital Ketu extracts: pricing style, negotiation patterns, new product info, FAQ answers

### How Learning Works:

```
Ketu manually types reply on WhatsApp
  → wwbun saves message to DB (as usual)
  → wwbun calls Digital Ketu /api/learn/wwbun-sync (NEW)
    → Digital Ketu analyzes Ketu's message
    → Extracts: style patterns, prices, FAQs, business knowledge
    → Updates: prompt.json, faq.json, style.json automatically
    → Dashboard shows "WhatsApp Sync: Last sync 2s ago"
```

### What Digital Ketu Learns From Manual Messages:

1. **Pricing** — "bhai 180 lagega bulk mein" → price list update
2. **Negotiation style** — "sample rate alag hoga, bulk mein discount milega" → style pattern
3. **New FAQs** — customer asks question, Ketu answers → auto-added to FAQ
4. **Tone & phrases** — "Ji bilkul", "Aap order place karo" → signature phrases
5. **Product knowledge** — "naya 280 GSM oversized aaya hai" → product catalog update
