# wwbun + Digital Ketu Integration Instructions

## For Claude Code (paste this in wwbun Cowork session)

---

Mujhe wwbun app mein Digital Ketu AI auto-reply feature add karni hai. Digital Ketu ek alag service hai jo Railway pe deploy hogi (`DIGITAL_KETU_URL` env var mein URL hoga). Jab AI mode ON ho, incoming WhatsApp messages ka reply Digital Ketu se aaye.

## Kya karna hai:

### 1. `server/index.js` mein changes

**A) Top of file mein (imports ke baad, around line 22):**
Add Digital Ketu config:

```javascript
// Digital Ketu AI Auto-Reply
const DIGITAL_KETU_URL = process.env.DIGITAL_KETU_URL || ''
let digitalKetuEnabled = process.env.DIGITAL_KETU_ENABLED === 'true'
```

**B) New function add karo (anywhere before `processIncomingMessage` function, around line 3300):**

```javascript
// ============ DIGITAL KETU AI REPLY ============
async function getDigitalKetuReply(message, customerPhone, customerName, conversationHistory = []) {
  if (!DIGITAL_KETU_URL || !digitalKetuEnabled) return null

  try {
    const response = await axios.post(`${DIGITAL_KETU_URL}/api/reply`, {
      message,
      customer_phone: customerPhone,
      customer_name: customerName,
      conversation_history: conversationHistory
    }, { timeout: 30000 })

    return response.data?.reply || null
  } catch (error) {
    console.error('[Digital Ketu] Error:', error.message)
    return null
  }
}
```

**C) `processIncomingMessage` function mein — line ~3793 ke baad (welcome auto-reply logic ke baad, `} else { console.log('[Auto-Reply] Skipped: Conversation already exists') }` ke baad):**

Add this Digital Ketu AI reply block:

```javascript
  // ========== DIGITAL KETU AI AUTO-REPLY ==========
  if (digitalKetuEnabled && DIGITAL_KETU_URL && content && messageType === 'TEXT') {
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

      console.log(`[Digital Ketu] Getting AI reply for ${contact.whatsappNumber}...`)
      const aiReply = await getDigitalKetuReply(content, contact.whatsappNumber, contact.name || '', history)

      if (aiReply) {
        await sendOutboundMessage({
          contactId: contact.id,
          content: aiReply,
          type: 'text'
        })
        console.log(`[Digital Ketu] AI reply sent to ${contact.whatsappNumber}: ${aiReply.substring(0, 50)}...`)
      }
    } catch (ketuError) {
      console.error('[Digital Ketu] Auto-reply error:', ketuError.message)
    }
  }
```

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
DIGITAL_KETU_URL=https://your-digital-ketu.railway.app
DIGITAL_KETU_ENABLED=false
```

(Initially `false` rakhna, testing ke baad `true` karna)

### 3. Frontend mein toggle (Optional — baad mein)

Settings page mein ek toggle button add karna chahiye "Digital Ketu AI Reply" ke liye jo `/api/settings/digital-ketu/toggle` ko call kare.

---

## How it works:

1. Customer WhatsApp pe message bhejta hai
2. wwbun message receive karta hai, save karta hai (as usual)
3. **NEW**: Agar Digital Ketu enabled hai → message Digital Ketu ko bhejta hai
4. Digital Ketu Claude AI se reply generate karta hai (Ketu ki style mein)
5. Reply wwbun ko wapas aata hai
6. wwbun customer ko WhatsApp pe reply bhejta hai

## Important:
- Jab aap manually type karo → woh normal kaam karega (Digital Ketu interfere nahi karega)
- Digital Ketu sirf incoming customer messages ka reply karega
- Toggle se ON/OFF kar sakte ho
- `DIGITAL_KETU_URL` env var mein Railway URL dalna hai

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
    const formattedMessages = recentMessages.reverse().map(m => ({
      sender_id: m.contactId ? m.contact?.whatsappNumber : ownerUserId,
      content: m.content || '',
      is_ai_generated: m.metadata?.ai_generated === true || m.metadata?.source === 'digital-ketu'
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
