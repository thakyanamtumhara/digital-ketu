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
