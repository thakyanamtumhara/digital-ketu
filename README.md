# Digital Ketu 🤖

**AI Digital Twin** for Own Knitted Blank Wears (Sale91.com) — Tiruppur se direct, factory rate!

Digital Ketu is a 24/7 autonomous AI that replies to customers exactly like Ketu — in Hinglish, with direct pricing, respectful tone, and complete product knowledge.

## Features

- **WhatsApp Auto-Reply** — Receives messages via WhatsApp Business API and replies in Ketu's style
- **IndiaMART Lead Reply** — Auto-replies to IndiaMART leads via WhatsApp
- **YouTube Comment Reply** — Replies to product-related YouTube comments
- **AI Reply API** — REST API for wwbun integration (POST /api/reply)
- **Auto-Learning** — Learns from Ketu's manual WhatsApp chats and YouTube videos
- **Central Knowledge Base** — Products, prices, FAQ, company info in one place

## Architecture

```
Customer → WhatsApp Business API → Digital Ketu → Claude AI → Reply
                                        ↑
                                   Knowledge Base
                                  (products, style,
                                   FAQ, company)
```

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/thakyanamtumhara/digital-ketu.git
cd digital-ketu
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key from console.anthropic.com |
| `WHATSAPP_ACCESS_TOKEN` | WhatsApp Business API access token |
| `WHATSAPP_PHONE_NUMBER_ID` | Your WhatsApp Business phone number ID |
| `WHATSAPP_VERIFY_TOKEN` | Webhook verification token (you choose) |
| `WHATSAPP_APP_SECRET` | Facebook App secret for webhook signature |

### 3. Run

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at http://localhost:8000. API docs at http://localhost:8000/docs

### 4. Deploy on Railway

```bash
# Railway CLI
railway login
railway init
railway up
```

Or connect your GitHub repo to Railway — it auto-deploys from Dockerfile.

## API Endpoints

### Core
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/reply` | POST | Get AI reply (for wwbun integration) |
| `/api/toggle` | POST/GET | Enable/disable auto-reply |
| `/api/knowledge` | GET | View knowledge base |
| `/api/knowledge/reload` | GET | Force reload knowledge |

### Webhooks
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/webhook/whatsapp` | GET/POST | WhatsApp Business API webhook |
| `/webhook/indiamart` | POST | IndiaMART lead webhook |

### Learning
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/learn/whatsapp-export` | POST | Learn from WhatsApp chat export |
| `/api/learn/wwbun-sync` | POST | Learn from wwbun messages (daily sync) |
| `/api/learn/youtube` | POST | Learn from YouTube video transcript |
| `/api/youtube/check-comments` | POST | Check & reply to YouTube comments |

## wwbun Integration

Digital Ketu works with wwbun (your WhatsApp web app) in two ways:

### Option 1: Direct Webhook (Standalone)
Point your WhatsApp Business API webhook to Digital Ketu's URL directly.

### Option 2: API Integration (Recommended)
wwbun continues handling WhatsApp → when AI mode is ON, wwbun calls Digital Ketu:

```javascript
// In wwbun, when auto-reply is enabled:
const response = await fetch('https://your-digital-ketu.railway.app/api/reply', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: customerMessage,
    customer_phone: senderPhone,
    customer_name: senderName,
    conversation_history: recentMessages
  })
});
const { reply } = await response.json();
// Send reply via WhatsApp
```

## Knowledge Base

Edit JSON files in `knowledge/` to update Digital Ketu's brain:

- `products.json` — Product catalog, prices, GSM, MOQ
- `company.json` — Company info, shipping, payment terms
- `style.json` — Reply style rules, example conversations
- `faq.json` — Common Q&A pairs

After editing, call `/api/knowledge/reload` or restart the server.

## Auto-Learning

### From WhatsApp (Daily)
wwbun sends Ketu's manual messages to `/api/learn/wwbun-sync`. Digital Ketu learns:
- Reply style and patterns
- New product info and prices
- New FAQ from customer interactions

**Important**: Only Ketu's manually typed messages are used for learning. AI-generated replies are ignored.

### From YouTube
Post video URLs to `/api/learn/youtube`. Digital Ketu extracts:
- Product details from video transcript
- Pricing information
- Business knowledge and tips

## Tech Stack

- **Python 3.11** + **FastAPI**
- **Anthropic Claude API** (Sonnet) for AI responses
- **httpx** for async HTTP calls
- **Pydantic** for data validation
- **Docker** for deployment
