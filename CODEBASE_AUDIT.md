# Digital Ketu - Complete Codebase Audit & Documentation

**Date:** 09 March 2026
**Audited by:** Claude AI
**Project:** Digital Ketu - AI Digital Twin for Sale91.com (B2B Blank Wears, Tiruppur)

---

## Overall Summary (Hindi-English)

**Digital Ketu** ek AI-powered WhatsApp chatbot hai jo Ketu (Sale91.com ka owner) ka "digital twin" hai. Ye customer ke WhatsApp messages ka auto-reply deta hai Ketu ke style mein (Hinglish). Architecture: FastAPI backend + PostgreSQL + Claude API (Anthropic) + wwbun (WhatsApp gateway).

### Kya Karta Hai Ye System:
1. **Customer ko auto-reply** - WhatsApp pe customer message aaye → Claude AI se reply generate karo Ketu ke style mein
2. **Learning system** - Ketu ke manual replies se seekhta hai (style, phrases, FAQs)
3. **Correction learning** - Agar AI galat reply de, Ketu correct kare → AI seekh le
4. **YouTube learning** - Ketu ke YouTube videos se product knowledge extract kare
5. **Catalog sync** - GitHub se product catalog sync kare (prices, colors, etc.)
6. **Dashboard** - Live monitoring dashboard (HTML)

---

## Architecture Flow

```
Customer (WhatsApp)
    → wwbun (WhatsApp Gateway)
    → Digital Ketu API (/api/reply)
    → Claude API (Haiku/Sonnet)
    → Reply back to wwbun
    → Customer ko WhatsApp pe bheje
```

**Data Storage:**
- PostgreSQL (Railway) - Primary, survives deploys
- JSON files (knowledge/) - Fallback
- GitHub backup (git_persist) - Last resort

---

## File-by-File Function Audit

---

### 1. `main.py` (~3131 lines) - Main FastAPI Application

**Ye file kya karti hai:** Poora API server. Saare endpoints, learning buffer logic, wwbun sync, dashboard UI serve karta hai.

#### Startup & Config (Lines 1-141)

| Function/Block | Line | Kya Karta Hai |
|---|---|---|
| `lifespan()` | 78-120 | App startup — DB init, knowledge load, scheduler start, GitHub restore |
| `app = FastAPI(...)` | 123-128 | FastAPI app create with CORS middleware |
| CORS middleware | 131-137 | `allow_origins=["*"]` — sabko allow (production mein risky) |
| Router includes | 140-141 | WhatsApp webhook + YouTube handler mount |

#### Health & Status APIs (Lines 144-190)

| Endpoint | Line | Kya Karta Hai |
|---|---|---|
| `GET /api/health` | 147-153 | Simple health check — returns status + auto_reply toggle |
| `GET /api/health/detailed` | 156-181 | Detailed health — DB status, errors, scheduler status |
| `GET /api/errors` | 184-190 | Recent errors list for dashboard |

#### AI Reply API (Lines 193-362)

| Function | Line | Kya Karta Hai |
|---|---|---|
| `ReplyRequest` model | 196-204 | Input: message, phone, name, history, audio fields |
| `ReplyResponse` model | 207-212 | Output: reply, status, should_reply, escalation info |
| `POST /api/reply` | 215-362 | **MAIN ENDPOINT** — Customer message → AI reply generate kare |
| Audio handling | 228-298 | Voice note aaye → Whisper se transcribe → reply generate |
| Media skip | 297-298 | Image/video/sticker → silently skip (no reply) |
| Reply to quoted msg | 300-306 | `[Replying to: "..."]` prefix strip kare |
| Ender detection | 317-329 | "Ok", "Thanks" → should_reply=false, no reply |
| Conversation logging | 337-343 | Every AI reply DB mein log hota hai |

#### Toggle & Settings APIs (Lines 365-457)

| Endpoint | Line | Kya Karta Hai |
|---|---|---|
| `POST /api/toggle` | 372-382 | Auto-reply on/off toggle |
| `GET /api/toggle` | 385-390 | Current toggle status |
| `POST /api/followup/toggle` | 396-402 | Follow-up intelligence on/off (independent) |
| `GET/POST /api/leaves` | 423-456 | Leave management — Ketu busy/on-leave toh different reply |
| `DELETE /api/leaves/{id}` | 449-456 | Delete a leave entry |

#### Knowledge Management (Lines 459-586)

| Endpoint | Line | Kya Karta Hai |
|---|---|---|
| `GET /api/knowledge/reload` | 462-470 | Force reload knowledge from JSON/DB |
| `GET /api/knowledge/cleanup-enders` | 473-538 | Greetings ko ender list se hataao (bug fix endpoint) |
| `GET /api/knowledge` | 541-544 | Full knowledge base return karo |
| `GET /api/knowledge/learned-summary` | 547-586 | Dashboard ke liye learned patterns, rules, traits summary |

#### Learning APIs (Lines 589-1810)

| Function | Line | Kya Karta Hai |
|---|---|---|
| `POST /api/learn/whatsapp-export` | 597-631 | WhatsApp chat export se seekho (manual upload) |
| **wwbun Stats System** | 639-898 | Complete sync stats tracking — total/today/quality/junk counts |
| `_load_wwbun_stats()` | 660-680 | DB se stats load karo (persist across deploys) |
| `_save_wwbun_stats()` | 683-695 | Stats DB mein save karo |
| `_track_wwbun_sync()` | 698-804 | Sync event track karo — quality pairs, junk filtered, etc. |
| `GET /api/wwbun/stats` | 807-837 | Dashboard ke liye live sync stats |
| `GET /api/wwbun/debug` | 840-899 | Debug endpoint — raw buffer, message classification, pairs |

#### Conversation Pair Extraction (Lines 902-1206)

| Function | Line | Kya Karta Hai |
|---|---|---|
| `_extract_conversation_pairs()` | 902-1190 | **CRITICAL** — Customer→Ketu message pairs extract kare |
| Detection priority | 914-923 | 1. sender_id match, 2. is_owner flag, 3. word-count heuristic |
| Chat grouping | 925-1027 | chat_id se group, synthetic groups jab chat_id missing |
| Multi-message combining | 1059-1178 | Consecutive customer msgs combine karo (5 min gap = new thread) |
| `_learn_ketu_only_pairs()` | 1193-1205 | Manual replies se ketu-only patterns seekho |

#### Learning Buffer System (Lines 1208-1543)

| Function | Line | Kya Karta Hai |
|---|---|---|
| Buffer constants | 1213-1216 | MIN_PAIRS=20, COOLDOWN=1800s, FORCE=50 pairs |
| `_save_owner_user_id()` | 1220-1229 | Owner ID persist karo for buffer-status after redeploy |
| `_get_learning_buffer()` | 1249-1259 | DB se buffered messages load karo |
| `_save_learning_buffer()` | 1262-1269 | Buffer DB mein save karo |
| Fingerprint dedup | 1276-1341 | Duplicate messages detect karo (message_id or composite key) |
| `_safe_bool()` | 1344-1352 | wwbun ke string "true"/"false" handle karo |
| `_is_owner_message()` | 1355-1377 | Owner vs Customer detect — sender_id > is_owner flag > fallback |
| `_all_sender_ids_same()` | 1380-1385 | Broken data detect karo (same sender_id for all) |
| `_is_owner_field_reliable()` | 1388-1405 | is_owner field mixed hai ya sab same (unreliable) |
| `_count_quality_owner_messages()` | 1417-1447 | Threshold ke liye quality count — pairs or individual msgs |
| `_flush_learning_buffer()` | 1455-1543 | Buffer flush → Claude se knowledge extract → apply → clear |
| `POST /api/learn/wwbun-sync` | 1546-1810 | **MAIN SYNC ENDPOINT** — wwbun se messages aaye → buffer → learn |

#### More Learning APIs (Lines 1813-2598)

| Endpoint | Line | Kya Karta Hai |
|---|---|---|
| `POST /api/learn/clear-buffer` | 1817-1826 | Buffer clear (bad data ke liye) |
| `POST /api/learn/flush-buffer` | 1829-1848 | Manual flush — force Claude learning |
| `GET /api/learn/buffer-status` | 1851-1896 | Buffer status with debug info |
| `POST /api/learn/youtube` | 1904-1924 | YouTube video se seekho |
| `POST /api/learn/youtube-scan` | 1927-1935 | YouTube channel scan trigger |
| `POST /api/learn/youtube-backfill` | 1938-1954 | Old videos process karo (batch) |
| YouTube OAuth | 1965-2092 | OAuth 2.0 setup for YouTube Captions API |
| `GET /api/learned-files` | 2095-2145 | Learned files list with details |
| `POST /api/learn/catalog-sync` | 2195-2219 | GitHub se catalog sync karo |
| Correction History | 2222-2245 | Ring buffer (50 max) + DB persist |
| `POST /api/learn/correction` | 2259-2309 | AI reply galat → Ketu correct kare → seekho |
| `POST /api/learn/voice-note` | 2322-2376 | Voice note se seekho (Whisper transcribe) |
| `GET /api/conversations/recent` | 2382-2389 | Recent AI conversations |
| `POST /api/ketu-replied` | 2414-2507 | **SHUT UP MODE** — Ketu replied → AI silent for 10 min |
| `GET /api/learn/correction-history` | 2519-2555 | Corrections with cloud analysis |
| `GET /api/costs` | 2568-2576 | API cost tracking (USD + INR) |
| `GET /api/rate-limit/stats` | 2590-2598 | Rate limiting stats |

#### Dashboard & Monitoring (Lines 2601-3131)

| Endpoint | Line | Kya Karta Hai |
|---|---|---|
| `GET /api/faq/health` | 2604-2610 | FAQ health report |
| `POST /api/faq/validate` | 2613-2637 | FAQ validate against catalog |
| `GET /api/dashboard` | 2659-2668 | Live dashboard data |
| `GET /api/insights/customers` | 2733-2736 | Customer analytics |
| `GET /api/customer/{phone}` | 2777-2781 | Customer memory profile |
| `GET /api/ketu-only/queue` | 3009-3020 | Questions deferred to real Ketu |
| `GET /api/backup` | 3049-3077 | Knowledge base ZIP download |
| `GET /api/cloud-payloads` | 3106-3111 | Claude API payload debug |
| `GET /` (Dashboard UI) | 3119-3122 | HTML dashboard serve |

---

### 2. `core/engine.py` (~1100 lines) - AI Reply Engine

**Ye file kya karti hai:** Core AI reply generation. Claude API call, conversation management, confidence scoring, ender detection, model selection.

| Function | Line | Kya Karta Hai |
|---|---|---|
| `_get_leave_aware_defer_reply()` | 30-48 | Ketu leave pe hai → deferral message change karo |
| `_conversations` dict | 52-54 | In-memory conversation history (phone → messages, 1hr TTL) |
| `_customer_message_counts` etc. | 57-68 | Customer insights tracking (in-memory + DB persist) |
| `_load_customer_insights_from_db()` | 109-137 | DB se customer insights load karo (first access only) |
| `_save_customer_insights_to_db()` | 140-159 | Insights DB mein save karo |
| `_load_prompt_config()` | 162-184 | Prompt config load — DB first, JSON fallback |
| `_load_repeat_buyer_style()` | 187-206 | Repeat buyer reply examples load karo |
| `_build_system_prompt()` | 216-332 | **CRITICAL** — System prompt build: STATIC (cached) + DYNAMIC |
| Static prompt | 235-257 | Ketu's personality, rules, style — same for every reply, cached |
| Dynamic prompt | 260-322 | Knowledge context + customer memory + escalation — per message |
| Smart context | 267-278 | Intent classify → relevant knowledge only (saves tokens) |
| Prompt caching | 271-278 | Same intent combo = cached knowledge context (60s TTL) |
| `_cleanup_old_conversations()` | 335-343 | 1 hour se purani conversations delete karo |
| `activate_shutup()` | 356-369 | Customer ke liye AI silent — ender/ketu_reply/ketu_only |
| `is_shutup_active()` | 372-385 | Check if AI should stay silent |
| `ketu_manual_reply()` | 388-401 | Ketu manually replied → 10 min shutup + track activity |
| `_load_ender_patterns()` | 404-480 | Ender patterns load: hardcoded + learned - non_enders - never_enders |
| `_is_conversation_ender()` | 490-524 | "Ok", "Thanks", "Theek hai" → don't reply |
| `_is_weak_reply()` | 527-552 | Weak/generic AI reply detect karo (retry with Sonnet) |
| `generate_reply()` | 555-1066 | **MAIN FUNCTION** — Full reply generation pipeline |
| → Insights tracking | 582-595 | Every message count karo (hourly + per customer) |
| → Shutup check | 598-634 | Cooldown active? Skip. New question? Break cooldown |
| → Ender detection | 636-679 | Customer ender → empty reply, activate shutup |
| → Ketu-Only check | 681-722 | Stock/order/custom pricing → defer to real Ketu |
| → Greeting fast-path | 724-742 | "Hi", "Hello" → never defer, always reply |
| → Confidence scoring | 744-799 | Low confidence → defer to Ketu |
| → Peak hours defer | 802-827 | Ketu online + borderline confidence → defer |
| → History trim | 831-837 | Last 3 messages only (save tokens) |
| → Escalation detect | 848-873 | Complaint/anger → flag for Ketu + use Sonnet |
| → System prompt build | 877-908 | Context, language, length constraint, first msg detection |
| → Model selection | 910-932 | Haiku (default, cheap) vs Sonnet (complex/high-value) |
| → Claude API call | 946-1066 | 3 attempts with backoff. max_tokens=45. Prompt caching. |
| → Cut-off fix | 960-970 | Truncated reply → trim to last complete sentence |
| → Cost tracking | 984-1008 | INR cost calculate, alert if >Rs.1 per reply |
| → Haiku fallback | 1011-1041 | Haiku weak reply → retry with Sonnet |
| `track_faq_hit()` | 1085-1096 | FAQ usage count karo |
| `track_wwbun_insights()` | 1099+ | wwbun se customer insights track karo (total messages) |
| `get_customer_insights()` | - | Dashboard ke liye customer analytics |
| `get_faq_hit_rates()` | - | FAQ hit rates return karo |
| `get_last_escalation()` | - | Last escalation for a customer |

---

### 3. `core/config.py` (69 lines) - Configuration

| Item | Line | Kya Karta Hai |
|---|---|---|
| `KNOWLEDGE_DIR` | 10 | `knowledge/` directory path |
| `LEARNED_DIR` | 11 | `knowledge/learned/` directory path |
| `init_knowledge_dir()` | 14-18 | Directories create karo on startup |
| `Settings` class | 21-66 | Pydantic settings — all env vars: API keys, DB URL, rate limits |
| `settings` singleton | 68 | Global settings instance (.env se load) |

**Key Settings:**
- `anthropic_api_key` - Claude API
- `openai_api_key` - Whisper (voice notes)
- `youtube_api_key` + OAuth tokens - YouTube learning
- `database_url` - PostgreSQL
- `admin_phone` - Ketu's number
- `auto_reply_enabled` / `followup_enabled` - Feature toggles
- Rate limits: 10/customer/hr, 100 global/hr, 3 at night

---

### 4. `core/database.py` (~350 lines) - PostgreSQL Layer

| Function | Line | Kya Karta Hai |
|---|---|---|
| `get_connection()` | 30-46 | DB connection get/create (singleton) |
| `_reconnect()` | 49-53 | Force reconnect on error |
| `_execute()` | 56-84 | Query execute with auto-reconnect |
| `init_db()` | 87-135 | Tables create: knowledge_store, activity_log, learned_files, kv_store |
| `save_knowledge()` | 145-152 | Knowledge JSON save (UPSERT) |
| `load_knowledge_from_db()` | 155-164 | Single knowledge key load |
| `load_all_knowledge_from_db()` | 167-172 | All knowledge load |
| `save_activity()` | 177-192 | Activity log entry save |
| `get_activity_from_db()` | 195-230 | Activity log query with filters |
| `kv_get()` / `kv_set()` | - | Key-value store operations |
| `seed_from_json_files()` | - | First-run: JSON files se DB seed karo |

**Tables:**
- `knowledge_store` (key TEXT, data JSONB) - FAQ, products, style, prompt
- `activity_log` (source, action, details JSONB) - All events
- `learned_files` (filename, content TEXT) - YouTube extracts etc.
- `kv_store` (key TEXT, value JSONB) - Internal state (stats, buffers)

---

### 5. `core/knowledge.py` (~150 lines) - Knowledge Base Loader

| Function | Line | Kya Karta Hai |
|---|---|---|
| `load_knowledge()` | 14-75 | Knowledge load: JSON + DB overlay + merge learned fields |
| `_load_learned_knowledge()` | 78+ | Learned files from DB/disk load karo |
| `format_context()` | - | Full knowledge context format for prompt |
| `invalidate_cache()` | - | Cache clear (after learning) |

**Merge Logic:** DB overwrites JSON, but learned list fields (learned_patterns, evolved_rules, evolved_traits, evolved_phrases, example_conversations) are MERGED not overwritten.

---

### 6. `core/context_selector.py` - Smart Context Selection

| Function | Kya Karta Hai |
|---|---|
| `classify_message()` | Customer message classify karo — intents + product_ids |
| `format_smart_context()` | Relevant knowledge only select karo (vs full dump) |

**Intents:** greeting, price_product, return_complaint, dropshipping, shipping_delivery, payment, order_how, location_visit, gst_invoice, moq, printing, gsm_fabric

---

### 7. `core/confidence.py` - Confidence Scoring

| Function | Kya Karta Hai |
|---|---|
| `score_confidence()` | Message pe confidence score (0-100) |
| High confidence | FAQ match, known product, simple greeting |
| Low confidence | Unknown topic, multiple questions, ambiguous |
| `CONFIDENCE_DEFER_THRESHOLD` | Below this → defer to Ketu |

---

### 8. `core/customer_memory.py` - Customer Profiles

| Function | Kya Karta Hai |
|---|---|
| `get_profile()` | Customer profile from DB (interests, stage, language) |
| `update_profile()` | Update after each message (auto-detect language, interests) |
| `format_customer_context()` | Profile → prompt context string |
| `get_all_profiles_summary()` | All customers summary for dashboard |

**Customer Stages:** new → interested → repeat → bought

---

### 9. `core/escalation.py` - Escalation Detection

| Function | Kya Karta Hai |
|---|---|
| `detect_escalation()` | Complaint/anger detect karo (keywords + sentiment) |
| `format_escalation_notice()` | Ketu ke liye alert format |
| `LEVEL_ESCALATE` | Threshold for escalation |

---

### 10. `core/ketu_only.py` - Ketu-Only Question Detection

| Function | Kya Karta Hai |
|---|---|
| `detect_ketu_only()` | Questions AI shouldn't answer — stock, order status, custom pricing |
| `log_deferred_question()` | Deferred question queue mein add |
| `log_manual_takeover()` | Ketu manually replied → log karo |
| `get_deferred_queue()` | Pending questions for Ketu |
| `mark_resolved()` | Ketu ne answer kiya → resolved |

**Categories:** stock_timeline, order_status, custom_pricing, payment_issues, delivery_tracking, godown_instructions

---

### 11. `core/token_budget.py` - Token Budget Management

| Function | Kya Karta Hai |
|---|---|
| `estimate_tokens()` | Text → approximate token count |
| `truncate_to_budget()` | Text ko budget mein fit karo |
| `truncate_history()` | Conversation history truncate |
| `log_budget_usage()` | Token usage log karo |

**Budgets:** Knowledge ~2500 tokens, History ~400 tokens, Total input ~4000 tokens

---

### 12. `core/cost_tracker.py` - API Cost Tracking

| Function | Kya Karta Hai |
|---|---|
| `track_api_cost()` | Every Claude API call ka cost track (Haiku vs Sonnet, cache savings) |
| `get_cost_summary()` | Today/week/month costs in USD + INR |

**Cost per reply:** Haiku ~Rs.0.05-0.08, Sonnet ~Rs.0.30-0.40

---

### 13. `core/reply_length.py` - Reply Length Auto-Constraint

| Function | Kya Karta Hai |
|---|---|
| `track_ai_reply()` | AI reply length track karo |
| `track_ketu_reply()` | Ketu's manual reply length track |
| `get_length_constraint()` | AI too long vs Ketu → constraint string for prompt |

---

### 14. `core/peak_hours.py` - Peak Hours Detection

| Function | Kya Karta Hai |
|---|---|
| `track_ketu_reply()` | Ketu's active hours detect karo |
| `should_defer_borderline()` | Ketu active + borderline confidence → defer |
| `track_ketu_replies_batch()` | Batch track from wwbun sync |

---

### 15. `core/followup.py` - Follow-up Intelligence

| Function | Kya Karta Hai |
|---|---|
| `get_pending_followups()` | Interested customers jo order nahi kiya |
| `execute_followup()` | Follow-up message generate karo |
| `get_followup_stats()` | Stats for dashboard |

---

### 16. `core/leave_manager.py` - Leave Management

| Function | Kya Karta Hai |
|---|---|
| `add_leave()` | Ketu leave pe hai / busy hours set karo |
| `check_leave_status()` | Abhi leave active hai? |
| `get_leave_auto_reply()` | Leave-aware auto reply generate |
| `get_active_leaves()` | All active/upcoming leaves |

---

### 17. `core/activity_log.py` - Activity Logging

| Function | Kya Karta Hai |
|---|---|
| `log_activity()` | Event log karo (DB + in-memory) |
| `get_activity_log()` | Activity entries with filters |
| `get_today_summary()` | Today's learning summary |
| `get_storage_stats()` | Knowledge base storage stats |

---

### 18. `core/conversation_log.py` - Conversation Logging

| Function | Kya Karta Hai |
|---|---|
| `log_conversation()` | Every AI reply save karo (for corrections) |
| `get_recent_conversations()` | Recent conversations list |
| `get_last_ai_reply()` | Last AI reply to specific customer |
| `init_conversation_log_table()` | DB table create |

---

### 19. `core/error_tracker.py` - Error Tracking

| Function | Kya Karta Hai |
|---|---|
| `track_error()` | Error log karo (in-memory ring buffer, 50 max) |
| `get_recent_errors()` | Recent errors for dashboard |
| `get_error_summary()` | Error counts by source, today, last hour |

---

### 20. `core/cloud_payload_log.py` - Claude API Debug

| Function | Kya Karta Hai |
|---|---|
| `get_anthropic_client()` | Anthropic client create karo |
| `get_recent_payloads()` | Recent Claude API calls debug data |

---

### 21. `core/git_persist.py` - GitHub Knowledge Backup

| Function | Kya Karta Hai |
|---|---|
| `restore_knowledge_from_github()` | Startup pe GitHub se knowledge restore (JSON-only mode) |
| `backup_to_github()` | Knowledge GitHub pe backup karo |

---

### 22. `learner/chat_learner.py` - WhatsApp Chat Learning

| Function | Kya Karta Hai |
|---|---|
| `parse_whatsapp_export()` | WhatsApp export text parse karo |
| `extract_knowledge_from_messages()` | Messages se knowledge extract (Claude API) |
| `extract_knowledge_from_wwbun_messages()` | wwbun sync messages se extract |
| `apply_knowledge_updates()` | Extracted knowledge apply karo (FAQs, style, etc.) |
| `learn_conversation_enders()` | Conversation ending patterns seekho |
| `detect_bought_customers_from_chat()` | Payment/order signals detect karo |
| `learn_repeat_buyer_patterns()` | Repeat buyer reply style seekho |
| `is_low_quality_owner_reply()` | "Ok", "Done", "Hmm" → skip |
| `has_business_intent()` | Customer msg mein business intent hai? |
| `is_junk_message()` | System message, media, garbage detect |

---

### 23. `learner/realtime_learner.py` - Realtime Learning

| Function | Kya Karta Hai |
|---|---|
| `learn_from_correction()` | AI wrong → Ketu correct → deep learning via Claude |
| `learn_from_voice_note()` | Voice note transcribe + learn |
| `learn_ketu_only_from_manual_chat()` | Manual chat se ketu-only patterns |
| `learn_ketu_defer_patterns()` | Cloud learning for ketu-only questions |
| `buffer_conversation()` | Conversation pair buffer karo for batch learning |
| `get_realtime_stats()` | Learner stats |
| `get_correction_stats()` | Correction pattern analysis |

---

### 24. `learner/youtube_learner.py` - YouTube Learning

| Function | Kya Karta Hai |
|---|---|
| `process_video()` | YouTube video → transcript → knowledge extract |

---

### 25. `learner/catalog_syncer.py` - Catalog Sync

| Function | Kya Karta Hai |
|---|---|
| `sync_catalog()` | GitHub repo se products.json fetch + diff + apply |

---

### 26. `learner/faq_validator.py` - FAQ Validation

| Function | Kya Karta Hai |
|---|---|
| `validate_faqs_against_catalog()` | FAQs vs catalog cross-check — deactivate outdated |
| `get_faq_health_report()` | Active/inactive/price-warned FAQs |
| `reactivate_faq()` | Owner override — bring back deactivated FAQ |

---

### 27. `scheduler.py` (~1000 lines) - Background Scheduler

| Function | Kya Karta Hai |
|---|---|
| `start_scheduler()` | Background tasks start — YouTube check, catalog sync, backfill |
| `check_youtube_channel()` | Every 12 hours — new videos check + learn |
| `backfill_youtube_channel()` | Old videos batch process |
| `get_backfill_status()` | Backfill progress |
| `get_scheduler_status()` | All scheduled tasks status |

---

### 28. `integrations/whatsapp/webhook.py` - WhatsApp Webhook

| Function | Kya Karta Hai |
|---|---|
| Webhook handler | WhatsApp Business API webhook — message receive + verify |
| Rate limiting | Per-customer + global rate limits |
| `get_rate_limit_stats()` | Rate limit stats for dashboard |

---

### 29. `integrations/whatsapp/sender.py` - WhatsApp Sender

| Function | Kya Karta Hai |
|---|---|
| `edit_message()` | WhatsApp message edit karo |
| `edit_last_message()` | Last sent message edit |
| `get_last_sent_message()` | Last message sent to a customer |

---

### 30. `integrations/youtube/handler.py` - YouTube Integration

| Function | Kya Karta Hai |
|---|---|
| YouTube API endpoints | OAuth flow + video processing endpoints |

---

### 31. `static/dashboard.html` - Dashboard UI

HTML + JavaScript single-page dashboard. Shows:
- Toggle controls (auto-reply, follow-up)
- Today's stats (replies, learning, errors)
- Customer insights
- WhatsApp sync stats
- Correction history
- Cost tracking
- FAQ health
- Ketu-only queue

---

## Identified Bugs & Issues

### Critical Issues

1. **CORS wildcard in production** (`main.py:132-137`)
   - `allow_origins=["*"]` — koi bhi origin se API call kar sakta hai
   - **Risk:** Unauthorized API access
   - **Fix:** Specific origins allow karo

2. **Single DB connection (no pool)** (`core/database.py:28`)
   - `_conn = None` — ek hi connection for entire app
   - **Risk:** Concurrent requests pe connection issues
   - **Fix:** Connection pooling use karo (psycopg2.pool)

3. **In-memory state loss on redeploy** (`core/engine.py:52-76`)
   - `_conversations`, `_shutup_until`, `_shutup_reason` — all in-memory
   - **Risk:** Redeploy pe active conversations lost, shutup cooldowns reset
   - **Fix:** Redis ya DB mein persist karo critical state

### Medium Issues

4. **Thread safety concerns** (`core/engine.py:52-68`)
   - Multiple dicts modified without locks from async handlers + background threads
   - **Risk:** Race conditions on concurrent requests

5. **Background thread fire-and-forget** (`main.py:2466-2475`)
   - `threading.Thread(target=..., daemon=True).start()` — no error tracking
   - **Risk:** Silent failures in learning

6. **Hard-coded model IDs** (`core/engine.py:86-87`)
   - `HAIKU_MODEL = "claude-haiku-4-5-20251001"` — needs update when models change
   - **Fix:** Config/env var mein move karo

7. **max_tokens=45 limit** (`core/engine.py:950`)
   - Very restrictive — some replies may get cut off
   - Has a fix (cut-off detection + trim), but still limits response quality

### Minor Issues

8. **Redundant ender check** (`core/engine.py:769`)
   - `_is_conversation_ender()` called twice — once at line 647, again at 769
   - Not a bug, but unnecessary work

9. **Import inside function** — Multiple `from core.xxx import` inside functions
   - Done intentionally (lazy loading, avoid circular imports)
   - Not a bug, but makes code harder to follow

10. **Stats double-counting risk** (`main.py:1769-1782`)
    - When buffering, quality_count = threshold count (individual msgs when broken)
    - But buffer_pairs = pairs count for display
    - These can differ, causing dashboard confusion

---

## Settings That Were Breaking (User's Original Issue)

Based on code analysis, likely issues with settings:

1. **`followup_enabled` default is `False`** (`core/config.py:56`) — Follow-up disabled by default, this is correct
2. **Settings are runtime-modified** (`main.py:375`) — `settings.auto_reply_enabled = req.enabled` — not persisted to DB, lost on redeploy
3. **No settings persistence** — Toggle changes (auto_reply, followup) are in-memory only. After Railway redeploy, everything resets to `.env` values.
   - **Fix needed:** Persist toggle state in DB

---

## Overall Assessment

**Code Quality: 7/10** — Well-structured with good logging, but main.py is too large (3131 lines). Should be split into routers.

**Architecture: 8/10** — Smart design: prompt caching, smart context selection, dual model (Haiku/Sonnet), learning buffer with dedup.

**Reliability: 6/10** — In-memory state loss, single DB connection, no connection pooling, thread safety concerns.

**Recommendation: Fix, don't rewrite.** The core architecture is sound. The issues are fixable without starting fresh. Priority fixes:
1. Persist settings/toggles in DB
2. Add connection pooling
3. Split main.py into routers
4. Add Redis for in-memory state (conversations, shutup, etc.)
