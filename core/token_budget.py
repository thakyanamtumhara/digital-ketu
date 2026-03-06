"""Token budget management — hard caps to keep costs under ₹0.50/reply.

Estimates token counts BEFORE sending to Claude API.
Uses character-based estimation (1 token ≈ 4 chars for English, ~3 chars for Hinglish).
This is approximate but good enough for budgeting — we don't need exact counts.

Target budget per reply:
- System prompt (static): ~800 tokens (cached via prompt caching = 90% cheaper)
- Knowledge context: max 1,200 tokens (HARD CAP — enforced by truncation)
- Conversation history: max 400 tokens (2 exchanges = 4 messages)
- Output: max 150 tokens
- Total: ~2,500 tokens → ~₹0.20/reply on Haiku, ~₹0.40/reply on Sonnet with caching
"""

import logging

logger = logging.getLogger(__name__)

# --- Token budget limits ---
# These are HARD CAPS — context gets truncated if it exceeds these
BUDGET_KNOWLEDGE_TOKENS = 1200      # Max tokens for knowledge context (products, FAQs, style)
BUDGET_HISTORY_TOKENS = 400         # Max tokens for conversation history
BUDGET_SYSTEM_STATIC_TOKENS = 1000  # Approximate tokens for static prompt (identity, rules, etc.)
BUDGET_TOTAL_INPUT_TOKENS = 2800    # Absolute max input tokens per reply

# Hinglish has more multi-byte chars, so use ~3.2 chars per token
CHARS_PER_TOKEN = 3.2


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length. Conservative estimate."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN)


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within token budget.

    Tries to cut at a line boundary to keep context coherent.
    """
    current = estimate_tokens(text)
    if current <= max_tokens:
        return text

    # Calculate max chars allowed
    max_chars = int(max_tokens * CHARS_PER_TOKEN)
    truncated = text[:max_chars]

    # Try to cut at last newline for cleaner truncation
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.7:  # Only if we don't lose too much
        truncated = truncated[:last_newline]

    tokens_cut = current - estimate_tokens(truncated)
    logger.info(f"[TokenBudget] Truncated context: {current} → {estimate_tokens(truncated)} tokens (cut {tokens_cut})")
    return truncated


def truncate_history(messages: list, max_tokens: int) -> list:
    """Truncate conversation history to fit within token budget.

    Keeps the most recent messages (they're most relevant).
    """
    if not messages:
        return messages

    total = sum(estimate_tokens(m.get("content", "")) for m in messages)
    if total <= max_tokens:
        return messages

    # Remove oldest messages until we fit
    while messages and total > max_tokens:
        removed = messages.pop(0)
        total -= estimate_tokens(removed.get("content", ""))

    logger.info(f"[TokenBudget] History trimmed to {len(messages)} messages (~{total} tokens)")
    return messages


def log_budget_usage(
    system_tokens: int,
    knowledge_tokens: int,
    history_tokens: int,
    total_tokens: int,
):
    """Log token budget usage for monitoring."""
    logger.info(
        f"[TokenBudget] system={system_tokens}, knowledge={knowledge_tokens}, "
        f"history={history_tokens}, total={total_tokens} "
        f"(budget: {BUDGET_TOTAL_INPUT_TOKENS})"
    )
    if total_tokens > BUDGET_TOTAL_INPUT_TOKENS:
        logger.warning(
            f"[TokenBudget] OVER BUDGET by {total_tokens - BUDGET_TOTAL_INPUT_TOKENS} tokens!"
        )
