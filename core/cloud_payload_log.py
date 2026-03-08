"""Log every payload sent to Claude API for dashboard debugging.

Keeps the last N payloads in memory so the dashboard can show
exactly what text is being sent to the cloud.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

_lock = threading.Lock()
_payloads: deque[dict[str, Any]] = deque(maxlen=50)  # keep last 50


def log_cloud_payload(
    *,
    source: str,
    prompt_text: str,
    model: str,
    max_tokens: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    pair_count: int | None = None,
    message_count: int | None = None,
    extra: dict | None = None,
) -> None:
    """Record a payload that was sent to Claude API."""
    word_count = len(prompt_text.split())
    char_count = len(prompt_text)
    est_tokens = int(word_count * 1.3)  # rough estimate for Hinglish

    entry = {
        "timestamp": time.time(),
        "source": source,
        "model": model,
        "max_tokens": max_tokens,
        "prompt_text": prompt_text,
        "prompt_word_count": word_count,
        "prompt_char_count": char_count,
        "estimated_input_tokens": est_tokens,
        "actual_input_tokens": input_tokens,
        "actual_output_tokens": output_tokens,
        "pair_count": pair_count,
        "message_count": message_count,
    }
    if extra:
        entry.update(extra)

    with _lock:
        _payloads.append(entry)


def get_recent_payloads(limit: int = 20) -> list[dict]:
    """Return recent payloads, newest first."""
    with _lock:
        items = list(_payloads)
    items.reverse()
    return items[:limit]
