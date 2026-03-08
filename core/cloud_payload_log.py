"""Gate-level capture of EVERY payload sent to Claude API.

This wraps the Anthropic client itself so that nothing can bypass the log.
Whatever goes through client.messages.create() is captured — that IS the gate.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

from anthropic import Anthropic
from core.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_payloads: deque[dict[str, Any]] = deque(maxlen=50)  # keep last 50


def _extract_full_text(messages: list[dict], system=None) -> str:
    """Extract all text from the messages + system param — exactly what Claude sees."""
    parts = []

    # System prompt (if any)
    if system:
        if isinstance(system, str):
            parts.append(f"[SYSTEM]\n{system}")
        elif isinstance(system, list):
            # Could be cache blocks
            for block in system:
                if isinstance(block, dict) and "text" in block:
                    parts.append(f"[SYSTEM]\n{block['text']}")
                elif isinstance(block, str):
                    parts.append(f"[SYSTEM]\n{block}")

    # User/assistant messages
    for msg in messages:
        role = msg.get("role", "?").upper()
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]\n{content}")
        elif isinstance(content, list):
            # Content blocks
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    parts.append(f"[{role}]\n{block['text']}")
                elif isinstance(block, str):
                    parts.append(f"[{role}]\n{block}")

    return "\n\n".join(parts)


class _TrackedMessages:
    """Wraps client.messages so .create() is intercepted at the gate."""

    def __init__(self, original_messages):
        self._original = original_messages

    def create(self, **kwargs):
        """THE GATE — captures everything, then forwards to real API."""
        # Capture BEFORE the call
        gate_time = time.time()
        model = kwargs.get("model", "unknown")
        max_tokens = kwargs.get("max_tokens", 0)
        messages = kwargs.get("messages", [])
        system = kwargs.get("system", None)

        full_text = _extract_full_text(messages, system)
        word_count = len(full_text.split())
        char_count = len(full_text)
        est_tokens = int(word_count * 1.3)

        # Call the REAL API
        response = self._original.create(**kwargs)

        # Capture AFTER — now we have actual token counts
        actual_in = getattr(response.usage, "input_tokens", None) if hasattr(response, "usage") else None
        actual_out = getattr(response.usage, "output_tokens", None) if hasattr(response, "usage") else None
        cache_read = getattr(response.usage, "cache_read_input_tokens", None) if hasattr(response, "usage") else None
        cache_create = getattr(response.usage, "cache_creation_input_tokens", None) if hasattr(response, "usage") else None

        entry = {
            "timestamp": gate_time,
            "source": "auto-detected",  # will be enriched by caller stack
            "model": model,
            "max_tokens": max_tokens,
            "prompt_text": full_text,
            "prompt_word_count": word_count,
            "prompt_char_count": char_count,
            "estimated_input_tokens": est_tokens,
            "actual_input_tokens": actual_in,
            "actual_output_tokens": actual_out,
            "cache_read_tokens": cache_read,
            "cache_create_tokens": cache_create,
            "message_count": len(messages),
            "has_system_prompt": system is not None,
        }

        # Try to detect source from call stack
        import traceback
        stack = traceback.extract_stack()
        for frame in reversed(stack):
            fname = frame.filename.lower()
            if "chat_learner" in fname:
                if "wwbun" in (frame.name or "").lower() or "wwbun" in str(frame.line or "").lower():
                    entry["source"] = "wwbun-learning"
                else:
                    entry["source"] = "whatsapp-learning"
                break
            elif "realtime_learner" in fname:
                if "correction" in (frame.name or "").lower():
                    entry["source"] = "correction-learning"
                elif "batch" in (frame.name or "").lower():
                    entry["source"] = "realtime-analysis"
                else:
                    entry["source"] = "realtime-learning"
                break
            elif "youtube_learner" in fname:
                entry["source"] = "youtube-learning"
                break
            elif "engine" in fname:
                entry["source"] = "customer-reply"
                break

        logger.info(
            f"[CLOUD GATE] {entry['source']} | model={model} | "
            f"words={word_count} | est_tokens={est_tokens} | "
            f"actual_in={actual_in} | actual_out={actual_out} | "
            f"cache_read={cache_read}"
        )

        with _lock:
            _payloads.append(entry)

        return response


def get_anthropic_client() -> Anthropic:
    """Get an Anthropic client with gate-level payload logging.

    Use this instead of Anthropic(api_key=...) everywhere.
    Every call to client.messages.create() will be automatically captured.
    """
    client = Anthropic(api_key=settings.anthropic_api_key)
    client.messages = _TrackedMessages(client.messages)
    return client


def get_recent_payloads(limit: int = 20) -> list[dict]:
    """Return recent payloads, newest first."""
    with _lock:
        items = list(_payloads)
    items.reverse()
    return items[:limit]
