"""
Groq LLM client wrapper.
Free tier: https://console.groq.com — no credit card required.
"""
from __future__ import annotations
import os
import time
import logging
from typing import List, Dict, Optional, Iterator
from groq import Groq, RateLimitError
from app.config import GROQ_API_KEY, GROQ_MODEL

logger = logging.getLogger(__name__)

# Max seconds to wait on a rate-limit before giving up
_MAX_RETRY_WAIT = 60


def _retry_on_rate_limit(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) and retry up to once if a 429 rate limit is hit.
    Reads the retry-after time from the error message when available.
    """
    try:
        return fn(*args, **kwargs)
    except RateLimitError as e:
        # Parse "try again in X.Xs" from the Groq error message
        wait = 30  # default wait
        msg = str(e)
        import re
        m = re.search(r"try again in ([0-9.]+)s", msg)
        if m:
            wait = min(float(m.group(1)) + 1, _MAX_RETRY_WAIT)
        logger.warning("Groq rate limit hit. Waiting %.1fs before retry...", wait)
        time.sleep(wait)
        return fn(*args, **kwargs)


class LLMClient:
    """Thin wrapper around Groq chat completions with streaming support."""

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not set. Copy .env.example → .env and add your key.\n"
                "Get a free key at: https://console.groq.com"
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = GROQ_MODEL

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Send a list of chat messages, return assistant reply as string."""
        def _call():
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        return _retry_on_rate_limit(_call)

    def complete(self, prompt: str, **kwargs) -> str:
        """Single-turn convenience wrapper."""
        return self.chat([{"role": "user", "content": prompt}], **kwargs)

    def complete_with_system(self, system: str, user: str, **kwargs) -> str:
        """Two-message convenience wrapper."""
        return self.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **kwargs,
        )

    def stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """
        Stream tokens from the LLM. Yields text chunks as they arrive.
        Falls back to non-streaming if rate-limited, emitting full response as one chunk.

        Usage:
            for chunk in llm.stream(messages):
                print(chunk, end="", flush=True)
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except RateLimitError as e:
            # On rate limit during stream: wait and fall back to a single blocking call
            wait = 30
            import re
            m = re.search(r"try again in ([0-9.]+)s", str(e))
            if m:
                wait = min(float(m.group(1)) + 1, _MAX_RETRY_WAIT)
            logger.warning("Groq rate limit during stream. Waiting %.1fs...", wait)
            time.sleep(wait)
            # Retry as non-streaming, yield the full response as one token
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            yield response.choices[0].message.content.strip()

    def stream_complete(self, prompt: str, **kwargs) -> Iterator[str]:
        """Single-turn streaming convenience wrapper."""
        yield from self.stream([{"role": "user", "content": prompt}], **kwargs)
