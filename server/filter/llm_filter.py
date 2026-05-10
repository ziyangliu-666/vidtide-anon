"""LLM-based filter: uses an OpenAI-compatible API to judge if a video is AI-generated."""

from __future__ import annotations

import json
import logging
import os
import time

import requests

from server.filter.base import BaseFilter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a classifier that determines whether a video is AI-generated content. "
    "You will be given a video title and description. Decide whether the video itself "
    "is AI-generated (not a tutorial, review, or commentary ABOUT AI video tools, but "
    "actual AI-generated video content). Reply with ONLY valid JSON, no markdown fences."
)

_USER_TEMPLATE = (
    "Based on this video title and description, is this video likely AI-generated content "
    "(not a tutorial/review ABOUT AI video, but actual AI-generated video)?\n\n"
    "Title: {title}\n"
    "Description: {description}\n\n"
    'Reply with JSON: {{"is_ai_generated": true/false, "confidence": 0.0-1.0, "reason": "..."}}'
)


class LLMFilter(BaseFilter):
    """Filter candidates by asking an LLM whether the video is AI-generated."""

    name = "llm"

    def __init__(self) -> None:
        self._stats: dict = {
            "total_in": 0,
            "total_out": 0,
            "llm_approved": 0,
            "llm_rejected": 0,
            "llm_errors": 0,
        }

    def filter(self, candidates: list[dict], config: dict) -> list[dict]:
        self._stats["total_in"] = len(candidates)

        if not config.get("enabled", False):
            self._stats["total_out"] = len(candidates)
            return candidates

        api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.warning("LLMFilter: no API key configured, passing all candidates through")
            self._stats["total_out"] = len(candidates)
            return candidates

        model = config.get("model", "gpt-4o-mini")
        threshold = config.get("confidence_threshold", 0.7)

        kept: list[dict] = []
        for i, candidate in enumerate(candidates):
            if i > 0:
                time.sleep(0.5)  # rate limit

            title = candidate.get("title", "") or ""
            description = candidate.get("description", "") or ""
            source_url = candidate.get("source_url", "")

            # If no title/description available, keep the candidate
            if not title and not description:
                kept.append(candidate)
                self._stats["llm_approved"] += 1
                continue

            try:
                result = self._call_llm(api_key, model, title, description)
            except Exception:
                logger.warning(
                    "LLMFilter: error calling LLM for %s, keeping candidate",
                    source_url,
                    exc_info=True,
                )
                self._stats["llm_errors"] += 1
                kept.append(candidate)
                continue

            is_ai = result.get("is_ai_generated", False)
            confidence = float(result.get("confidence", 0.0))

            if is_ai and confidence >= threshold:
                kept.append(candidate)
                self._stats["llm_approved"] += 1
                logger.debug(
                    "LLMFilter: KEEP %s (confidence=%.2f, reason=%s)",
                    source_url,
                    confidence,
                    result.get("reason", ""),
                )
            else:
                self._stats["llm_rejected"] += 1
                logger.debug(
                    "LLMFilter: REJECT %s (is_ai=%s, confidence=%.2f, reason=%s)",
                    source_url,
                    is_ai,
                    confidence,
                    result.get("reason", ""),
                )

        self._stats["total_out"] = len(kept)
        return kept

    def stats(self) -> dict:
        return dict(self._stats)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _call_llm(api_key: str, model: str, title: str, description: str) -> dict:
        """Call the OpenAI chat completions API and parse the JSON response."""
        user_msg = _USER_TEMPLATE.format(title=title, description=description)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        }

        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()

        body = resp.json()
        content = body["choices"][0]["message"]["content"].strip()

        # Strip markdown code fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        return json.loads(content)
