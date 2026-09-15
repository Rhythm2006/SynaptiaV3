"""
Hint Provider Module — Synaptia
AI chatbot assistant powered entirely by the Groq API (groq.com).
Uses the OpenAI-compatible endpoint at https://api.groq.com/openai/v1.
Model: llama-3.3-70b-versatile
"""

import requests
import logging
import os
import time
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Groq API Configuration ───────────────────────────────────────────────────

import re

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL    = os.getenv("GROQ_MODEL", "groq/compound-mini")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_TIMEOUT  = 60

FALLBACK_MODELS = [
    GROQ_MODEL,
    "groq/compound-mini",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]


# ── Groq chat helper ─────────────────────────────────────────────────────────

def _groq_chat(messages: list, temperature: float = 0.75, max_tokens: int = 600) -> Optional[str]:
    """
    Send a chat request to the Groq API via the OpenAI-compatible endpoint with model fallbacks.
    Returns the assistant's response string, or None on failure.
    """
    api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    if not api_key:
        logger.warning("[HintProvider] GROQ_API_KEY is not set.")
        return None

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    primary_model = os.getenv("GROQ_MODEL") or GROQ_MODEL
    models_to_try = [primary_model] + [m for m in FALLBACK_MODELS if m != primary_model]

    for model in models_to_try:
        payload = {
            "model":       model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=GROQ_TIMEOUT)
            if resp.status_code == 401:
                logger.error("[HintProvider] Groq API: invalid API key (401).")
                return None
            if resp.status_code == 429:
                logger.warning("[HintProvider] Groq rate limit hit on model %s. Retrying in 2s...", model)
                time.sleep(2)
                resp = requests.post(url, headers=headers, json=payload, timeout=GROQ_TIMEOUT)
            if resp.ok:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if text:
                    # Strip any reasoning or think tokens
                    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                    logger.info("[HintProvider] Groq responded successfully with model %s.", model)
                    return cleaned or text
            else:
                logger.warning("[HintProvider] Model %s failed with status %d: %s", model, resp.status_code, resp.text[:120])
        except Exception as exc:
            logger.warning("[HintProvider] Request failed for model %s: %s", model, exc)

    return None


# ── HintProvider ─────────────────────────────────────────────────────────────

class HintProvider:
    """Provides hints and multi-mode AI chatbot functionality powered by Groq."""

    _FALLBACK_RESPONSES = [
        (
            "I'm temporarily unable to reach the Groq API. "
            "Please check your GROQ_API_KEY or network connection, then try again."
        ),
        (
            "The Groq AI model is currently unavailable. "
            "Please verify your API key has credits or retry shortly."
        ),
        (
            "I couldn't get a response from Groq. "
            "Please verify your GROQ_API_KEY and try again."
        ),
    ]
    _fallback_index = 0

    def __init__(self, *args, **kwargs):
        self.conversations: dict = {}
        self._active_model_id: str = GROQ_MODEL
        self._active_model_display: str = f"Groq / {GROQ_MODEL}"

    # ── Public: active model ─────────────────────────────────────────────────

    @property
    def active_model(self) -> str:
        return self._active_model_id or GROQ_MODEL

    @property
    def active_model_display(self) -> str:
        return self._active_model_display or f"Groq / {GROQ_MODEL}"

    # ── Public: hints ────────────────────────────────────────────────────────

    def get_hint(self, puzzle: dict, hint_number: int = 0) -> str:
        if puzzle.get("hints") and hint_number < len(puzzle["hints"]):
            return puzzle["hints"][hint_number]
        return self._generate_hint(puzzle, hint_number)

    def _generate_hint(self, puzzle: dict, hint_number: int) -> str:
        """Generate a dynamic hint via Groq."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a compassionate cognitive support assistant. "
                    "Generate a single concise hint. Return ONLY the hint text, nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Puzzle: {puzzle.get('question', '')}\n"
                    f"Answer: {puzzle.get('answer', '')}\n\n"
                    f"Generate hint #{hint_number + 1}. "
                    "Progressive difficulty: "
                    "Clue I — subtle, directional nudge; "
                    "Clue II — stronger, narrows possibilities; "
                    "Clue III — almost reveals the answer.\n"
                    "Return ONLY the hint text."
                ),
            },
        ]
        result = _groq_chat(messages, temperature=0.7, max_tokens=200)
        return result or "Unable to generate a clue at this time. Please try again shortly."

    # ── Public: chat ─────────────────────────────────────────────────────────

    def chat(
        self,
        session_id: str,
        user_message: str,
        puzzle: dict = None,
        hints_used: int = 0,
        chat_mode: str = "hint_bot",
    ) -> str:
        if session_id not in self.conversations:
            self.conversations[session_id] = []

        system_content = self._build_system_prompt(puzzle, hints_used, chat_mode)

        self.conversations[session_id].append(
            {"role": "user", "content": user_message}
        )

        messages = [{"role": "system", "content": system_content}]
        messages += [
            {"role": m["role"] if m["role"] in ("user", "assistant") else "user",
             "content": m["content"]}
            for m in self.conversations[session_id][-12:]
        ]

        response_text = _groq_chat(messages, temperature=0.75, max_tokens=600)

        if response_text:
            self._active_model_id = GROQ_MODEL
            self._active_model_display = f"Groq / {GROQ_MODEL}"
            self.conversations[session_id].append(
                {"role": "assistant", "content": response_text}
            )
        else:
            response_text = self._next_fallback()
            logger.warning("[HintProvider] Groq unavailable; sending fallback to user.")

        return response_text

    def clear_conversation(self, session_id: str) -> bool:
        if session_id in self.conversations:
            del self.conversations[session_id]
            return True
        return False

    # ── Fallback message rotation ─────────────────────────────────────────────

    def _next_fallback(self) -> str:
        msg = HintProvider._FALLBACK_RESPONSES[HintProvider._fallback_index]
        HintProvider._fallback_index = (
            HintProvider._fallback_index + 1
        ) % len(HintProvider._FALLBACK_RESPONSES)
        return msg

    # ── System prompts ───────────────────────────────────────────────────────

    def _build_system_prompt(self, puzzle: dict, hints_used: int, chat_mode: str) -> str:
        mode_prompts = {
            "hint_bot": (
                "You are Synaptia's Cognitive Companion — a warm, patient, and encouraging guide "
                "designed to support individuals facing neurological challenges. Your role is to "
                "gently help users reason through cognitive exercises without revealing answers prematurely.\n"
                "PRINCIPLES:\n"
                "1. Provide directional guidance, not direct answers.\n"
                "2. Ask gentle Socratic questions that steer thinking without overwhelming.\n"
                "3. Keep responses concise — two to four sentences.\n"
                "4. Celebrate every reasoning step, not just correct answers.\n"
                "5. Maintain a calm, compassionate, and affirming tone at all times."
            ),
            "free_chat": (
                "You are Synaptia's Open Dialogue companion — a thoughtful, empathetic conversationalist "
                "designed to walk alongside individuals facing neurological or memory challenges.\n"
                "PRINCIPLES:\n"
                "1. Respond with warmth, clarity, and intellectual substance.\n"
                "2. Keep answers focused — three to five sentences unless more is warranted.\n"
                "3. Be encouraging and patient — never rushed or dismissive.\n"
                "4. Maintain a professional yet deeply approachable tone.\n"
                "5. Freely discuss memory, cognition, puzzles, or anything the user needs."
            ),
            "tutor": (
                "You are Professor Synaptia — a dedicated Cognitive Tutor. Your mission is to "
                "build the user's reasoning capabilities through structured, patient instruction "
                "tailored for individuals who may be navigating neurological challenges.\n"
                "PRINCIPLES:\n"
                "1. Break complex logic into numbered, digestible steps.\n"
                "2. Ask questions that prompt independent discovery before giving answers.\n"
                "3. Identify and gently correct logical misconceptions without judgment.\n"
                "4. Reinforce correct reasoning with specific, meaningful praise.\n"
                "5. Use analogies and concrete examples to make abstract reasoning accessible."
            ),
            "creative": (
                "You are the Creative Mind — Synaptia's lateral thinking specialist. "
                "You help users explore unconventional solution paths through imaginative, "
                "compassionate reasoning — especially valuable for reconnecting creative thinking.\n"
                "PRINCIPLES:\n"
                "1. Challenge obvious interpretations — look for unexpected angles.\n"
                "2. Draw on metaphor, wordplay, and cross-domain analogies.\n"
                "3. Encourage speculative thinking without sacrificing clarity.\n"
                "4. Keep the intellectual energy curious, warm, and exploratory.\n"
                "5. Validate all ideas — creative leaps are signs of resilience."
            ),
        }

        system = mode_prompts.get(chat_mode, mode_prompts["hint_bot"])

        if puzzle and chat_mode == "hint_bot":
            system += (
                f"\n\nACTIVE EXERCISE CONTEXT:\n"
                f"Question: {puzzle.get('question', '')}\n"
                f"Answer: {puzzle.get('answer', '')}\n"
                f"Difficulty: {puzzle.get('difficulty', 'medium')}\n"
                f"Type: {puzzle.get('type', 'riddle')}"
            )
            if hints_used < 2:
                system += (
                    "\n\nIMPORTANT CONSTRAINT: The user has used fewer than two clues. "
                    "Do NOT reveal the answer even if explicitly requested. "
                    "Gently direct them to use the clue buttons first."
                )
            else:
                system += (
                    "\n\nThe user has used two or more clues. If they explicitly and "
                    "persistently need the full answer, you may reveal it — but first "
                    "offer one gentle final nudge toward independent discovery."
                )

        return system
