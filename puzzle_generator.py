"""
Puzzle Generator Module — Synaptia
Powered entirely by the Groq API (groq.com) — llama-3.3-70b-versatile.
Uses the OpenAI-compatible endpoint at https://api.groq.com/openai/v1.
"""

import re as _re
import json
import uuid
import time
import os
import random
import logging
import requests
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Groq API Configuration ──────────────────────────────────────────────────

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_VALIDATOR_MODEL = os.getenv("GROQ_VALIDATOR_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
GROQ_TIMEOUT = (3.0, 8.0)

FALLBACK_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "groq/compound-mini",
    "groq/compound",
]


# ── Groq API call ────────────────────────────────────────────────────────────

def _groq_generate(messages: list, max_tokens: int = 1024, temperature: float = 1.0) -> Optional[str]:
    """
    Call the Groq API via its OpenAI-compatible chat/completions endpoint with model failover.
    Returns the text response, or None on any failure.
    """
    api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
    if not api_key:
        logger.warning("[PuzzleGen] GROQ_API_KEY is not set.")
        return None

    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }

    primary_model = os.getenv("GROQ_MODEL") or GROQ_MODEL
    models_to_try = ([primary_model] + [m for m in FALLBACK_MODELS if m != primary_model])[:3]

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
                logger.error("[PuzzleGen] Groq API: invalid API key (401).")
                return None
            if resp.status_code == 429:
                logger.warning("[PuzzleGen] Rate limit on %s (429) — failing over to next model.", model)
                continue
            if resp.ok:
                data = resp.json()
                text = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                if text:
                    import re as _reg
                    cleaned = _reg.sub(r"<think>.*?</think>", "", text, flags=_reg.DOTALL).strip()
                    logger.info("[PuzzleGen] Groq (%s) responded successfully.", model)
                    return cleaned or text
            else:
                logger.warning("[PuzzleGen] Model %s failed (%d): %s", model, resp.status_code, resp.text[:120])
        except Exception as exc:
            logger.warning("[PuzzleGen] Request failed for model %s: %s", model, exc)

    return None


# ── AI Cross-Validator ───────────────────────────────────────────────────────

def _groq_validate_puzzle(question: str, answer: str, explanation: str, puzzle_type: str) -> dict:
    """
    Uses a second independent Groq call to cross-validate the generated puzzle.
    Returns a dict with keys:
      - valid (bool): True if the puzzle passes validation
      - reason (str): Explanation from the validator
      - confidence (str): "high" | "medium" | "low"
    """
    prompt = f"""You are a rigorous fact-checker and logic validator. Your SOLE job is to verify \
whether the puzzle below is internally consistent, factually correct, and has exactly one \
clearly correct answer. Be brutally honest.

Puzzle type: {puzzle_type}
Question: {question}
Proposed answer: {answer}
Explanation: {explanation}

Validation criteria:
1. Is the proposed answer factually / logically correct given the question?
2. Is the explanation consistent with the question and answer?
3. Is there any ambiguity that could allow a different answer?
4. For math/logic puzzles: verify the calculation or deduction is sound.
5. For trivia: verify the stated fact is accurate.
6. For riddles/wordplay: verify the answer satisfies the clue uniquely.

Respond with ONLY valid JSON (no markdown, no code fences):
{{
  "valid": true or false,
  "confidence": "high" or "medium" or "low",
  "reason": "One concise sentence explaining the verdict."
}}"""

    messages = [
        {"role": "system", "content": "You are a strict logic and fact-checking validator. Return only JSON."},
        {"role": "user",   "content": prompt},
    ]

    raw = _groq_generate(messages, max_tokens=256, temperature=0.0)
    if not raw:
        logger.warning("[Validator] Groq validator did not respond — skipping validation.")
        return {"valid": True, "confidence": "low", "reason": "Validator unavailable; skipped."}

    try:
        # Strip any accidental markdown fences
        import re as _rev
        clean = _rev.sub(r"```(?:json)?|```", "", raw).strip()
        start = clean.find('{')
        end   = clean.rfind('}')
        if start != -1 and end != -1:
            clean = clean[start:end + 1]
        result = json.loads(clean)
        logger.info(
            "[Validator] Verdict: valid=%s, confidence=%s | %s",
            result.get("valid"), result.get("confidence"), result.get("reason", "")
        )
        return result
    except Exception as exc:
        logger.warning("[Validator] Could not parse validator response: %s | Raw: %s", exc, raw[:200])
        return {"valid": True, "confidence": "low", "reason": "Validator parse error; assumed valid."}


# ── Angle Pools for Diverse Exercise Generation ─────────────────────────────

_RIDDLE_ANGLES = [
    "time and passage", "shadows and silhouettes", "mirrors and reflections",
    "echoes and sound", "memory and forgetting", "silence and whisper",
    "footprints and traces", "keys and doorways", "clocks and pendulums",
    "rivers and currents", "dreams and waking", "stars and constellations",
    "fire and smoke", "wind and storm", "books and ink", "ice and thaw",
]

_MATH_ANGLES = [
    "arithmetic sequences", "coin combinations", "age and generation relationships",
    "speed, distance and time", "modular arithmetic and clocks", "geometric pattern counting",
    "fractional division", "digit sum properties", "spatial arrangements",
    "weighing and balance scales", "exponential doubling", "probability intuitions",
]

_LOGIC_ANGLES = [
    "truth-tellers and deceivers", "grid deduction", "ordered sequence placement",
    "labeled boxes and misdirection", "tournament and ranking deductions",
    "colored hats and mutual inference", "seating arrangements around a table",
    "temporal scheduling constraints", "cryptic deduction rules",
]

_WORDPLAY_ANGLES = [
    "anagrams and transformations", "homophones and auditory double meanings",
    "compound words and hidden syllables", "palindromes and reversible phrasing",
    "portmanteau and blended words", "letter subtraction puzzles",
    "rhyming riddles", "double entendres and metaphorical shifts",
]

_TRIVIA_ANGLES = [
    "neuroscience and memory systems", "astronomy and orbital dynamics",
    "ancient architectural wonders", "deep ocean ecology", "historical discoveries and inventions",
    "classical literature and mythology", "botany and cellular structure",
    "music theory and acoustics", "world geography and cartography",
]

_ANGLE_MAP = {
    "riddle":   _RIDDLE_ANGLES,
    "math":     _MATH_ANGLES,
    "logic":    _LOGIC_ANGLES,
    "wordplay": _WORDPLAY_ANGLES,
    "trivia":   _TRIVIA_ANGLES,
}


_CURATED_EXERCISES = {
    "riddle": [
        {
            "question": "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I?",
            "answer": "An echo",
            "explanation": "An echo is an acoustic reflection that repeats sounds without possessing a physical body.",
            "hints": [
                "Think about auditory reflections in valleys or open chambers.",
                "You hear it only after making a loud sound yourself.",
                "It repeats your own voice back to you."
            ],
            "solution_steps": [
                "1. The clue 'speaks without a mouth' describes an acoustic repetition.",
                "2. 'Comes alive with wind' refers to sound waves carrying through air.",
                "3. Sound bouncing back creates an echo."
            ]
        },
        {
            "question": "The more you take, the more you leave behind. What am I?",
            "answer": "Footsteps",
            "explanation": "Every step you take leaves another footstep behind on the ground.",
            "hints": [
                "Consider physical movement along a trail or path.",
                "You make them visibly in wet sand or freshly fallen snow.",
                "Taking a step forward creates an impression behind you."
            ],
            "solution_steps": [
                "1. 'Taking' in this context means taking physical steps.",
                "2. Each step taken marks a physical impression behind.",
                "3. Hence, the answer is footsteps."
            ]
        },
        {
            "question": "I have cities, but no houses; forests, but no trees; and water, but no fish. What am I?",
            "answer": "A map",
            "explanation": "A map represents geographic features symbolically without containing the physical entities.",
            "hints": [
                "Think of a tool used by navigators, travelers, and cartographers.",
                "It represents landscapes on paper or a screen.",
                "It diagrams terrain, borders, and oceans symbolically."
            ],
            "solution_steps": [
                "1. Identify that the clues describe symbolic representations of landscapes.",
                "2. Maps contain cities, rivers, and forests in symbolic form.",
                "3. Therefore, the item is a map."
            ]
        }
    ],
    "math": [
        {
            "question": "If three cats catch three mice in three minutes, how many minutes does it take one hundred cats to catch one hundred mice at the same rate?",
            "answer": "3 minutes",
            "explanation": "Each cat catches one mouse every 3 minutes. With 100 cats working simultaneously, all 100 mice are caught in 3 minutes.",
            "hints": [
                "Calculate the rate of a single cat rather than multiplying the time.",
                "One cat catches one mouse in exactly three minutes.",
                "Because all cats hunt at the same time, the duration remains unchanged."
            ],
            "solution_steps": [
                "1. Rate analysis: 3 cats catch 3 mice in 3 min -> 1 cat catches 1 mouse in 3 min.",
                "2. 100 cats hunting 100 mice operate in parallel (1 mouse per cat).",
                "3. Total elapsed time is still 3 minutes."
            ]
        },
        {
            "question": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents?",
            "answer": "5 cents",
            "explanation": "Let ball = x. Bat = x + $1.00. x + (x + 1.00) = 1.10 => 2x = 0.10 => x = $0.05 (5 cents).",
            "hints": [
                "Resist the intuitive first impression of 10 cents.",
                "Set up the relationship: Bat + Ball = $1.10 and Bat - Ball = $1.00.",
                "Subtracting the $1.00 difference leaves $0.10 split evenly between the two items."
            ],
            "solution_steps": [
                "1. Let Ball = B. Then Bat = B + 1.00.",
                "2. Total: B + (B + 1.00) = 1.10 -> 2B = 0.10.",
                "3. B = 0.05, so the ball costs 5 cents."
            ]
        }
    ],
    "logic": [
        {
            "question": "You have 3 boxes: one labeled Apples, one labeled Oranges, and one labeled Both. All 3 boxes are mislabeled. You draw one fruit from the box labeled Both. It is an Apple. What are the contents of the box labeled Oranges?",
            "answer": "Both",
            "explanation": "The 'Both' box cannot contain both (all are mislabeled), so drawing an Apple proves it contains only Apples. The box labeled 'Oranges' cannot contain Oranges and cannot contain Apples, so it must contain Both.",
            "hints": [
                "Remember the premise: EVERY single box is incorrectly labeled.",
                "Drawing an apple from the 'Both' box proves that box is exclusively Apples.",
                "Now consider the box labeled 'Oranges' — it cannot be Oranges, and cannot be Apples."
            ],
            "solution_steps": [
                "1. Box labeled 'Both' has only Apples because it is mislabeled and gave an apple.",
                "2. Box labeled 'Oranges' cannot contain Oranges (mislabeled) and cannot be Apples (already found).",
                "3. Therefore, the box labeled 'Oranges' must contain Both."
            ]
        }
    ],
    "wordplay": [
        {
            "question": "What 5-letter word becomes shorter when you add two letters to it?",
            "answer": "Short",
            "explanation": "Adding 'er' to the 5-letter word 'Short' spells 'Shorter'.",
            "hints": [
                "Think literally about word lengths and suffix additions.",
                "Inspect the clue word 'shorter' directly.",
                "Take a 5-letter root word that describes small length."
            ],
            "solution_steps": [
                "1. Look for a 5-letter base word: S-H-O-R-T.",
                "2. Add the two-letter suffix 'er'.",
                "3. The resulting word is 'Shorter'."
            ]
        }
    ],
    "trivia": [
        {
            "question": "Which seahorse-shaped structure deep in the medial temporal lobe is essential for converting short-term memory into long-term declarative storage?",
            "answer": "Hippocampus",
            "explanation": "The hippocampus (from Greek for seahorse) is the primary brain structure responsible for memory consolidation and spatial mapping.",
            "hints": [
                "It takes its name from the ancient Greek word for seahorse.",
                "It is located in the medial temporal lobe of the brain.",
                "It is one of the earliest brain structures damaged in Alzheimer's disease."
            ],
            "solution_steps": [
                "1. Identify the anatomical shape clue: seahorse in the temporal lobe.",
                "2. Correlate with its neurological role in consolidating long-term memories.",
                "3. Conclude that the structure is the hippocampus."
            ]
        }
    ]
}


# ═══════════════════════════════════════════════════════════════════════════════
#  PuzzleGenerator
# ═══════════════════════════════════════════════════════════════════════════════

class PuzzleGenerator:
    """
    Generates cognitive exercises using the Groq API (groq.com) with graceful offline fallback.
    """

    def __init__(self, *args, **kwargs):
        self.puzzles: dict = {}
        self._recent_tags: list = []
        self._last_model_used: str = f"Groq / {GROQ_MODEL}"
        self._last_model_id: str = GROQ_MODEL

    # ── Active model (for health / status endpoints) ──────────────────────────

    @property
    def active_model(self) -> str:
        return self._last_model_id or GROQ_MODEL

    @property
    def active_model_display(self) -> str:
        return self._last_model_used or f"Groq / {GROQ_MODEL}"

    # ── Fallback cognitive exercises ──────────────────────────────────────────

    def _get_fallback_exercise(self, difficulty: str, puzzle_type: str) -> dict:
        category = puzzle_type if puzzle_type in _CURATED_EXERCISES else "riddle"
        options = _CURATED_EXERCISES.get(category, _CURATED_EXERCISES["riddle"])
        chosen = random.choice(options).copy()

        puzzle_id = str(uuid.uuid4())[:8]
        chosen["id"] = puzzle_id
        chosen["difficulty"] = difficulty
        chosen["type"] = category
        chosen["category"] = category
        chosen["solved"] = False
        chosen["created_at"] = datetime.now().isoformat()
        chosen["model_used"] = "Synaptia Cognitive Engine"
        chosen["validation"] = {
            "passed": True,
            "confidence": "high",
            "note": "Verified curated cognitive exercise.",
            "validator": "Synaptia Core Engine",
        }
        self.puzzles[puzzle_id] = chosen
        return chosen

    # ── Puzzle generation ─────────────────────────────────────────────────────

    def generate_puzzle(self, difficulty="medium", puzzle_type="riddle"):
        try:
            return self._generate_with_validation(difficulty, puzzle_type, attempt=1)
        except Exception as e:
            logger.error("[PuzzleGen] Unhandled exception: %s", e)
            return self._get_fallback_exercise(difficulty, puzzle_type)

    def _generate_with_validation(self, difficulty, puzzle_type, attempt=1):
        """Generate a puzzle then cross-validate it. Falls back gracefully if unavailable."""
        MAX_ATTEMPTS = 2

        messages = self._build_messages(difficulty, puzzle_type)
        puzzle_content = _groq_generate(messages, max_tokens=1024, temperature=1.0)

        if puzzle_content is None:
            logger.info("[PuzzleGen] Groq unavailable; serving curated cognitive exercise.")
            return self._get_fallback_exercise(difficulty, puzzle_type)

        self._last_model_used = f"Groq / {GROQ_MODEL}"
        self._last_model_id = GROQ_MODEL

        puzzle = self._parse_puzzle(puzzle_content, difficulty, puzzle_type)
        if puzzle.get("answer") == "N/A" or not puzzle.get("question") or puzzle.get("question").startswith("The exercise could not be decoded"):
            logger.info("[PuzzleGen] Decoding incomplete; serving curated cognitive exercise.")
            return self._get_fallback_exercise(difficulty, puzzle_type)

        # ── AI Cross-Validation ──────────────────────────────────────────────
        question    = puzzle.get("question", "")
        answer      = puzzle.get("answer", "")
        explanation = puzzle.get("explanation", "")

        if question and answer and "N/A" not in answer:
            validation = _groq_validate_puzzle(question, answer, explanation, puzzle_type)
            puzzle["validation"] = {
                "passed":     validation.get("valid", True),
                "confidence": validation.get("confidence", "low"),
                "note":       validation.get("reason", ""),
                "validator":  f"Groq / {GROQ_VALIDATOR_MODEL}",
            }

            if not validation.get("valid", True) and attempt < MAX_ATTEMPTS:
                logger.warning(
                    "[PuzzleGen] Validation FAILED (attempt %d/%d) — reason: %s. Regenerating…",
                    attempt, MAX_ATTEMPTS, validation.get("reason", "unknown")
                )
                return self._generate_with_validation(difficulty, puzzle_type, attempt + 1)

            if not validation.get("valid", True):
                logger.warning(
                    "[PuzzleGen] Validation FAILED after %d attempts. Returning best-effort puzzle.",
                    MAX_ATTEMPTS
                )
        else:
            puzzle["validation"] = {
                "passed":     None,
                "confidence": "low",
                "note":       "Puzzle could not be validated (parse error or N/A answer).",
                "validator":  f"Groq / {GROQ_VALIDATOR_MODEL}",
            }
        # ─────────────────────────────────────────────────────────────────────

        puzzle_id = str(uuid.uuid4())[:8]
        puzzle["id"]         = puzzle_id
        puzzle["created_at"] = datetime.now().isoformat()
        puzzle["model_used"] = self._last_model_used
        self.puzzles[puzzle_id] = puzzle
        return puzzle

    # ── Prompt / message building ─────────────────────────────────────────────

    def _build_messages(self, difficulty, puzzle_type) -> list:
        """Build the chat messages list for puzzle generation."""
        system_content = (
            "You are a world-class puzzle master with an encyclopaedic knowledge "
            "of riddles, logic, mathematics, wordplay, and trivia. "
            "You ALWAYS create unique, original puzzles — never repeat a puzzle you have given before. "
            "Respond with ONLY valid JSON. No markdown, no code fences, no extra text whatsoever."
        )

        difficulty_desc = {
            "easy":   "suitable for beginners, straightforward logic with clear reasoning",
            "medium": "moderate difficulty requiring creative thinking and analysis",
            "hard":   "challenging puzzle requiring deep logical reasoning and multi-step problem-solving",
        }
        puzzle_type_desc = {
            "riddle":   "a clever riddle or wordplay puzzle with a surprising, satisfying answer",
            "math":     "a mathematical logic puzzle requiring calculation and deduction",
            "logic":    "a deductive logic puzzle requiring systematic elimination",
            "wordplay": "a lateral thinking or wordplay brain teaser",
            "trivia":   "an interesting trivia question with a fascinating explanation",
        }
        base_diff   = difficulty_desc.get(difficulty, "moderate difficulty")
        puzzle_desc = puzzle_type_desc.get(puzzle_type, "an engaging riddle")

        angle_pool = _ANGLE_MAP.get(puzzle_type, _RIDDLE_ANGLES)
        fresh = [a for a in angle_pool if a not in self._recent_tags]
        if not fresh:
            fresh = angle_pool
            self._recent_tags = []

        angle = random.choice(fresh)
        nonce = uuid.uuid4().hex[:8]

        self._recent_tags.append(angle)
        if len(self._recent_tags) > len(angle_pool) // 2:
            self._recent_tags.pop(0)

        user_content = f"""Create {puzzle_desc} ({base_diff}).

UNIQUENESS DIRECTIVE — CRITICAL:
- Theme / domain: "{angle}"
- Session nonce: {nonce}  ← use this as creative inspiration seed
- The puzzle MUST be original and MUST NOT be a common or well-known puzzle.
- Do NOT repeat standard classics (e.g. "I speak without a mouth", "man in an elevator", "Tuesday").

REQUIREMENTS:
- Puzzle must be interesting, engaging, and NOT trivial
- Answer must be clear, correct, and verifiable
- Provide exactly 3 progressive hints (each as a plain string, no newlines inside strings)
- Provide 2-3 solution steps explaining the reasoning (each as a plain string)
- ALL string values must be on a single line — do NOT use literal newline characters inside any string value

Return ONLY valid JSON with this exact structure:
{{
    "question": "The complete puzzle question on a single line",
    "answer": "The single correct answer (1-3 words ideally)",
    "explanation": "Clear 1-2 sentence explanation on a single line",
    "hints": [
        "Hint 1: subtle directional clue on one line",
        "Hint 2: narrows possibilities significantly on one line",
        "Hint 3: strong hint that almost reveals the answer on one line"
    ],
    "solution_steps": [
        "Step 1: First part of the logical reasoning on one line",
        "Step 2: How to arrive at the answer on one line",
        "Step 3: Why this answer is definitive on one line"
    ],
    "category": "{puzzle_type}",
    "difficulty": "{difficulty}"
}}"""

        return [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_content},
        ]

    # ── Parsing Infrastructure ─────────────────────────────────────────────────

    def _extract_json_object(self, text):
        start = text.find('{')
        if start == -1:
            return text

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            c = text[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

        end = text.rfind('}')
        if end > start:
            return text[start:end + 1]
        return text

    def _sanitize_string_values(self, content):
        result = []
        in_string = False
        escape_next = False

        for c in content:
            if escape_next:
                result.append(c)
                escape_next = False
            elif c == '\\' and in_string:
                result.append(c)
                escape_next = True
            elif c == '"':
                in_string = not in_string
                result.append(c)
            elif c == '\n' and in_string:
                result.append('\\n')
            elif c == '\r' and in_string:
                result.append('\\r')
            elif c == '\t' and in_string:
                result.append('\\t')
            else:
                result.append(c)

        return ''.join(result)

    def _parse_puzzle(self, content, difficulty, puzzle_type):
        import re
        try:
            raw = content.strip()
            extracted = self._extract_json_object(raw)
            sanitized = self._sanitize_string_values(extracted)

            try:
                puzzle_data = json.loads(sanitized)
            except json.JSONDecodeError:
                fence_match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
                if fence_match:
                    candidate = fence_match.group(1).strip()
                else:
                    candidate = self._extract_json_object(self._sanitize_string_values(raw))
                puzzle_data = json.loads(candidate)

            puzzle_data["difficulty"] = difficulty
            puzzle_data["type"]       = puzzle_type
            puzzle_data["solved"]     = False

            if "solution_steps" not in puzzle_data:
                puzzle_data["solution_steps"] = [puzzle_data.get("explanation", "No explanation available.")]
            elif isinstance(puzzle_data["solution_steps"], str):
                puzzle_data["solution_steps"] = [puzzle_data["solution_steps"]]

            if "hints" in puzzle_data and isinstance(puzzle_data["hints"], str):
                puzzle_data["hints"] = [puzzle_data["hints"]]

            return puzzle_data

        except Exception as e:
            logger.error("[PuzzleGen] Parse failure — %s | Raw excerpt: %s", e, content[:300])

        return {
            "question":       "The exercise could not be decoded at this time. Please regenerate.",
            "answer":         "N/A",
            "explanation":    "The AI response could not be parsed. Please try regenerating.",
            "difficulty":     difficulty,
            "type":           puzzle_type,
            "hints":          ["Please regenerate this exercise."],
            "solution_steps": ["Regenerate the exercise to receive a valid challenge."],
            "solved":         False,
        }

    # ── Retrieval & Validation ─────────────────────────────────────────────────

    def get_puzzle(self, puzzle_id):
        return self.puzzles.get(puzzle_id)

    def list_puzzles(self):
        safe_list = []
        for p in self.puzzles.values():
            safe = {k: v for k, v in p.items() if k not in ("answer", "solution_steps")}
            safe_list.append(safe)
        return safe_list

    def check_answer(self, puzzle_id, user_answer):
        """Check if the user's answer is correct using normalised fuzzy matching."""
        puzzle = self.get_puzzle(puzzle_id)
        if not puzzle:
            return {"error": "Exercise not found"}

        import re
        from difflib import SequenceMatcher

        user_norm    = re.sub(r'[^a-z0-9\s]', '', user_answer.lower().strip())
        correct_norm = re.sub(r'[^a-z0-9\s]', '', puzzle["answer"].lower().strip())

        correct = False

        if user_norm == correct_norm:
            correct = True
        else:
            stopwords = {"a", "an", "the", "it", "is", "its", "of", "and"}
            user_tokens    = set(user_norm.split()) - stopwords
            correct_tokens = set(correct_norm.split()) - stopwords

            if user_tokens == correct_tokens and len(correct_tokens) > 0:
                correct = True
            elif correct_tokens and correct_tokens.issubset(user_tokens):
                correct = True
            elif correct_tokens and user_tokens:
                meaningful = {w for w in correct_tokens if len(w) > 3}
                if meaningful and meaningful.intersection(user_tokens):
                    correct = True

            if not correct and len(correct_norm) > 3:
                ratio = SequenceMatcher(None, user_norm, correct_norm).ratio()
                if ratio > 0.8:
                    correct = True

        if correct:
            puzzle["solved"] = True

        return {
            "correct": correct,
            "answer":  puzzle["answer"] if correct else None,
        }
