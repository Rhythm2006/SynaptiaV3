"""LLM client for conversation processing (Groq, Fireworks.ai, or OpenAI)."""

import asyncio
import json
import logging
import os
import re
from typing import List

from dotenv import load_dotenv
from openai import AsyncOpenAI

from models import ConversationUtterance

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llm_client")

# Configuration with multi-provider fallback (Groq prioritized, then Fireworks, then OpenAI)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if GROQ_API_KEY:
    api_key = GROQ_API_KEY
    base_url = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = os.getenv("GROQ_MODEL", "groq/compound")
    provider_name = "Groq"
elif FIREWORKS_API_KEY:
    api_key = FIREWORKS_API_KEY
    base_url = "https://api.fireworks.ai/inference/v1"
    DEFAULT_MODEL = os.getenv("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p1-70b-instruct")
    provider_name = "Fireworks.ai"
elif OPENAI_API_KEY:
    api_key = OPENAI_API_KEY
    base_url = "https://api.openai.com/v1"
    DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    provider_name = "OpenAI"
else:
    api_key = "dummy"
    base_url = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "groq/compound"
    provider_name = "Mock/Disabled"

logger.info(f"Initialized LLM client using provider: {provider_name} (model: {DEFAULT_MODEL})")

# Initialize client (uses OpenAI-compatible API)
client = AsyncOpenAI(
    api_key=api_key,
    base_url=base_url
)


async def call_llm_with_retry(messages: list, temperature: float = 0.5, max_tokens: int = 150) -> str:
    """Call the LLM with exponential backoff on rate limits or API errors."""
    retries = 5
    delay = 1.0
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return clean_llm_response(response.choices[0].message.content)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit_exceeded" in str(e)
            if (is_rate_limit or attempt < retries - 1) and attempt < retries - 1:
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{retries}): {e}. Retrying in {delay:.1f}s..."
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error(f"LLM call failed permanently on attempt {attempt + 1}: {e}")
                raise e
    raise RuntimeError("LLM call failed after all retries")


def clean_llm_response(text: str) -> str:
    """Clean LLM output of reasoning tokens or markdown artifacts."""
    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return cleaned


async def aggregate_conversation_context(
    person_name: str,
    current_context: str,
    new_conversation: List[ConversationUtterance]
) -> str:
    """
    Model #1: Context Aggregation
    Takes the current aggregated context and a new conversation, returns updated summary.
    """
    # Format the conversation for the prompt
    conversation_text = "\n".join([
        f"{utt.speaker}: {utt.text}"
        for utt in new_conversation
    ])

    # Prompt for context aggregation model
    system_prompt = """You are a context aggregation assistant for a dementia care AR system.

Your job is to maintain a running summary of all conversations with a person. Focus strictly on only the necessary key details (such as names, relationships, important updates, critical life events, and upcoming plans). Ignore any trivial small talk, filler, or pleasantries.
Do NOT repeat the transcription text or dialogue verbatim. Instead, summarize what happened or what was learned in a factual, third-person format.

Given:
1. The current aggregated context (summary of past conversations)
2. A new conversation that just happened

You should output an UPDATED aggregated context that:
- Summarizes the necessary new information from the latest conversation in a third-person format
- Maintains important details from previous conversations
- Is concise and strictly focused on essential facts (1-2 sentences maximum)
- Completely ignores small talk and greetings

Output ONLY the updated context, nothing else."""

    user_prompt = f"""Person: {person_name}

Current Aggregated Context:
{current_context}

New Conversation:
{conversation_text}

Provide the updated aggregated context:"""

    try:
        logger.info(f"Calling LLM Model #1 (Context Aggregation) for {person_name}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        updated_context = await call_llm_with_retry(messages, temperature=0.3, max_tokens=200)

        logger.info(f"Context aggregation complete for {person_name}")
        logger.debug(f"Updated context: {updated_context}")

        return updated_context

    except Exception as e:
        logger.error(f"Error calling LLM context aggregation: {e}")
        # Fallback: append simple summary to existing context
        fallback = f"{current_context} Recently discussed: {conversation_text[:100]}..."
        return fallback


async def generate_ar_description(
    person_name: str,
    relationship: str,
    aggregated_context: str
) -> str:
    """
    Model #2: AR Description Generation
    Takes person info and aggregated context, returns one-line AR display description.
    """
    system_prompt = """You are an AR description generator for a dementia care system helping patients with memory recall.

Your job is to create a helpful, specific description that reminds the patient about their recent interaction with this person.

IMPORTANT Requirements:
- Focus on SPECIFIC, memorable details: names of places, specific topics, concrete events
- Include a time reference when the interaction happened ("3 days ago", "yesterday", "last week")
- Use concrete details, not generic phrases
- Keep it to ONE sentence (15-20 words)
- DO NOT include the person's name or relationship (those are shown separately)
- Start with time reference and action

GOOD examples:
- "Visited 3 days ago and mentioned her new job at Google and the kids' soccer game"
- "Brought groceries yesterday and talked about his camping trip to Yosemite next month"
- "Met last Tuesday at book club to discuss the new Agatha Christie mystery novel"
- "Called last week about Thanksgiving dinner plans and Aunt Mary's health update"

BAD examples (too generic):
- "Just talked about work" ❌
- "Recently discussed family" ❌
- "Last spoke about hobbies" ❌

Output ONLY the description, nothing else."""

    user_prompt = f"""Conversation History for {person_name} ({relationship}):
{aggregated_context}

Generate a specific, memorable AR description:"""

    try:
        logger.info(f"Calling LLM Model #2 (Description Generation) for {person_name}")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        description = await call_llm_with_retry(messages, temperature=0.5, max_tokens=100)
        description = description.strip('"\'')

        logger.info(f"Description generation complete for {person_name}")
        logger.debug(f"Generated description: {description}")

        return description

    except Exception as e:
        logger.error(f"Error calling LLM description generation: {e}")
        return f"Recently interacted with {person_name}"


def deduplicate_sentences(text: str) -> str:
    """Remove duplicate sentences or clauses from text."""
    return format_highlight_summary(text)


def format_highlight_summary(text: str, max_points: int = 3, max_words: int = 25) -> str:
    """Format and cap memory summary into maximum 2-3 concise highlight bullet points (<=25 words)."""
    if not text:
        return ""

    # Split by explicit bullets, newlines, or sentence boundaries
    raw_lines = [line.strip() for line in re.split(r"[\n\r]+|[•\-\*]\s+", text) if line.strip()]
    if not raw_lines or len(raw_lines) == 1:
        raw_lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    unique_points: list[str] = []
    seen_norms: list[str] = []

    for item in raw_lines:
        cleaned = item.lstrip("•-* ").strip()
        if not cleaned or len(cleaned) < 3:
            continue
        norm = re.sub(r"[^\w\s]", "", cleaned.lower())
        norm = " ".join(norm.split())
        if not norm or norm in seen_norms:
            continue
        if any(norm in s or s in norm for s in seen_norms):
            continue
        seen_norms.append(norm)
        unique_points.append(cleaned)
        if len(unique_points) >= max_points:
            break

    if not unique_points:
        return ""

    final_points: list[str] = []
    total_words = 0
    for pt in unique_points:
        words = pt.split()
        if total_words + len(words) > max_words:
            remaining = max_words - total_words
            if remaining >= 3:
                final_points.append(" ".join(words[:remaining]) + "…")
            break
        final_points.append(pt)
        total_words += len(words)

    if not final_points:
        final_points = [unique_points[0]]

    return "\n".join(f"• {pt.lstrip('• ')}" for pt in final_points)


async def summarize_person_memory(
    person_name: str,
    current_summary: str | None,
    transcript: str,
) -> str:
    """Create a grounded rolling memory capped at 2-3 key highlight points."""
    system_prompt = """You are an AR smart glasses memory assistant for an Alzheimer's/dementia patient.
Your job is to maintain a distilled, scannable highlight memory summary for this person that NEVER overwhelms the user.

CRITICAL DISTILLATION & HIGHLIGHT RULES:
- Cap the memory at a MAXIMUM of 2 to 3 key highlight bullet points (e.g. • Highlight 1 \n • Highlight 2).
- Maximum 25 words total across all highlight points.
- When new conversational facts arrive, merge them with prior memory into at most 2-3 concise bullets.
- Discard older transient chit-chat, outdated details, or filler so the summary stays compact and rests comfortably at 2-3 highlights.
- Ignore greetings, polite small talk, and pleasantries.
- Never repeat or duplicate points.
- Use clear, third-person factual phrasing.

Example Output:
• Rhythm (software engineer) met at tech event
• Mentioned working on AI project
• Meeting tomorrow at 3 PM

Output ONLY 2-3 bulleted highlights ('• '), nothing else."""
    user_prompt = f"""Person: {person_name}

Prior memory:
{current_summary or "(none)"}

New real transcript:
{transcript}

Updated memory highlights (max 3 bullets):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_text = await call_llm_with_retry(messages, temperature=0, max_tokens=100)
    cleaned = response_text.strip().strip('"')
    return format_highlight_summary(cleaned)


async def infer_new_person_details(conversation: List[ConversationUtterance]) -> dict:
    """
    Model #3: New Person Inference
    Analyzes a first-time conversation to infer person details.
    """
    conversation_text = "\n".join([
        f"{utt.speaker}: {utt.text}"
        for utt in conversation
    ])

    system_prompt = """You are a person identification AI for a dementia care system.

Your job is to analyze a conversation and infer details about a NEW person the patient just met. Extract only the necessary, critical information.

Extract and return the following in JSON format:
{
  "name": "person's first name",
  "relationship": "relationship to patient (e.g., 'Your daughter', 'Your neighbor', 'Your nurse')",
  "summary": "1-2 sentence summary of only the necessary key details about this person and what was discussed"
}

IMPORTANT:
- Look for the person's name in the dialogue
- Infer relationship from context (daughter, son, friend, caregiver, neighbor, etc.)
- If name isn't mentioned, use a placeholder like "New Visitor" or "Friend"
- Relationship should start with "Your" (e.g., "Your daughter", not just "daughter")
- Summary should capture only essential, necessary details from this conversation (updates, plans, key relations) and completely ignore trivial chit-chat or filler.

Output ONLY valid JSON, nothing else."""

    user_prompt = f"""Analyze this conversation and extract person details:

{conversation_text}

Return JSON with name, relationship, and summary:"""

    try:
        logger.info("Calling LLM Model #3 (New Person Inference)")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        result_text = await call_llm_with_retry(messages, temperature=0.2, max_tokens=300)

        # Handle markdown JSON formatting if present
        if "```" in result_text:
            import re
            match = re.search(r"```(?:json)?(.*?)```", result_text, re.DOTALL)
            if match:
                result_text = match.group(1).strip()

        person_details = json.loads(result_text)

        logger.info(f"New person inferred: {person_details.get('name')} ({person_details.get('relationship')})")

        initial_description = f"First met today. {person_details.get('summary', '')}"
        if len(initial_description) > 100:
            initial_description = initial_description[:97] + "..."

        return {
            "name": person_details.get("name", "New Person"),
            "relationship": person_details.get("relationship", "Unknown relationship"),
            "aggregated_context": person_details.get("summary", "First conversation with this person."),
            "cached_description": initial_description
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from person inference: {e}")
        return {
            "name": "New Person",
            "relationship": "Someone you know",
            "aggregated_context": f"First conversation: {conversation_text[:200]}...",
            "cached_description": "Just met today for the first time"
        }
    except Exception as e:
        logger.error(f"Error inferring new person details: {e}")
        return {
            "name": "New Person",
            "relationship": "Someone you know",
            "aggregated_context": "First conversation with this person.",
            "cached_description": "Just met today for the first time"
        }


async def test_fireworks_connection() -> bool:
    """Test that the LLM API is accessible."""
    try:
        response = await client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5
        )
        logger.info(f"✓ LLM ({provider_name}) connection test successful")
        return True
    except Exception as e:
        logger.error(f"✗ LLM ({provider_name}) connection test failed: {e}")
        return False
