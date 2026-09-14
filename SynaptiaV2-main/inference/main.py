"""Main inference service for AR glasses - handles two event types."""

import asyncio
import base64
import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import AsyncGenerator, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from database import (
    create_person,
    get_person_by_id,
    get_person_memory,
    get_recent_transcripts,
    get_voice_identity,
    list_face_identities,
    record_face_observation,
    record_transcript,
    upsert_face_identity,
    upsert_person_memory,
    upsert_voice_identity,
    update_person_context,
)
from voice_verifier import verify_speaker_voice
from fireworks_client import (
    GROQ_API_KEY,
    aggregate_conversation_context,
    deduplicate_sentences,
    generate_ar_description,
    infer_new_person_details,
    summarize_person_memory,
)
from models import ConversationEvent, InferenceResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference_service")

WHISPER_HALLUCINATIONS = [
    "subtitles by",
    "thank you for watching",
    "thanks for watching",
    "thank you very much",
    "thank you",
    "thanks",
    "amara.org",
    "subscribe to",
    "like and subscribe",
    "bye bye",
    "goodbye",
    "see you next time",
    "see you later",
    "transcription by",
    "translated by",
    "all rights reserved",
    "farcaster",
    "the end",
]

# Single-word or noise tokens Whisper hallucinates when processing silence/background room hiss
SILENCE_PHANTOM_TOKENS = {
    "you", "you.", "thank", "thanks", "okay", "ok", "so", "music", "applause",
    "bye", "hello", "hi", "yes", "no", "silence", "watching", "subscribe",
    "...", "yeah", "uh", "um", "ah"
}


def deduplicate_text_phrases(text: str) -> str:
    """Clean Whisper output of repetitive phrases, hallucination artifacts, and silence phantoms."""
    if not text:
        return ""

    lowered = text.lower().strip()

    # Filter bracketed/parenthetical sound cues like [Music], (Applause), [Silence]
    if re.fullmatch(r"^[\[\(].*?[\]\)]\.?$", lowered):
        return ""

    # Filter exact single-token phantom hallucinations
    normalized_single = re.sub(r"[^\w]", "", lowered)
    if normalized_single in SILENCE_PHANTOM_TOKENS:
        return ""

    for artifact in WHISPER_HALLUCINATIONS:
        if artifact in lowered:
            # If the entire phrase is just the hallucination, discard completely
            if len(lowered.replace(artifact, "").strip()) < 8:
                return ""

    cleaned_sentences = deduplicate_sentences(text)
    if not cleaned_sentences:
        return ""

    words = cleaned_sentences.split()
    # Reject single-word utterances from ambient noise
    if len(words) < 2:
        return ""

    deduped_words = []
    for w in words:
        norm_w = re.sub(r"[^\w]", "", w.lower())
        if deduped_words and norm_w:
            prev_norm = re.sub(r"[^\w]", "", deduped_words[-1].lower())
            if norm_w == prev_norm:
                continue
        deduped_words.append(w)

    result = " ".join(deduped_words).strip()
    return result if len(result.split()) >= 2 else ""



app = FastAPI(title="Inference Service - AR Glasses")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connected SSE clients — each gets their own queue so events aren't lost
_client_queues: list[asyncio.Queue[InferenceResult]] = []

# An external metadata stream is optional. It is deliberately disabled by
# default so development never fabricates people or conversations from the
# simulator. Set METADATA_SERVICE_URL explicitly only for a real producer.
METADATA_SERVICE_URL = os.getenv("METADATA_SERVICE_URL", "http://localhost:9000/stream/conversation")
_memory_ai_retry_at: datetime | None = None


class BoundingBoxPayload(BaseModel):
    originX: float = Field(ge=0)
    originY: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class FaceObservationPayload(BaseModel):
    session_id: str = Field(min_length=1, max_length=120)
    face_id: str = Field(min_length=1, max_length=120)
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: dict[str, float]
    image_data_url: str | None = Field(default=None, max_length=1_500_000)


class FaceIdentityPayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    descriptor: list[float] = Field(min_items=128, max_items=128)


class VoiceIdentityPayload(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    descriptor: list[float] = Field(min_items=16, max_items=512)


class TranscriptPayload(BaseModel):
    person_name: str | None = Field(default=None, max_length=80)
    text: str = Field(min_length=1, max_length=2000)


class AudioTranscriptionPayload(BaseModel):
    audio_base64: str = Field(min_length=1, max_length=14_000_000)
    filename: str = Field(default="recording.webm", min_length=1, max_length=120)
    content_type: str = Field(default="audio/webm", min_length=1, max_length=100)
    person_name: str | None = Field(default=None, max_length=80)
    voice_descriptor: list[float] | None = Field(default=None)



async def transcribe_audio(audio: bytes, filename: str, content_type: str | None) -> str:
    """Transcribe one browser-recorded audio segment with the configured Groq API."""
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY is not configured for transcription; returning empty transcript.")
        return ""
    if not audio or len(audio) < 2500:  # Audio shorter than ~1s is dropped immediately
        return ""

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    files = {"file": (filename or "recording.webm", audio, content_type or "audio/webm")}
    data = {
        # Turbo Whisper provides sub-200ms latency for near-instant conversational summaries
        "model": "whisper-large-v3-turbo",
        "response_format": "verbose_json",
        "language": "en",
        "temperature": "0",
        "prompt": "Conversational memory log for smart glasses: user speaks clearly about daily activities, ideas, and plans.",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
            )
        if response.is_error:
            logger.error("Speech provider rejected audio (%s): %s", response.status_code, response.text[:300])
            return ""

        res_json = response.json()
        raw_text = str(res_json.get("text", "")).strip()

        # Quality check segments from verbose_json:
        # Reject complete silence / pure noise hallucinations
        segments = res_json.get("segments", [])
        if segments:
            for seg in segments:
                no_speech = seg.get("no_speech_prob", 0.0)
                avg_logprob = seg.get("avg_logprob", 0.0)
                compression = seg.get("compression_ratio", 1.0)
                if no_speech > 0.40 or avg_logprob < -0.95 or compression > 2.4:
                    logger.info("Discarding non-speech hallucination (no_speech=%.3f, logprob=%.3f, comp=%.2f, text='%s')", no_speech, avg_logprob, compression, raw_text)
                    return ""

        cleaned = deduplicate_text_phrases(raw_text)
        if len(cleaned.split()) < 2:
            logger.info("Discarding single-word token: '%s'", cleaned)
            return ""

        return cleaned

    except Exception as exc:
        logger.warning("Speech transcription request error: %s", exc)
        return ""




async def persist_transcript_memory(person_name: str | None, text: str) -> tuple[dict, str | None]:
    """Store a final transcript, then update the named person's real memory."""
    cleaned_text = deduplicate_text_phrases(text)
    if not cleaned_text:
        existing = get_person_memory(person_name) if person_name else None
        return {"person_name": person_name, "text": "", "created_at": datetime.utcnow()}, (existing.get("summary") if existing else None)

    if person_name:
        recent = get_recent_transcripts(person_name, limit=8)  # check more history
        norm_new = re.sub(r"[^\w\s]", "", cleaned_text.lower()).strip()
        for prev in recent:
            norm_prev = re.sub(r"[^\w\s]", "", prev.lower()).strip()
            # Exact match OR new text is a substring of something already stored
            if norm_new == norm_prev or (len(norm_new) > 4 and norm_new in norm_prev):
                logger.info("Ignoring duplicate transcript for %s: '%s'", person_name, cleaned_text)
                existing = get_person_memory(person_name)
                return {"person_name": person_name, "text": cleaned_text, "created_at": datetime.utcnow()}, (existing.get("summary") if existing else None)

    record = record_transcript(person_name, cleaned_text)
    if not person_name:
        return record, None

    summary = await build_person_memory(person_name)
    if summary:
        upsert_person_memory(person_name, summary)
    return record, summary


def is_displayable_memory(memory: dict | None) -> bool:
    """Do not display the old transcript fallback as a user-facing summary."""
    summary = memory.get("summary") if memory else None
    return bool(summary and not summary.startswith("Recent conversation:"))


def extractive_memory_summary(text: str, previous_summary: str | None) -> str | None:
    """Return the existing summary unchanged, or build a compact first-time cue.

    Critically: never APPEND raw transcript text to an existing summary — that
    is exactly the cloning/duplication behaviour the user reported.
    """
    # If we already have a real displayable summary, keep it as-is.
    # Appending new transcript snippets to it causes the duplicate-sentence bug.
    if previous_summary:
        return previous_summary

    cleaned_text = deduplicate_text_phrases(text)
    if not cleaned_text:
        return None

    sentences = [
        " ".join(sentence.split())
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned_text)
        if len(sentence.split()) >= 4
    ]
    if not sentences:
        return None

    important_terms = {
        "appointment", "bring", "call", "coffee", "doctor", "library", "meet",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "today", "tomorrow", "visit", "will", "am", "pm",
    }
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: (
            sum(word.strip(".,!?;:").lower() in important_terms for word in item[1].split()),
            min(len(item[1]), 180),
        ),
        reverse=True,
    )
    selected_indexes = sorted(index for index, _ in ranked[:1])
    cue = " ".join(sentences[index] for index in selected_indexes)
    if len(cue) > 200:
        cue = f"{cue[:197].rstrip()}…"
    return deduplicate_sentences(cue) or None


async def build_person_memory(person_name: str) -> str | None:
    """Prefer an AI summary, with a clearly grounded real-transcript fallback."""
    global _memory_ai_retry_at

    existing_memory = get_person_memory(person_name)
    existing_summary = existing_memory.get("summary") if is_displayable_memory(existing_memory) else None
    latest_transcript = get_recent_transcripts(person_name, limit=1)

    # Only call the LLM if we are NOT currently in a rate-limit backoff window.
    ai_available = (_memory_ai_retry_at is None) or (datetime.utcnow() >= _memory_ai_retry_at)
    if ai_available and latest_transcript:
        try:
            summary = await summarize_person_memory(
                person_name,
                existing_summary,
                latest_transcript[-1],
            )
            if summary:
                _memory_ai_retry_at = None  # clear backoff on success
                return summary
        except Exception:  # noqa: BLE001
            # Audio-to-text (Whisper) and LLM summarization use separate quotas.
            # Back off LLM requests for 10 minutes without blocking transcription.
            _memory_ai_retry_at = datetime.utcnow() + timedelta(minutes=10)
            logger.warning("AI memory summary unavailable; retaining the last concise memory.")

    # LLM is optional. Return the existing AI summary if we have one, otherwise
    # generate a lightweight extractive cue from the latest real transcript.
    return existing_summary or extractive_memory_summary(
        latest_transcript[-1] if latest_transcript else "",
        None,
    )


async def broadcast_result(result: InferenceResult) -> None:
    """Push a result to ALL connected SSE clients."""
    dead: list[asyncio.Queue] = []
    for q in _client_queues:
        try:
            # Bounded put — if client is slow, drop the oldest item
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(result)
        except Exception:  # noqa: BLE001
            dead.append(q)
    for q in dead:
        _client_queues.remove(q)


def safe_get_person(person_id: str) -> Optional[dict]:
    try:
        return get_person_by_id(person_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Database lookup failed for %s: %s", person_id, exc)
        return None


def handle_person_detected(event: ConversationEvent) -> InferenceResult:
    """
    Handle PERSON_DETECTED event - return AR display information from MongoDB.
    """
    # Query MongoDB for person data
    person_doc = safe_get_person(event.person_id)

    latest_utterance = None
    if event.conversation:
        try:
            latest_utterance = event.conversation[-1].text
        except Exception:  # noqa: BLE001
            latest_utterance = None

    if person_doc:
        result = InferenceResult(
            person_id=event.person_id,
            name=person_doc["name"],
            relationship=person_doc["relationship"],
            description=person_doc["cached_description"]
        )
        logger.info(f"Person detected: {person_doc['name']} ({event.person_id})")
    else:
        person_label = event.person_id or "speaker_unknown"
        friendly_name = person_label.replace("_", " ").title()
        if person_label and person_label.startswith("speaker_"):
            suffix = person_label[8:]
            if suffix.isdigit():
                friendly_name = f"Speaker {suffix}"
        description = latest_utterance or "No previous interactions"
        # Person not found in database
        result = InferenceResult(
            person_id=person_label,
            name=friendly_name,
            relationship="Unidentified speaker",
            description=description,
        )
        logger.warning(f"Person not found in database: {event.person_id}")

    return result


async def handle_conversation_end(event: ConversationEvent) -> None:
    """
    Handle CONVERSATION_END event - store conversation for future reference.

    Two scenarios:
    1. Existing person: Update their context and description
    2. New person: Infer their details from conversation and create entry

    Uses Fireworks.ai models for both scenarios.
    After processing, pushes an updated InferenceResult so the frontend
    gets the AI-generated summary immediately.
    """
    if not event.conversation:
        logger.warning(f"CONVERSATION_END event for {event.person_id} has no conversation data")
        return

    # Get current person data from MongoDB
    person_doc = safe_get_person(event.person_id)

    # Scenario 1: NEW PERSON - Infer details from conversation
    if not person_doc:
        logger.info(f"🆕 NEW PERSON DETECTED: {event.person_id}")
        logger.info(f"Analyzing first conversation ({len(event.conversation)} utterances) to infer details...")

        try:
            # Call Fireworks Model #3: Infer person details
            inferred_details = await infer_new_person_details(event.conversation)

            try:
                create_person(
                    person_id=event.person_id,
                    name=inferred_details["name"],
                    relationship=inferred_details["relationship"],
                    aggregated_context=inferred_details["aggregated_context"],
                    cached_description=inferred_details["cached_description"],
                )

                logger.info(
                    f"✓ Created new person: {inferred_details['name']} ({inferred_details['relationship']})"
                )
                logger.info(f"  Description: {inferred_details['cached_description']}")
            except Exception as create_exc:  # noqa: BLE001
                logger.warning(
                    "Could not persist new person %s: %s", event.person_id, create_exc
                )

            # Push the new person result to the frontend immediately
            result = InferenceResult(
                person_id=event.person_id,
                name=inferred_details["name"],
                relationship=inferred_details["relationship"],
                description=inferred_details["cached_description"],
            )
            await broadcast_result(result)
            logger.info(f"📢 Pushed new person result to frontend: {inferred_details['name']}")
            return

        except Exception as e:
            logger.error(f"Error inferring new person details: {e}")
            logger.error(f"Conversation will not be stored for {event.person_id}")
            return

    # Scenario 2: EXISTING PERSON - Update with new conversation
    logger.info(
        "Processing conversation end for %s (%s)",
        person_doc["name"],
        event.person_id,
    )
    logger.info(f"Conversation: {len(event.conversation)} utterances")

    try:
        # Call Fireworks Model #1: Aggregate conversation context
        updated_context = await aggregate_conversation_context(
            person_name=person_doc["name"],
            current_context=person_doc["aggregated_context"],
            new_conversation=event.conversation,
        )

        # Call Fireworks Model #2: Generate AR description
        new_description = await generate_ar_description(
            person_name=person_doc["name"],
            relationship=person_doc["relationship"],
            aggregated_context=updated_context,
        )

        try:
            updated = update_person_context(
                person_id=event.person_id,
                aggregated_context=updated_context,
                cached_description=new_description,
            )
        except Exception as update_exc:  # noqa: BLE001
            logger.warning(
                "Could not update person %s: %s", event.person_id, update_exc
            )
            return

        if updated:
            logger.info(
                f"✓ Successfully updated {person_doc['name']} with AI-generated content"
            )
            logger.info(f"  New context: {updated_context[:100]}...")
            logger.info(f"  New description: {new_description}")

            # Push the updated result to the frontend immediately
            result = InferenceResult(
                person_id=event.person_id,
                name=person_doc["name"],
                relationship=person_doc["relationship"],
                description=new_description,
            )
            await broadcast_result(result)
            logger.info(f"📢 Pushed updated result to frontend: {person_doc['name']}")
        else:
            logger.error(f"Failed to update MongoDB for person {event.person_id}")

    except Exception as e:
        logger.error(f"Error processing conversation with Fireworks.ai: {e}")
        logger.error(f"Conversation will not be stored for {event.person_id}")


async def consume_metadata_stream():
    """Background task to consume SSE from metadata service and process events."""
    logger.info(f"Starting metadata stream consumer from {METADATA_SERVICE_URL}")

    retry_delay = 5
    max_retry_delay = 60

    while True:
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", METADATA_SERVICE_URL) as response:
                    response.raise_for_status()
                    logger.info("Connected to metadata stream")
                    retry_delay = 5  # Reset retry delay on successful connection

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            try:
                                event_data = json.loads(data)
                                event = ConversationEvent(**event_data)

                                # Auto-generate timestamp if not provided
                                if not event.timestamp:
                                    event.timestamp = datetime.utcnow()

                                logger.info(f"Received {event.event_type} event for {event.person_id}")

                                # Route to appropriate handler based on event type
                                if event.event_type == "PERSON_DETECTED":
                                    result = handle_person_detected(event)
                                    # Broadcast result to connected AR glasses clients
                                    await broadcast_result(result)

                                elif event.event_type == "CONVERSATION_END":
                                    await handle_conversation_end(event)
                                    # No result to stream - just storage

                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse event data: {e}")
                            except Exception as e:
                                logger.error(f"Error processing event: {e}")

        except httpx.ConnectError:
            logger.error(f"Cannot connect to metadata service. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)
        except Exception as e:
            logger.error(f"Unexpected error in metadata consumer: {e}")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, max_retry_delay)


async def generate_inference_results() -> AsyncGenerator[dict, None]:
    """Generate SSE events from processed inference results."""
    # Each connected client gets its own bounded queue
    client_queue: asyncio.Queue[InferenceResult] = asyncio.Queue(maxsize=10)
    _client_queues.append(client_queue)
    logger.info("New client connected to inference stream (total: %d)", len(_client_queues))

    try:
        while True:
            try:
                result = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                yield {
                    "event": "inference",
                    "data": result.model_dump_json(),
                }
            except asyncio.TimeoutError:
                yield {
                    "comment": "keepalive"
                }
    except asyncio.CancelledError:
        logger.info("Client disconnected from inference stream")
        raise
    finally:
        if client_queue in _client_queues:
            _client_queues.remove(client_queue)


@app.on_event("startup")
async def startup_event():
    """Start background tasks on application startup."""
    logger.info("Starting inference service for AR glasses...")
    if METADATA_SERVICE_URL:
        asyncio.create_task(consume_metadata_stream())
    else:
        logger.info("No metadata stream configured; simulator input is disabled.")


@app.get("/stream/inference")
async def stream_inference():
    """SSE endpoint that streams processed inference results."""
    return EventSourceResponse(generate_inference_results())


@app.post("/api/faces/observations", status_code=201)
async def store_face_observation(payload: FaceObservationPayload):
    """Store a confirmed browser-side face observation in MongoDB.

    This endpoint records detection evidence only. Identity remains the output
    of the conversation/AI pipeline, so a detected face is never falsely
    labelled as a known person merely because it was captured.
    """
    try:
        record = record_face_observation(
            session_id=payload.session_id,
            face_id=payload.face_id,
            confidence=payload.confidence,
            bounding_box=payload.bounding_box if isinstance(payload.bounding_box, dict) else payload.bounding_box.model_dump(),
            image_data_url=payload.image_data_url,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not store face observation %s", payload.face_id)
        raise HTTPException(
            status_code=503,
            detail="Face storage is unavailable. Check MongoDB configuration.",
        ) from exc

    return {"stored": True, "observation": record}


@app.get("/api/face-identities")
async def get_face_identities():
    """Return browser-matchable descriptors for people who explicitly enrolled."""
    try:
        return {"identities": list_face_identities()}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not load enrolled face identities")
        raise HTTPException(status_code=503, detail="Face identity storage is unavailable.") from exc


@app.put("/api/face-identities")
async def enroll_face_identity(payload: FaceIdentityPayload):
    """Create or replace an enrolled face descriptor for a supplied name."""
    try:
        record = upsert_face_identity(payload.name.strip(), payload.descriptor)
        return {"enrolled": True, "identity": record}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not enroll face identity %s", payload.name)
        raise HTTPException(status_code=503, detail="Face identity storage is unavailable.") from exc


@app.put("/api/voice-identities")
async def enroll_voice_identity(payload: VoiceIdentityPayload):
    """Create or replace an enrolled acoustic voice descriptor for a supplied name."""
    try:
        record = upsert_voice_identity(payload.name.strip(), payload.descriptor)
        return {"enrolled": True, "identity": record}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not enroll voice identity %s", payload.name)
        raise HTTPException(status_code=503, detail="Voice identity storage is unavailable.") from exc


@app.get("/api/voice-identities/{person_name}")
async def get_person_voice_identity(person_name: str):
    """Retrieve enrolled voice descriptor status for a person."""
    desc = get_voice_identity(person_name.strip())
    return {"name": person_name, "has_voiceprint": desc is not None, "descriptor": desc}


@app.post("/api/transcripts", status_code=201)
async def store_transcript(payload: TranscriptPayload):
    """Store a final browser speech-recognition phrase for downstream AI use."""
    try:
        record, summary = await persist_transcript_memory(payload.person_name, payload.text.strip())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not store speech-recognition result")
        raise HTTPException(status_code=503, detail="Transcript storage is unavailable.") from exc
    return {"stored": True, "transcript": record, "summary": summary}


@app.post("/api/transcriptions", status_code=201)
async def create_transcription(payload: AudioTranscriptionPayload):
    """Transcribe a real browser microphone segment and store its result."""
    if not payload.content_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="A browser audio recording is required.")
    try:
        recording = base64.b64decode(payload.audio_base64, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="The audio recording is not valid base64.") from exc
    if len(recording) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio segment is too large.")

    # Speaker Verification: If the person has an enrolled voiceprint, verify candidate voice
    if payload.person_name:
        enrolled_voice = get_voice_identity(payload.person_name.strip())
        if enrolled_voice and payload.voice_descriptor:
            is_match, sim_score = verify_speaker_voice(
                payload.voice_descriptor,
                enrolled_voice,
                threshold=0.55,
            )
            if not is_match:
                logger.info(
                    "Bystander / unrecognized voice filtered for %s (similarity=%.4f < 0.55)",
                    payload.person_name,
                    sim_score,
                )
                return {
                    "stored": False,
                    "text": "",
                    "reason": "Bystander voice filtered out (unverified speaker)",
                    "verified": False,
                    "similarity": sim_score,
                }


    try:
        text = await transcribe_audio(recording, payload.filename, payload.content_type)
    except httpx.HTTPStatusError as exc:
        logger.exception("Speech provider rejected the recording")
        raise HTTPException(status_code=502, detail="Speech provider rejected the recording.") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Speech transcription failed")
        raise HTTPException(status_code=502, detail="Speech transcription failed.") from exc

    if not text:
        return {"stored": False, "text": "", "reason": "No speech detected"}

    try:
        record, summary = await persist_transcript_memory(payload.person_name, text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not store transcription")
        raise HTTPException(status_code=503, detail="Transcription succeeded but storage is unavailable.") from exc

    return {"stored": True, "text": text, "summary": summary, "transcript": record, "verified": True}



@app.get("/api/person-memories/{person_name}")
async def get_person_memory_summary(person_name: str):
    """Return saved memory; recover it from prior named transcripts if needed."""
    name = person_name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="A person name is required.")

    try:
        memory = get_person_memory(name)
        if is_displayable_memory(memory):
            return {"memory": memory}

        transcript_history = get_recent_transcripts(name)
        if not transcript_history:
            return {"memory": None}

        summary = await build_person_memory(name)
        if not summary:
            return {"memory": None}
        return {"memory": upsert_person_memory(name, summary)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not load memory for %s", name)
        raise HTTPException(status_code=503, detail="Person memory is unavailable.") from exc


@app.get("/api/persons")
async def list_persons():
    """REST endpoint: list all known persons from MongoDB."""
    try:
        from database import list_all_people
        people = list_all_people()
        return [
            {
                "person_id": p["person_id"],
                "name": p["name"],
                "relationship": p["relationship"],
                "description": p.get("cached_description", ""),
            }
            for p in people
        ]
    except Exception as e:
        logger.error(f"Error listing persons: {e}")
        return []


@app.get("/api/person/{person_id}")
async def get_person(person_id: str):
    """REST endpoint: get a specific person from MongoDB."""
    person_doc = safe_get_person(person_id)
    if not person_doc:
        return {"error": "Person not found"}
    return {
        "person_id": person_doc["person_id"],
        "name": person_doc["name"],
        "relationship": person_doc["relationship"],
        "description": person_doc.get("cached_description", ""),
    }


@app.post("/api/reset-all")
async def reset_all():
    """Clear all face identities, voiceprints, conversation transcripts, and person memories."""
    try:
        from database import clear_all_data
        result = clear_all_data()
        return {"status": "ok", "message": "All face data, voiceprints, transcripts, and memories have been wiped clean.", **result}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to clear data")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "inference_service",
        "connected_clients": len(_client_queues)
    }



@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "inference_service",
        "version": "0.3.0",
        "focus": "ar_glasses_two_event_types",
        "endpoints": {
            "inference_stream": "/stream/inference",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
