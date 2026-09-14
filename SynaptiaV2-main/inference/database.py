"""MongoDB database operations for person data with resilient local fallback."""

import logging
import os
from datetime import datetime
from typing import Any, Optional

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

import json
from pathlib import Path

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

# Persistent storage fallback path
STORAGE_DIR = Path(__file__).parent / "data"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
IDENTITIES_FILE = STORAGE_DIR / "face_identities.json"
MEMORIES_FILE = STORAGE_DIR / "person_memories.json"
PEOPLE_FILE = STORAGE_DIR / "people.json"

# In-memory storage fallbacks with disk persistence
_mem_identities: dict[str, dict[str, Any]] = {}
_mem_observations: list[dict[str, Any]] = []
_mem_transcripts: list[dict[str, Any]] = []
_mem_person_memories: dict[str, dict[str, Any]] = {}
_mem_people: dict[str, dict[str, Any]] = {}

def _load_persisted_data():
    global _mem_identities, _mem_person_memories, _mem_people
    try:
        if IDENTITIES_FILE.exists():
            with open(IDENTITIES_FILE, "r", encoding="utf-8") as f:
                _mem_identities = json.load(f)
                logger.info("Loaded %d enrolled face identities from persistent storage", len(_mem_identities))
    except Exception as exc:
        logger.warning("Could not load face_identities.json: %s", exc)

    try:
        if MEMORIES_FILE.exists():
            with open(MEMORIES_FILE, "r", encoding="utf-8") as f:
                _mem_person_memories = json.load(f)
    except Exception as exc:
        logger.warning("Could not load person_memories.json: %s", exc)

    try:
        if PEOPLE_FILE.exists():
            with open(PEOPLE_FILE, "r", encoding="utf-8") as f:
                _mem_people = json.load(f)
    except Exception as exc:
        logger.warning("Could not load people.json: %s", exc)

_load_persisted_data()

def _save_identities():
    try:
        with open(IDENTITIES_FILE, "w", encoding="utf-8") as f:
            json.dump(_mem_identities, f, default=str, indent=2)
    except Exception as exc:
        logger.warning("Could not persist face identities: %s", exc)

def _save_memories():
    try:
        with open(MEMORIES_FILE, "w", encoding="utf-8") as f:
            json.dump(_mem_person_memories, f, default=str, indent=2)
    except Exception as exc:
        logger.warning("Could not persist person memories: %s", exc)

def _save_people():
    try:
        with open(PEOPLE_FILE, "w", encoding="utf-8") as f:
            json.dump(_mem_people, f, default=str, indent=2)
    except Exception as exc:
        logger.warning("Could not persist people: %s", exc)

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "dementia_care_db")

# Global MongoDB client and database
_client: Optional[MongoClient] = None
_db: Optional[Database] = None
_db_disabled: bool = False


def get_database() -> Optional[Database]:
    """Get MongoDB database instance with resilient fallback handling."""
    global _client, _db, _db_disabled

    if _db_disabled:
        return None

    if _db is None:
        if not MONGODB_URI:
            logger.warning("MONGODB_URI not set; using local in-memory database fallback.")
            _db_disabled = True
            return None

        logger.info("Connecting to MongoDB Atlas...")
        import certifi
        try:
            client = MongoClient(MONGODB_URI, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=2000)
            client.admin.command("ping")
            _client = client
            _db = _client[MONGODB_DATABASE]
            logger.info("Connected to MongoDB database: %s", MONGODB_DATABASE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("MongoDB Atlas SSL/connection failed (%s). Using local in-memory database fallback.", exc)
            _db_disabled = True
            return None

    return _db


def list_face_identities() -> list[dict[str, Any]]:
    """Return the enrolled descriptor records used by the local browser matcher."""
    try:
        db = get_database()
        if db is not None:
            projection = {"_id": 0, "name": 1, "descriptor": 1, "updated_at": 1}
            return list(db["face_identities"].find({}, projection))
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB list_face_identities failed (%s); using in-memory fallback.", exc)

    return list(_mem_identities.values())


def upsert_face_identity(name: str, descriptor: list[float]) -> dict[str, Any]:
    """Store a user-enrolled face descriptor without storing a camera image."""
    now = datetime.utcnow()
    existing = _mem_identities.get(name, {})
    record = {**existing, "name": name, "descriptor": descriptor, "updated_at": now}
    _mem_identities[name] = record
    _save_identities()

    try:
        db = get_database()
        if db is not None:
            db["face_identities"].update_one(
                {"name": name},
                {"$set": record, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            logger.info("Enrolled local face identity %s in MongoDB Atlas", name)
            return record
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB upsert_face_identity failed (%s); stored in-memory + persistent disk.", exc)

    logger.info("Enrolled local face identity %s (saved to disk)", name)
    return record


def upsert_voice_identity(name: str, voice_descriptor: list[float]) -> dict[str, Any]:
    """Store a user-enrolled acoustic voice descriptor."""
    now = datetime.utcnow()
    existing = _mem_identities.get(name, {})
    record = {**existing, "name": name, "voice_descriptor": voice_descriptor, "voice_updated_at": now}
    _mem_identities[name] = record
    _save_identities()

    try:
        db = get_database()
        if db is not None:
            db["face_identities"].update_one(
                {"name": name},
                {"$set": record, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            logger.info("Enrolled voice identity for %s in MongoDB Atlas", name)
            return record
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB upsert_voice_identity failed (%s); stored in-memory + persistent disk.", exc)

    logger.info("Enrolled voice identity for %s (saved to disk)", name)
    return record


def get_voice_identity(name: str) -> Optional[list[float]]:
    """Retrieve enrolled voice descriptor for a person if available."""
    try:
        db = get_database()
        if db is not None:
            doc = db["face_identities"].find_one({"name": name}, {"_id": 0, "voice_descriptor": 1})
            if doc and "voice_descriptor" in doc:
                return doc["voice_descriptor"]
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB get_voice_identity failed (%s); checking in-memory.", exc)

    record = _mem_identities.get(name)
    return record.get("voice_descriptor") if record else None



def record_transcript(person_name: str | None, text: str) -> dict[str, Any]:
    """Persist a short speech-recognition result for the AI pipeline."""
    record = {
        "person_name": person_name,
        "text": text,
        "created_at": datetime.utcnow(),
    }
    _mem_transcripts.append(record)

    try:
        db = get_database()
        if db is not None:
            db["conversation_observations"].insert_one({**record})
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB record_transcript failed (%s); saved in-memory.", exc)

    return record


def get_person_memory(name: str) -> Optional[dict[str, Any]]:
    """Return the latest saved AI memory for an explicitly enrolled person."""
    try:
        db = get_database()
        if db is not None:
            doc = db["person_memories"].find_one({"name": name}, {"_id": 0})
            if doc:
                return doc
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB get_person_memory failed (%s); checking in-memory.", exc)

    return _mem_person_memories.get(name)


def upsert_person_memory(name: str, summary: str) -> dict[str, Any]:
    """Persist a grounded rolling summary for a named, enrolled person."""
    now = datetime.utcnow()
    record = {"name": name, "summary": summary, "updated_at": now}
    _mem_person_memories[name] = record
    _save_memories()

    try:
        db = get_database()
        if db is not None:
            db["person_memories"].update_one(
                {"name": name},
                {"$set": record, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            return record
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB upsert_person_memory failed (%s); saved in-memory.", exc)

    return record


def get_recent_transcripts(person_name: str, limit: int = 12) -> list[str]:
    """Return stored speech segments in chronological order for memory recovery."""
    try:
        db = get_database()
        if db is not None:
            cursor = (
                db["conversation_observations"]
                .find({"person_name": person_name}, {"_id": 0, "text": 1})
                .sort("created_at", -1)
                .limit(limit)
            )
            res = [item["text"] for item in reversed(list(cursor)) if item.get("text")]
            if res:
                return res
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB get_recent_transcripts failed (%s); checking in-memory.", exc)

    matching = [t["text"] for t in _mem_transcripts if t.get("person_name") == person_name and t.get("text")]
    return matching[-limit:]


def record_face_observation(
    session_id: str,
    face_id: str,
    confidence: float,
    bounding_box: dict[str, float],
    image_data_url: str | None = None,
) -> dict[str, Any]:
    """Persist face observation evidence."""
    now = datetime.utcnow()
    key = {"session_id": session_id, "face_id": face_id}
    record = {**key, "confidence": confidence, "bounding_box": bounding_box, "last_seen": now}
    _mem_observations.append(record)

    try:
        db = get_database()
        if db is not None:
            set_values: dict[str, Any] = {
                "last_seen": now,
                "last_confidence": confidence,
                "last_bounding_box": bounding_box,
            }
            set_on_insert: dict[str, Any] = {
                "session_id": session_id,
                "face_id": face_id,
                "first_seen": now,
            }
            if image_data_url:
                set_on_insert["image_data_url"] = image_data_url

            db["face_observations"].update_one(
                key,
                {
                    "$set": set_values,
                    "$setOnInsert": set_on_insert,
                    "$inc": {"detection_count": 1},
                },
                upsert=True,
            )
            logger.info("Recorded face observation %s/%s in MongoDB Atlas", session_id, face_id)
            return {**key, **set_values}
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB record_face_observation failed (%s); saved in-memory.", exc)

    return record


def get_person_by_id(person_id: str) -> Optional[dict]:
    """Retrieve a person document by person_id."""
    try:
        db = get_database()
        if db is not None:
            person_doc = db["people"].find_one({"person_id": person_id})
            if person_doc:
                return person_doc
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB get_person_by_id failed (%s); checking in-memory.", exc)

    return _mem_people.get(person_id)


def create_person(
    person_id: str,
    name: str,
    relationship: str,
    aggregated_context: str = "",
    cached_description: str = "No previous interactions",
) -> dict:
    """Create a new person document."""
    person_doc = {
        "person_id": person_id,
        "name": name,
        "relationship": relationship,
        "aggregated_context": aggregated_context,
        "cached_description": cached_description,
        "last_updated": datetime.utcnow(),
    }
    _mem_people[person_id] = person_doc

    try:
        db = get_database()
        if db is not None:
            db["people"].update_one(
                {"person_id": person_id},
                {"$setOnInsert": person_doc},
                upsert=True,
            )
            return person_doc
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB create_person failed (%s); saved in-memory.", exc)

    return person_doc


def update_person_context(
    person_id: str,
    aggregated_context: str,
    cached_description: str,
) -> bool:
    """Update a person's aggregated context and cached description."""
    if person_id in _mem_people:
        _mem_people[person_id]["aggregated_context"] = aggregated_context
        _mem_people[person_id]["cached_description"] = cached_description
        _mem_people[person_id]["last_updated"] = datetime.utcnow()

    try:
        db = get_database()
        if db is not None:
            res = db["people"].update_one(
                {"person_id": person_id},
                {
                    "$set": {
                        "aggregated_context": aggregated_context,
                        "cached_description": cached_description,
                        "last_updated": datetime.utcnow(),
                    }
                },
            )
            return res.matched_count > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB update_person_context failed (%s).", exc)

    return person_id in _mem_people


def list_all_people() -> list[dict]:
    """List all people in the database."""
    try:
        db = get_database()
        if db is not None:
            return list(db["people"].find())
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB list_all_people failed (%s); using in-memory.", exc)

    return list(_mem_people.values())


def delete_all_people() -> int:
    """Delete all people from the database."""
    count = len(_mem_people)
    _mem_people.clear()
    try:
        db = get_database()
        if db is not None:
            res = db["people"].delete_many({})
            return res.deleted_count
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB delete_all_people failed (%s).", exc)

    return count


def clear_all_data() -> dict[str, Any]:
    """Completely wipe all enrolled face identities, voiceprints, memories, and transcripts."""
    _mem_identities.clear()
    _mem_observations.clear()
    _mem_transcripts.clear()
    _mem_person_memories.clear()
    _mem_people.clear()

    _save_identities()
    _save_memories()
    _save_people()

    try:
        db = get_database()
        if db is not None:
            db["face_identities"].delete_many({})
            db["person_memories"].delete_many({})
            db["people"].delete_many({})
            db["conversation_observations"].delete_many({})
            db["face_observations"].delete_many({})
    except Exception as exc:  # noqa: BLE001
        logger.warning("MongoDB clear_all_data failed (%s).", exc)

    logger.info("All face identities, voiceprints, and person memories have been wiped clean.")
    return {"cleared": True}


def close_connection():
    """Close MongoDB connection."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db = None
        logger.info("MongoDB connection closed")

