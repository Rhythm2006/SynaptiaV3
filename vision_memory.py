"""
Synaptia V2 — Vision & Multimodal Memory Engine
Provides real-time person context management, face recognition tracking store,
interaction logging, and AI-assisted conversational memory summarization.
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import requests

logger = logging.getLogger(__name__)

# Sample palette for avatars
AVATAR_COLORS = [
    "#6366f1", "#06b6d4", "#10b981", "#f59e0b",
    "#ec4899", "#8b5cf6", "#14b8a6", "#f43f5e"
]

DEFAULT_PROFILES = [
    {
        "id": "person_001",
        "name": "Sarah Jenkins",
        "relationship": "Daughter",
        "description": "Last spoke 2 days ago about her promotion to Lead Architect and the grandchildren visiting this weekend for soccer practice.",
        "avatar_color": "#6366f1",
        "created_at": "2026-03-01T10:00:00",
        "last_seen": "2026-03-08T16:45:00",
        "confidence": 0.96,
        "tags": ["Family", "Frequent", "Priority"],
        "interactions": [
            {
                "ts": "2026-03-08T16:45:00",
                "summary": "Shared news regarding the architecture team expansion. Reminded about Sunday brunch at 11:00 AM.",
                "sentiment": "Positive / Joyful"
            },
            {
                "ts": "2026-03-05T19:20:00",
                "summary": "Discussed travel arrangements for upcoming spring break visit.",
                "sentiment": "Warm"
            }
        ]
    },
    {
        "id": "person_002",
        "name": "Dr. Aris Thorne",
        "relationship": "Neurologist & Physician",
        "description": "Primary cognitive wellness physician. Recommended daily Synaptia logical puzzles and tracking working memory retention.",
        "avatar_color": "#06b6d4",
        "created_at": "2026-02-15T09:30:00",
        "last_seen": "2026-03-06T14:10:00",
        "confidence": 0.94,
        "tags": ["Medical", "Wellness", "Advisor"],
        "interactions": [
            {
                "ts": "2026-03-06T14:10:00",
                "summary": "Reviewed 30-day cognitive puzzle metrics. Noted a 14% improvement in pattern decomposition speed.",
                "sentiment": "Encouraging"
            }
        ]
    },
    {
        "id": "person_003",
        "name": "Marcus Vance",
        "relationship": "Research Partner",
        "description": "Collaborating on multimodal perceptual systems, real-time speaker diarization, and edge AI memory indexing.",
        "avatar_color": "#10b981",
        "created_at": "2026-02-20T11:15:00",
        "last_seen": "2026-03-09T18:00:00",
        "confidence": 0.91,
        "tags": ["Work", "AI Research", "Active"],
        "interactions": [
            {
                "ts": "2026-03-09T18:00:00",
                "summary": "Brainstormed low-latency WebRTC pipelines and Ray-Ban smart glass HUD overlays for instant memory recall.",
                "sentiment": "Focused / Collaborative"
            }
        ]
    }
]


class VisionMemoryEngine:
    """Manages multimodal perceptual memory, contact identities, and conversation summarization."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "")

        # Storage path for persistent profiles
        primary_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "data")
        try:
            os.makedirs(primary_dir, exist_ok=True)
            self.storage_path = os.path.join(primary_dir, "vision_profiles.json")
        except Exception:
            self.storage_path = "/tmp/vision_profiles.json"

        # In-memory dictionary for profiles
        self.people: Dict[str, Dict[str, Any]] = {}
        self.events_log: List[Dict[str, Any]] = []

        # Load default profiles
        for p in DEFAULT_PROFILES:
            self.people[p["id"]] = dict(p)

        self._load_from_disk()

    def _save_to_disk(self):
        """Persist people profiles to disk."""
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.people, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist vision profiles to disk: {e}")

    def _load_from_disk(self):
        """Load stored profiles from disk."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.people.update(data)
            except Exception as e:
                logger.warning(f"Failed to load vision profiles from disk: {e}")

    def list_people(self) -> List[Dict[str, Any]]:
        """Return list of all registered person profiles."""
        return list(self.people.values())

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific profile by ID."""
        return self.people.get(person_id)

    def find_person_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find person by case-insensitive name match."""
        name_clean = name.strip().lower()
        for p in self.people.values():
            if p.get("name", "").strip().lower() == name_clean:
                return p
        return None

    def enroll_person(
        self,
        name: str,
        relationship: str = "Acquaintance",
        description: str = "",
        face_crop: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Enroll a new identity or update an existing one."""
        name = name.strip()
        if not name:
            raise ValueError("Person name cannot be empty")

        existing = self.find_person_by_name(name)
        if existing:
            person_id = existing["id"]
            if relationship:
                existing["relationship"] = relationship
            if description:
                existing["description"] = description
            if face_crop:
                existing["face_thumbnail"] = face_crop
            if tags:
                existing["tags"] = tags
            existing["last_seen"] = datetime.now().isoformat()
            self._log_event("person_updated", {"person_id": person_id, "name": name})
            self._save_to_disk()
            return existing

        person_id = f"person_{int(time.time() * 1000) % 1000000:06d}"
        color_idx = len(self.people) % len(AVATAR_COLORS)
        new_profile = {
            "id": person_id,
            "name": name,
            "relationship": relationship or "Contact",
            "description": description or f"Enrolled on {datetime.now().strftime('%b %d, %Y')}.",
            "avatar_color": AVATAR_COLORS[color_idx],
            "face_thumbnail": face_crop,
            "created_at": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "confidence": 0.95,
            "tags": tags or ["Enrolled"],
            "interactions": []
        }
        self.people[person_id] = new_profile
        self._log_event("person_enrolled", {"person_id": person_id, "name": name})
        self._save_to_disk()
        return new_profile

    def delete_person(self, person_id: str) -> bool:
        """Remove a person from the store."""
        if person_id in self.people:
            deleted = self.people.pop(person_id)
            self._log_event("person_deleted", {"person_id": person_id, "name": deleted.get("name")})
            self._save_to_disk()
            return True
        return False

    def record_interaction(
        self,
        person_id: str,
        transcript: str,
        summary: Optional[str] = None,
        sentiment: Optional[str] = None
    ) -> Dict[str, Any]:
        """Log a conversation interaction for an identified person."""
        person = self.people.get(person_id)
        if not person:
            raise ValueError(f"Person ID '{person_id}' not found")

        if not summary:
            summary = self.generate_memory_summary(person.get("name", "Unknown"), transcript)["summary"]

        interaction = {
            "ts": datetime.now().isoformat(),
            "transcript": transcript,
            "summary": summary,
            "sentiment": sentiment or "Engaged"
        }

        if "interactions" not in person:
            person["interactions"] = []
        person["interactions"].insert(0, interaction)
        person["last_seen"] = datetime.now().isoformat()
        person["description"] = summary  # Update latest context

        self._log_event("interaction_recorded", {"person_id": person_id, "summary": summary})
        self._save_to_disk()
        return interaction

    def generate_memory_summary(self, person_name: str, transcript: str) -> Dict[str, Any]:
        """
        Use Groq LLM (or robust heuristic fallback) to distill spoken transcripts
        into actionable cognitive memory summaries and talking point recall cues.
        """
        transcript = transcript.strip()
        if not transcript:
            return {
                "summary": "Brief observation with no extended conversation recorded.",
                "key_points": ["Visual presence confirmed"],
                "sentiment": "Neutral",
                "recall_cues": [f"Ask {person_name} how their day is going."]
            }

        # Try Groq AI generation first
        if self.groq_api_key:
            try:
                system_prompt = (
                    "You are Synaptia V2's Multimodal Perceptual Memory Engine. "
                    "Analyze spoken conversation transcripts between the user and a recognized person. "
                    "Provide a clear, high-signal 1-2 sentence memory summary, 2-3 key bullet points, "
                    "perceived emotional sentiment, and 2 proactive conversational recall cues for future interactions.\n"
                    "Respond STRICTLY in JSON format with keys:\n"
                    '{"summary": "...", "key_points": ["...", "..."], "sentiment": "...", "recall_cues": ["...", "..."]}'
                )
                user_content = f"Person Name: {person_name}\nSpoken Transcript:\n\"{transcript}\""

                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.groq_api_key}"},
                    json={
                        "model": os.getenv("GROQ_MODEL", "groq/compound-mini"),
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        "temperature": 0.3,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=20
                )
                if resp.status_code == 200:
                    data = resp.json()["choices"][0]["message"]["content"]
                    parsed = json.loads(data)
                    return {
                        "summary": parsed.get("summary", transcript[:120] + "..."),
                        "key_points": parsed.get("key_points", ["Conversation registered"]),
                        "sentiment": parsed.get("sentiment", "Positive"),
                        "recall_cues": parsed.get("recall_cues", [f"Follow up with {person_name}."])
                    }
            except Exception as e:
                logger.warning(f"Groq memory summarization error, falling back to heuristic: {e}")

        # Heuristic Rule-Based Fallback
        words = transcript.split()
        summary_text = (
            f"Discussed {len(words)} words covering key topics with {person_name}. "
            f"Last point noted: '{words[-10:] if len(words) > 10 else transcript}'"
        )
        if len(transcript) > 140:
            summary_text = transcript[:140] + "..."

        return {
            "summary": summary_text,
            "key_points": [
                f"Engaged in active dialogue ({len(words)} words)",
                f"Recorded at {datetime.now().strftime('%H:%M:%S')}"
            ],
            "sentiment": "Neutral / Positive",
            "recall_cues": [
                f"Mention the recent discussion regarding '{words[0] if words else 'topics'}'",
                f"Check in on next scheduled milestone with {person_name}"
            ]
        }

    def simulate_event(self, person_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a simulated detection event for interactive testing & demo."""
        if not self.people:
            return {"type": "idle", "message": "No profiles enrolled"}

        if person_id and person_id in self.people:
            target = self.people[person_id]
        else:
            # Pick first available or rotate
            target = list(self.people.values())[0]

        event = {
            "type": "person_detected",
            "person_id": target["id"],
            "name": target["name"],
            "relationship": target["relationship"],
            "description": target["description"],
            "confidence": target.get("confidence", 0.95),
            "timestamp": datetime.now().isoformat(),
            "avatar_color": target.get("avatar_color", "#6366f1")
        }
        self._log_event("simulation_event", event)
        return event

    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Append internal event log."""
        self.events_log.insert(0, {
            "type": event_type,
            "data": data,
            "ts": datetime.now().isoformat()
        })
        if len(self.events_log) > 100:
            self.events_log = self.events_log[:100]

    def get_status(self) -> Dict[str, Any]:
        """Return system status for V2 module."""
        return {
            "status": "operational",
            "version": "2.0.0",
            "module": "Synaptia Multimodal Perception & Memory",
            "enrolled_count": len(self.people),
            "ai_summarizer": "groq-llama-3.3-70b" if self.groq_api_key else "heuristic-engine",
            "timestamp": datetime.now().isoformat()
        }
