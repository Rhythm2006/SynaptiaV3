"""
Synaptia V2 — Unit Test Suite
Tests for VisionMemoryEngine and Synaptia V2 Flask API endpoints.
"""

import sys
import os
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_memory import VisionMemoryEngine
from app import app


# ═══════════════════════════════════════════════════════════════════════════════
#  Vision Memory Engine Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVisionMemoryEngine:

    def setup_method(self):
        self.engine = VisionMemoryEngine()

    def test_default_profiles_seeded(self):
        people = self.engine.list_people()
        assert len(people) >= 3
        names = [p["name"] for p in people]
        assert "Sarah Jenkins" in names
        assert "Dr. Aris Thorne" in names
        assert "Marcus Vance" in names

    def test_get_person_existing(self):
        person = self.engine.get_person("person_001")
        assert person is not None
        assert person["name"] == "Sarah Jenkins"
        assert person["relationship"] == "Daughter"

    def test_get_person_missing(self):
        person = self.engine.get_person("non_existent_999")
        assert person is None

    def test_find_person_by_name(self):
        person = self.engine.find_person_by_name("sarah jenkins")
        assert person is not None
        assert person["id"] == "person_001"

    def test_enroll_new_person(self):
        new_p = self.engine.enroll_person(
            name="Professor Katherine Bell",
            relationship="Mentor",
            description="Leading specialist in cognitive neurology and neural plasticity.",
            tags=["Mentor", "Academic"]
        )
        assert new_p["id"].startswith("person_")
        assert new_p["name"] == "Professor Katherine Bell"
        assert new_p["relationship"] == "Mentor"
        assert "Mentor" in new_p["tags"]

        # Check retrieval
        retrieved = self.engine.get_person(new_p["id"])
        assert retrieved is not None
        assert retrieved["name"] == "Professor Katherine Bell"

    def test_enroll_existing_person_updates(self):
        updated = self.engine.enroll_person(
            name="Sarah Jenkins",
            relationship="Eldest Daughter",
            description="Updated notes: planning weekend visit."
        )
        assert updated["id"] == "person_001"
        assert updated["relationship"] == "Eldest Daughter"
        assert "weekend visit" in updated["description"]

    def test_enroll_empty_name_raises_error(self):
        with pytest.raises(ValueError):
            self.engine.enroll_person(name="   ")

    def test_delete_person(self):
        # Create temp person to delete
        temp = self.engine.enroll_person(name="Temp Contact")
        temp_id = temp["id"]

        assert self.engine.get_person(temp_id) is not None
        deleted = self.engine.delete_person(temp_id)
        assert deleted is True
        assert self.engine.get_person(temp_id) is None

        # Deleting again returns False
        assert self.engine.delete_person(temp_id) is False

    def test_record_interaction(self):
        interaction = self.engine.record_interaction(
            person_id="person_001",
            transcript="We talked about the summer vacation plans and travel tickets.",
            summary="Discussed summer travel plans.",
            sentiment="Excited"
        )
        assert interaction["summary"] == "Discussed summer travel plans."
        assert interaction["sentiment"] == "Excited"

        person = self.engine.get_person("person_001")
        assert len(person["interactions"]) >= 1
        assert person["interactions"][0]["summary"] == "Discussed summer travel plans."

    def test_generate_memory_summary_heuristic(self):
        summary_data = self.engine.generate_memory_summary(
            person_name="Marcus Vance",
            transcript="We discussed multimodal neural embeddings and optimizing real-time latency."
        )
        assert "summary" in summary_data
        assert "key_points" in summary_data
        assert "sentiment" in summary_data
        assert "recall_cues" in summary_data
        assert len(summary_data["key_points"]) > 0

    def test_generate_memory_summary_empty(self):
        summary_data = self.engine.generate_memory_summary("Test Contact", "")
        assert "Brief observation" in summary_data["summary"]

    def test_simulate_event(self):
        event = self.engine.simulate_event("person_001")
        assert event["type"] == "person_detected"
        assert event["person_id"] == "person_001"
        assert event["name"] == "Sarah Jenkins"

    def test_system_status(self):
        status = self.engine.get_status()
        assert status["status"] == "operational"
        assert status["version"] == "2.0.0"
        assert status["enrolled_count"] >= 3


# ═══════════════════════════════════════════════════════════════════════════════
#  Flask API Integration Tests for V2 Routes
# ═══════════════════════════════════════════════════════════════════════════════

class TestV2ApiRoutes:

    def setup_method(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_api_v2_status(self):
        res = self.client.get("/api/v2/status")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "operational"
        assert "enrolled_count" in data

    def test_api_v2_people_get(self):
        res = self.client.get("/api/v2/people")
        assert res.status_code == 200
        data = res.get_json()
        assert "people" in data
        assert data["count"] >= 3

    def test_api_v2_people_post_success(self):
        payload = {
            "name": "Dr. Clara Oswald",
            "relationship": "Research Collaborator",
            "description": "Specialist in neurological linguistics."
        }
        res = self.client.post("/api/v2/people", json=payload)
        assert res.status_code == 201
        data = res.get_json()
        assert data["name"] == "Dr. Clara Oswald"
        assert data["relationship"] == "Research Collaborator"

    def test_api_v2_people_post_missing_name(self):
        res = self.client.post("/api/v2/people", json={"relationship": "Friend"})
        assert res.status_code == 400
        data = res.get_json()
        assert "error" in data

    def test_api_v2_person_detail_get_and_delete(self):
        # Enroll a test contact
        res = self.client.post("/api/v2/people", json={"name": "Delete Me Test", "relationship": "Test"})
        person_id = res.get_json()["id"]

        # Get details
        get_res = self.client.get(f"/api/v2/people/{person_id}")
        assert get_res.status_code == 200
        assert get_res.get_json()["name"] == "Delete Me Test"

        # Delete
        del_res = self.client.delete(f"/api/v2/people/{person_id}")
        assert del_res.status_code == 200

        # Get again -> 404
        assert self.client.get(f"/api/v2/people/{person_id}").status_code == 404

    def test_api_v2_memory_summary(self):
        payload = {
            "person_name": "Sarah Jenkins",
            "transcript": "We discussed organizing the soccer gear and packing lunch for Sunday.",
            "person_id": "person_001"
        }
        res = self.client.post("/api/v2/memory/summary", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert "summary" in data
        assert "sentiment" in data
        assert "recall_cues" in data

    def test_api_v2_events_simulate(self):
        res = self.client.get("/api/v2/events/simulate?person_id=person_001")
        assert res.status_code == 200
        data = res.get_json()
        assert data["type"] == "person_detected"
        assert data["name"] == "Sarah Jenkins"
