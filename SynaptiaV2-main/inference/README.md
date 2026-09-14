# Real-Time Inference Service

The local persistence and inference API for the AR-glasses client. It stores
confirmed face observations, explicit face enrollments, and local speech
transcripts in MongoDB.

## Architecture

```
Browser face detection ─── POST face observation ──┐
Browser local Whisper ASR ─── POST transcript ─────┼→ Inference Service → MongoDB
Browser face enrollment ─── PUT descriptor ────────┘     (Port 8002)
```

## Components

1. **Inference Service** (`main.py`)
   - Persists confirmed browser data in MongoDB.
   - Does not invent identities or conversation events.
   - Can consume an explicitly configured real SSE metadata source.

2. **Optional simulator files** (`mock_metadata_service.py`, `mock_consumer.py`)
   - Retained only for isolated development experiments.
   - Never start automatically and are not part of the main app workflow.

## Setup

### Option 1: Automated Setup (Recommended)
```bash
cd inference
./setup.sh
```

This will:
- Create a virtual environment in `venv/`
- Install all dependencies
- Provide instructions for running

### Option 2: Manual Setup
```bash
cd inference

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Service

**Important**: Make sure to activate the virtual environment first:
```bash
source venv/bin/activate
```

Start the service:
```bash
source venv/bin/activate
python main.py
```
This starts on http://localhost:8002

To consume a real metadata producer, configure it explicitly before starting
the service. It is disabled by default:

```bash
METADATA_SERVICE_URL=http://localhost:8000/stream/conversation python main.py
```

## API Endpoints

### Inference Service (Port 8002)
- `GET /api/face-identities` - fetch explicitly enrolled face descriptors
- `PUT /api/face-identities` - create or update an explicit face enrollment
- `POST /api/faces/observations` - stores one stable, downscaled face crop and detection metadata in MongoDB
- `POST /api/transcripts` - stores local speech-to-text output in MongoDB
- `GET /health` - Health check with queue status
- `GET /` - Service information

## Testing with cURL

### Confirm service health:
```bash
curl http://localhost:8002/health
```

### List enrolled faces:
```bash
curl http://localhost:8002/api/face-identities
```

## Notes

- Face identity is opt-in: a person must enter their name and choose **Enroll**.
- The frontend runs Whisper speech recognition locally in the browser. Its model
  is downloaded once, cached by the browser, and does not use the browser's
  cloud speech-recognition service.
- Face observations and final transcripts are stored only after successful API
  requests; failures are visible in the UI and are never replaced with demo data.
