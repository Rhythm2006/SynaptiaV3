# 📘 Synaptia: Complete Codebase Pipeline & System Architecture Guide

> **Comprehensive Technical Deep-Dive into Synaptia's Dual-Architecture: The Cognitive Training Engine & The Real-Time Assistive AR Memory Recall System.**

---

## 📑 Table of Contents
1. [High-Level Architecture Overview](#1-high-level-architecture-overview)
2. [Dual-System Architecture Topology](#2-dual-system-architecture-topology)
3. [Deep-Dive: AR Memory & Vision Pipeline (SynaptiaV2)](#3-deep-dive-ar-memory--vision-pipeline-synaptiav2)
   - [Pipeline 3.1: Face Recognition & Continuous Identity Tracking](#pipeline-31-face-recognition--continuous-identity-tracking)
   - [Pipeline 3.2: 180Hz High-Pass VAD & Audio Filtration](#pipeline-32-180hz-high-pass-vad--audio-filtration)
   - [Pipeline 3.3: 128D Speaker Voiceprint Verification & Bystander Rejection](#pipeline-33-128d-speaker-voiceprint-verification--bystander-rejection)
   - [Pipeline 3.4: End-of-Sentence Cadence & Whisper-Turbo Transcription](#pipeline-34-end-of-sentence-cadence--whisper-turbo-transcription)
   - [Pipeline 3.5: Capped Memory Highlight Distillation ($\le 3$ Bullets, $\le 25$ Words)](#pipeline-35-capped-memory-highlight-distillation-le-3-bullets-le-25-words)
   - [Pipeline 3.6: Dual-Engine Summarizer (In-Browser ONNX vs Cloud Groq)](#pipeline-36-dual-engine-summarizer-in-browser-onnx-vs-cloud-groq)
   - [Pipeline 3.7: Dual-Storage Persistence (MongoDB Atlas + Local Resilient JSON)](#pipeline-37-dual-storage-persistence-mongodb-atlas--local-resilient-json)
4. [Deep-Dive: Cognitive Training Platform (Synaptia Core)](#4-deep-dive-cognitive-training-platform-synaptia-core)
   - [Pipeline 4.1: Dynamic Problem Generation & Hallucination Defense](#pipeline-41-dynamic-problem-generation--hallucination-defense)
   - [Pipeline 4.2: Adaptive Difficulty Scaling Engine](#pipeline-42-adaptive-difficulty-scaling-engine)
   - [Pipeline 4.3: Multi-Tier Neurological Hint Hierarchy](#pipeline-43-multi-tier-neurological-hint-hierarchy)
   - [Pipeline 4.4: Firebase Auth & User-Scoped Firestore Synchronization](#pipeline-44-firebase-auth--user-scoped-firestore-synchronization)
5. [Complete File-by-File Codebase Map](#5-complete-file-by-file-codebase-map)
6. [Data Schemas & State Contracts](#6-data-schemas--state-contracts)
7. [Deployment Architecture (Vercel + Serverless WSGI)](#7-deployment-architecture-vercel--serverless-wsgi)

---

## 1. High-Level Architecture Overview

Synaptia is built as an end-to-end cognitive assistance platform comprising two cooperating subsystems:
1. **Synaptia AR Smart Glasses Engine (`SynaptiaV2-main`)**: A sub-second, multi-modal perceptual pipeline running in Next.js + FastAPI. It detects faces in the wearer's visual field, verifies the speaker's voice using 128D acoustic descriptors, captures spoken speech upon sentence completion, and displays non-intrusive, 3rd-person highlight memory cues on an augmented reality Heads-Up Display (HUD).
2. **Synaptia Cognitive Training System (`/`)**: A Flask + Firebase platform that generates non-repetitive reasoning, logic, and memory exercises dynamically using LPU-accelerated Groq models with cross-validation, adaptive difficulty adjustment, and real-time cloud profile sync.

---

## 2. Dual-System Architecture Topology

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           USER INTERFACE                                               │
├────────────────────────────────────────────────────┬───────────────────────────────────────────────────┤
│         AR Smart Glasses HUD (Next.js 14)          │           Cognitive Training Web (Flask)          │
│   • WebRTC Camera Stream & Glassmorphism           │   • Responsive Glass UI + Dark/Light Theme        │
│   • Dynamic Reticle Corner Tracking                │   • Live Interactive Exercise Board               │
│   • In-Browser Hugging Face ONNX Neural Engine     │   • Firebase Google / Email Authentication        │
│   • Web Audio API 180Hz High-Pass VAD Filter       │   • Multi-Tier Progressive Hint Drawer            │
└─────────────────────────┬──────────────────────────┴─────────────────────────┬─────────────────────────┘
                          │                                                    │
                          ▼                                                    ▼
┌────────────────────────────────────────────────────┐ ┌─────────────────────────────────────────────────┐
│     FASTAPI PERCEPTION BACKEND (:8002)             │ │       FLASK CORE ENGINE (:5002 / Vercel)        │
│   • Face Vector Matcher (SSD MobileNet V1)         │ │   • PuzzleGenerator (Cross-Validation Layer)    │
│   • Voiceprint Cosine Similarity Verifier          │ │   • DifficultyEngine (Dynamic PID Scaling)      │
│   • Whisper-Turbo Speech-to-Text Pipeline          │ │   • HintProvider (3-Tier Neurological Clues)    │
│   • Highlight Point Distillation (Max 3 Bullets)   │ │   • AnalyticsEngine & ProgressTracker           │
│   • Reset & Fresh-Slate Controller                 │ │   • VercelWSGIWrapper Normalizer                │
└─────────────────────────┬──────────────────────────┘ └───────────────────────┬─────────────────────────┘
                          │                                                    │
                          ▼                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  CLOUD AI & PERSISTENCE LAYER                                          │
│   • Groq Cloud LPU (`groq/compound-mini`, `whisper-large-v3-turbo`)                                    │
│   • Google Cloud Firestore (User Progress, History, Sessions) & Firebase Auth                         │
│   • MongoDB Atlas Cloud (`face_identities`, `person_memories`, `voice_identities`)                     │
│   • Local Resilient JSON Cache (`face_identities.json`, `person_memories.json`)                        │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep-Dive: AR Memory & Vision Pipeline (SynaptiaV2)

### Pipeline 3.1: Face Recognition & Continuous Identity Tracking
1. **Capture**: The browser initializes the video stream via `navigator.mediaDevices.getUserMedia` in [`webcam-stream.tsx`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/frontend/components/webcam-stream.tsx).
2. **Landmark Extraction**: In [`use-face-recognition.ts`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/frontend/hooks/use-face-recognition.ts), Face-API.js processes each frame using an SSD MobileNet V1 backbone to extract 68 facial landmarks and compute a **128-dimensional unit vector** (face descriptor).
3. **Bounding Box Mapping**: [`coordinate-mapper.ts`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/frontend/lib/coordinate-mapper.ts) applies scale and aspect ratio transformations to align video coordinates with screen overlay coordinates with smooth sub-pixel interpolation.
4. **Recognition**: Descriptors are sent to `POST /api/faces/observations` on the FastAPI backend ([`inference/main.py`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/inference/main.py)). The backend evaluates Euclidean distances against stored identities. If $d \le 0.60$, the identity is confirmed with confidence score $1.0 - d$.
5. **HUD Rendering**: [`face-notification.tsx`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/frontend/components/face-notification.tsx) renders animated corner reticles, confidence percentage counters, and glowing status pills around the tracked face.

---

### Pipeline 3.2: 180Hz High-Pass VAD & Audio Filtration
To prevent Whisper hallucinations caused by mic breath, ambient HVAC rumble, and background noise:
1. **Web Audio Graph**: In [`use-server-transcription.ts`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/frontend/hooks/use-server-transcription.ts), an `AudioContext` constructs a **180Hz High-Pass Biquad Filter** (`filter.type = "highpass"`, `filter.frequency.value = 180`).
2. **Frequency Attenuation**: Frequencies below 180Hz (breathing, wind, mic thumps) are filtered out, leaving human vocal formant frequencies (300Hz–3400Hz) intact.
3. **Dual-RMS Thresholding**:
   - **Speech Onset Trigger**: Requires $RMS \ge 0.018$ for 2 consecutive 50ms frames (100ms duration) to initiate recording.
   - **Speech Continuation**: Requires $RMS \ge 0.010$ to keep recording active.

---

### Pipeline 3.3: 128D Speaker Voiceprint Verification & Bystander Rejection
1. **Spectral Feature Extraction**: While recording an utterance, an `AnalyserNode` computes the average Fast Fourier Transform (FFT) spectrum across all active speech frames.
2. **128D Normalization**: In [`voice-descriptor.ts`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/frontend/lib/voice-descriptor.ts), `extractVoiceDescriptor()` bins the spectrum into a normalized 128-dimensional vector ($L_2\text{-norm} = 1.0$).
3. **Verification Matcher**: In [`voice_verifier.py`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/inference/voice_verifier.py), `verify_speaker_voiceprint()` computes the cosine similarity:
   $$\text{Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
   - **Match Threshold ($T = 0.66$)**: If similarity $\ge 0.66$, the voice is verified as the enrolled person.
   - **Bystander Rejection**: If similarity $< 0.66$, the audio is discarded with reason `Bystander voice filtered out (unverified speaker)` and an amber warning pill appears on the HUD.

---

### Pipeline 3.4: End-of-Sentence Cadence & Whisper-Turbo Transcription
1. **Silence Cadence Detector**: If speech falls below $RMS = 0.010$ for **800ms**, the system recognizes that the speaker has concluded their sentence and immediately stops the `MediaRecorder`.
2. **Payload Dispatch**: Audio chunks are converted into Base64 and dispatched to `POST /api/transcriptions` on the FastAPI server.
3. **Whisper Turbo Transcription**: Uses Groq's `whisper-large-v3-turbo` with `temperature = 0`.
4. **Hallucination Filtration**:
   - Checks verbose JSON segments from Whisper:
     - `no_speech_prob > 0.40` $\rightarrow$ Discarded.
     - `avg_logprob < -0.95` $\rightarrow$ Discarded.
     - `compression_ratio > 2.4` $\rightarrow$ Discarded.
   - Discards single-word artifacts and hallucination phrases (e.g. *"Thank you for watching"*, *"Subtitles by..."*).

---

### Pipeline 3.5: Capped Memory Highlight Distillation ($\le 3$ Bullets, $\le 25$ Words)
To prevent overwhelming dementia patients with walls of text:
1. **Distillation Prompt**: In [`fireworks_client.py`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/inference/fireworks_client.py), `summarize_person_memory()` instructs the LLM to synthesize the previous memory and new real transcript into **at most 2–3 concise highlight points**.
2. **Dynamic Pruning**: When new conversational facts arrive, older transient chit-chat (greetings, weather) is discarded to preserve a compact, stable memory footprint.
3. **Hard Cap Enforcer (`format_highlight_summary`)**:
   - Splits incoming text by bullet markers (`•`, `*`, `-`) or sentence boundaries.
   - Removes duplicate points using normalized substring similarity.
   - Enforces a strict limit of $\le 3$ highlight items and $\le 25$ words total.
4. **Glassmorphic Bullet Rendering**: In [`face-notification.tsx`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/frontend/components/face-notification.tsx), `parseHighlights()` breaks down the summary and renders individual glassmorphic bullet items accompanied by glowing emerald accent dots.

---

### Pipeline 3.6: Dual-Engine Summarizer (In-Browser ONNX vs Cloud Groq)
Synaptia employs a hybrid architecture for high availability:
* **Primary (Cloud Groq)**: Sub-second high-precision LLM reasoning (`groq/compound-mini`).
* **Fallback / Offline (In-Browser ONNX)**: In [`local-summarizer.ts`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/frontend/lib/local-summarizer.ts), Hugging Face Transformers.js loads an ONNX-quantized `Xenova/distilbart-cnn-6-6` model running via WebAssembly/WebGPU directly on the client's device with zero server dependency.

---

### Pipeline 3.7: Dual-Storage Persistence (MongoDB Atlas + Local Resilient JSON)
In [`database.py`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/inference/database.py):
* **Cloud Storage**: Automatically pushes enrolled face identities, voiceprints, and distilled memories to MongoDB Atlas (`dementia_care_db`).
* **Local Resilient JSON Fallback**: Concurrently mirrors all records to [`SynaptiaV2-main/inference/data/`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/SynaptiaV2-main/inference/data/) (`face_identities.json`, `person_memories.json`, `people.json`), ensuring offline durability and zero network latency.

---

## 4. Deep-Dive: Cognitive Training Platform (Synaptia Core)

### Pipeline 4.1: Dynamic Problem Generation & Hallucination Defense
In [`puzzle_generator.py`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/puzzle_generator.py):
1. **Dynamic Prompting**: Solicits domain-specific reasoning challenges (Riddles, Math, Logic, Wordplay, Trivia) across 3 pacing levels.
2. **Cross-Validation Layer**: Prompts the LLM with strict formatting rules requiring verifiable single-answer solutions.
3. **Template Fallback Matrix**: If API latency exceeds threshold or rate limits occur, the engine falls back to a curated pool of deterministic cognitive puzzles.

---

### Pipeline 4.2: Adaptive Difficulty Scaling Engine
In [`difficulty_engine.py`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/difficulty_engine.py):
* Tracks user performance metrics in real time:
  - Time-to-solve ($\Delta t$)
  - Mistake / Attempt frequency ($E$)
  - Hint reliance count ($H$)
* Computes performance score $P$:
  $$P = \frac{\text{BaseScore}}{\Delta t} \times (1 - 0.2 \cdot H) \times (1 - 0.15 \cdot E)$$
* Dynamically increases or decreases puzzle difficulty (Easy $\leftrightarrow$ Medium $\leftrightarrow$ Hard) without explicit user configuration.

---

### Pipeline 4.3: Multi-Tier Neurological Hint Hierarchy
In [`hint_provider.py`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/hint_provider.py):
Provides progressive, non-intrusive cognitive scaffolding:
- **Tier 1 — Gentle Nudge**: Re-frames the question to activate conceptual associations.
- **Tier 2 — Structural Clue**: Highlights key patterns or eliminates misleading interpretations.
- **Tier 3 — Methodical Breakdown**: Guides the user step-by-step toward the solution without revealing the final answer directly.

---

### Pipeline 4.4: Firebase Auth & User-Scoped Firestore Synchronization
In [`static/firebase-config.js`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/static/firebase-config.js) and [`static/script.js`](file:///Users/rhythmsanjeev/Downloads/Synaptia-main/static/script.js):
1. **Authentication**: Google Sign-In and Email/Password authentication via Firebase Auth SDK v10.
2. **User Profile Sync**: On authentication, initializes or merges user progress at document `users/{uid}`.
3. **Firestore Security**: Enforces `request.auth.uid == userId` rules to ensure total data isolation.
4. **Session Persistence**: Syncs scores, streaks, hints used, and solve timestamps after every completed puzzle.

---

## 5. Complete File-by-File Codebase Map

```
Synaptia-main/
├── app.py                      # Flask Application Server, REST APIs, and Static file routing
├── puzzle_generator.py         # Dynamic AI puzzle generator & cross-validation engine
├── hint_provider.py            # 3-Tier cognitive hint provider
├── difficulty_engine.py        # Dynamic difficulty adaptation logic
├── progress_tracker.py         # Cognitive metrics, streaks, and session analytics
├── analytics.py                # Session logging and event aggregation
├── interview_coach.py          # Interactive reasoning & cognitive coaching dialogs
├── vision_memory.py            # Local vision memory integration (v1 engine)
├── config.py                   # Environment configuration dictionary
├── requirements.txt            # Python dependencies for Flask core & tests
├── vercel.json                 # Vercel serverless rewrite & Edge CDN configuration
│
├── api/
│   └── index.py                # Vercel WSGI entrypoint with HTTP_X_FORWARDED_PATH normalizer
│
├── public/                     # Vercel Edge CDN public directory
│   └── static/                 # CSS stylesheets, client scripts, and images
│
├── static/
│   ├── style.css               # Design system styling (Glassmorphism, animations, responsive layout)
│   ├── script.js               # Frontend controller, state machine, and DOM interactions
│   └── firebase-config.js      # Firebase SDK initialization, Auth, and Firestore sync helpers
│
├── templates/
│   ├── index.html              # Main cognitive training application interface
│   └── login.html              # Authentication & user onboarding portal
│
├── tests/
│   └── test_suite.py           # Automated pytest suite (70 unit & integration tests)
│
└── SynaptiaV2-main/            # AR Smart Glasses Perception Subsystem
    ├── frontend/               # Next.js 14 Web Application
    │   ├── components/
    │   │   ├── webcam-stream.tsx      # WebRTC camera feed, Reticle positioning, and UI controls
    │   │   ├── face-notification.tsx  # Glassmorphic AR HUD card with highlight bullet points
    │   │   ├── rayban-overlay.tsx     # Smart glasses optical overlay simulator
    │   │   └── glass-container.tsx    # Glassmorphism visual container primitives
    │   ├── hooks/
    │   │   ├── use-face-recognition.ts     # Face-API landmark extractor & identity tracker
    │   │   └── use-server-transcription.ts # 180Hz High-Pass VAD & end-of-sentence cadence
    │   └── lib/
    │       ├── local-summarizer.ts    # In-browser Hugging Face Transformers.js ONNX summarizer
    │       ├── voice-descriptor.ts    # 128D acoustic spectral embedding extractor
    │       └── coordinate-mapper.ts   # Bounding box screen transformation & interpolation
    │
    └── inference/              # FastAPI High-Performance Perception Backend
        ├── main.py             # REST endpoints (/api/faces, /api/transcriptions, /api/reset-all)
        ├── voice_verifier.py   # Cosine similarity voiceprint comparator
        ├── fireworks_client.py # Groq LLM integration & format_highlight_summary distillation
        ├── database.py         # MongoDB Atlas driver with resilient local JSON fallback
        └── models.py           # Pydantic schemas for observations, identities, and memories
```

---

## 6. Data Schemas & State Contracts

### 6.1 Face Identity Contract (`face_identities.json` / MongoDB `face_identities`)
```json
{
  "rhythm": {
    "name": "Rhythm",
    "relationship": "Your Friend",
    "descriptor": [0.042, -0.015, 0.112, "... 128 float coordinates ..."],
    "voice_descriptor": [0.009, 0.045, 0.081, "... 128 spectral bins ..."],
    "created_at": "2026-09-15T02:00:00Z",
    "confidence": 0.94
  }
}
```

### 6.2 Distilled Memory Summary Contract (`person_memories.json` / MongoDB `person_memories`)
```json
{
  "rhythm": {
    "person_name": "Rhythm",
    "summary": "• Rhythm (software engineer) met at tech event\n• Mentioned working on AI project\n• Meeting tomorrow at 3 PM",
    "updated_at": "2026-09-15T02:05:00Z"
  }
}
```

### 6.3 Firebase Firestore User Profile Contract (`users/{uid}`)
```json
{
  "uid": "google_oauth_1029384756",
  "displayName": "Rhythm",
  "email": "rhythm@example.com",
  "photoURL": "https://lh3.googleusercontent.com/...",
  "score": 450,
  "solved": 14,
  "streak": 5,
  "totalHints": 3,
  "achievements": ["first_solve", "streak_3", "speed_solver"],
  "lastLogin": "2026-09-15T02:00:00Z"
}
```

---

## 7. Deployment Architecture (Vercel + Serverless WSGI)

```
                            Incoming Request (e.g. synaptia.vercel.app/login)
                                                  │
                                                  ▼
                                    ┌───────────────────────────┐
                                    │     Vercel Edge Router    │
                                    └─────────────┬─────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         │                                                 │
                  Static Request                                    Dynamic Request
             (/static/* or public/*)                              (/, /login, /api/*)
                         │                                                 │
                         ▼                                                 ▼
             ┌───────────────────────────┐                    ┌─────────────────────────┐
             │   Vercel Global Edge CDN  │                    │ Vercel Serverless Lambda│
             │   (0ms Serverless Delay)  │                    │ (Python Runtime 3.11)   │
             └───────────────────────────┘                    └────────────┬────────────┘
                                                                           │
                                                                           ▼
                                                              ┌─────────────────────────┐
                                                              │  `api/index.py`         │
                                                              │  (VercelWSGIWrapper)    │
                                                              │  Normalizes PATH_INFO   │
                                                              └────────────┬────────────┘
                                                                           │
                                                                           ▼
                                                              ┌─────────────────────────┐
                                                              │   Flask Application     │
                                                              │   (`app.py`)            │
                                                              └─────────────────────────┘
```

1. **Static Files**: Requests matching `/static/(.*)` are resolved immediately by Vercel's Edge CDN from `/public/static/` with accurate MIME headers.
2. **Serverless WSGI Wrapper**: Dynamic routes (`/`, `/login`, `/api/puzzle/generate`) are routed to `api/index.py`. The `VercelWSGIWrapper` reads `HTTP_X_FORWARDED_PATH` from incoming proxy headers and routes cleanly to Flask without rewrite mismatches.

---

*Authored for the Synaptia Engineering Team — MIT License.*
