# Synaptia 🧠✨
### *Augmented Cognitive Training & Real-Time Assistive AR Memory System*

[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black.svg?logo=next.js&logoColor=white)](https://nextjs.org)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers.js-yellow.svg)](https://huggingface.co/docs/transformers.js)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-orange.svg)](https://groq.com)
[![Firebase](https://img.shields.io/badge/Firebase-Auth%20%26%20Firestore-FFA611.svg?logo=firebase&logoColor=white)](https://firebase.google.com)

---

## 🌟 Overview

**Synaptia** is an intelligent dual-system platform designed for **neuro-assistive care, memory recall for dementia/Alzheimer's patients, and adaptive cognitive training**.

1. **AR Memory Recall Companion (SynaptiaV2)**: Designed for smart glasses and live cameras. Delivers real-time face identification, speaker-verified voice activity transcription, and distilled highlight-point memory cues directly onto a futuristic, non-intrusive heads-up display (HUD).
2. **Cognitive Training & Evaluation Engine**: An interactive, dynamic problem-solving platform featuring adaptive difficulty scaling, multi-tier neurological hints, and cloud user-progress synchronization via Firebase.

---

## 📚 In-Depth Project Guides & Analysis

* 📖 **[Clinical Use Cases, Feasibility & Scalability Guide](USE_CASES_AND_FEASIBILITY.md)**: Comprehensive evaluation of target patient personas, global dementia/stroke health data, real-world deployment environments, technical feasibility, economic ROI, and scalability benchmarks.
* 🛠️ **[Architecture & Pipeline Technical Manual](PIPELINE_EXPLANATION.md)**: Deep technical dive into the dual-engine pipeline, sub-second latency breakdown, data schemas, WSGI serverless routing, and code maps.

---

## 🚀 Key Feature: AR Memory Recall & Capped Highlight Distillation

Smart glasses wearers and dementia patients need instant, scannable memory reminders without being overwhelmed by endless paragraphs of conversation history. Synaptia introduces an **End-of-Sentence Voice-Verified Memory Distillation Pipeline**:

```
 ┌────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
 │  Webcam Feed   │ ──> │ Face Detection &     │ ──> │ Visual Identity        │
 │  (AR Glasses)  │     │ Recognition Pipeline │     │ HUD Notification Card  │
 └────────────────┘     └──────────────────────┘     └───────────┬────────────┘
                                                                 │
 ┌────────────────┐     ┌──────────────────────┐                 │ Real-Time
 │  Microphone    │ ──> │ 180Hz High-Pass VAD  │                 │ Highlight
 │  Audio Stream  │     │ + 128D Voice Verifier│                 │ Updates
 └────────────────┘     └──────────┬───────────┘                 │
                                   ▼                             │
                        ┌──────────────────────┐                 │
                        │ End-of-Sentence      │                 │
                        │ Utterance Trigger    │                 │
                        └──────────┬───────────┘                 │
                                   ▼                             │
                        ┌──────────────────────┐                 │
                        │ Dual Summarizer:     │                 │
                        │ • In-Browser HF ONNX │ ────────────────┘
                        │ • Groq LLM (Cloud)   │
                        │ Max 2-3 Highlights   │
                        └──────────────────────┘
```

### 1. Distilled Highlight Capping (Anti-Overwhelm Memory)
* **Max 2–3 Key Bullet Points ($\le 25$ Words)**: Rather than accumulating dialogue into an unreadable wall of text, conversations "rest" into concise, memorable highlights (`• Rhythm (Engineer): working on AI • Meeting tomorrow at 3 PM`).
* **Dynamic Pruning**: When new vital facts are mentioned, older transient chit-chat or filler is automatically consolidated and pruned.
* **Glassmorphic AR Bullet UI**: Rendered with subtle glowing emerald indicator dots, optimized for readability on optical see-through displays (Ray-Ban Meta, Apple Vision Pro, AR glasses).

### 2. Speaker Voiceprint Verification & Bystander Filtering
* **128-Dimensional Acoustic Descriptor**: Extracts frequency spectral signatures to verify if the registered person is speaking.
* **Bystander Rejection**: Voices from bystanders, televisions, or background noise that do not match the speaker's acoustic profile are discarded with an on-screen indicator (`Bystander voice filtered`).
* **180Hz High-Pass VAD Filter**: Eliminates heavy breathing, mic pops, and ambient hums to stop hallucinated speech transcriptions.

### 3. Sub-Second End-of-Sentence Cadence Trigger
* **Instant Pause Detection**: Detects natural conversational pauses (800ms silence) after spoken sentences.
* **Ultra-Low Latency**: Triggers Whisper-Turbo transcription + in-browser neural summarization within ~1.2s of the speaker finishing a sentence.

### 4. Dual-Tier Hybrid Neural Inference
* **In-Browser Hugging Face Transformers (`@huggingface/transformers`)**: Runs a quantized `Xenova/distilbart-cnn-6-6` ONNX model directly inside the client's browser (WebAssembly/WebGPU). Zero server cost, 100% private, and offline-capable.
* **Cloud Groq LPU Engine**: Ultra-fast fallback for deep context synthesis via Groq's low-latency LPU endpoints (`groq/compound-mini` / `whisper-large-v3-turbo`).

---

## 🧩 Synaptia Cognitive Training System

* **Generative Problem Construction**: Dynamically builds unique reasoning, pattern recognition, and logic puzzles at runtime.
* **Adaptive Difficulty Engine**: Continuously tracks time-to-solve, mistakes, and hint reliance to dynamically scale challenge levels (Easy $\rightarrow$ Hard).
* **Multi-Tier Hint Provider**: Progressive guidance (Gentle Nudge $\rightarrow$ Conceptual Clue $\rightarrow$ Step-by-Step Breakdown).
* **Firebase User Authentication & Progress Sync**:
  * Seamless Google Sign-In & Email Authentication.
  * Real-time Firestore synchronization of user scores, achievements, streaks, and solve history.
  * Strict Firestore security rules protecting user data privacy.

---

## 🛠️ Architecture & Tech Stack

| Layer | Technologies |
|---|---|
| **Frontend UI** | Next.js 14, React, TailwindCSS, Glassmorphism, Web Audio API, Vanilla HTML5/CSS3 |
| **In-Browser Neural AI** | `@huggingface/transformers` (ONNX / WebAssembly), TensorFlow.js Face-API |
| **Backend Services** | FastAPI / Python 3.11+, Flask, Uvicorn |
| **Cloud AI & Speech** | Groq API (`whisper-large-v3-turbo`, `groq/compound-mini`), Fireworks.ai |
| **Database & Auth** | Firebase Authentication, Google Cloud Firestore, MongoDB Atlas |
| **Deployment** | Vercel (Frontend & Serverless CDN), Docker-ready for backend inference |

---

## 📦 Getting Started

### Prerequisites
* **Node.js** v18+ & **npm**
* **Python** 3.11+
* (Optional) **Groq API Key** & **Firebase Project**

### 1. Clone & Setup Environment
```bash
git clone https://github.com/your-username/Synaptia.git
cd Synaptia

# Configure environment variables
cp .env.example .env
```

Add your keys into `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

### 2. Run the AR Glasses Inference & Next.js App (SynaptiaV2)

```bash
# Terminal 1: Run Python Inference Backend
cd SynaptiaV2-main/inference
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8002 --reload

# Terminal 2: Run Next.js Frontend
cd ../frontend
npm install
npm run dev
```
Open **`http://localhost:3000`** in Chrome/Edge.

### 3. Run the Cognitive Training Flask Platform

```bash
# Terminal 3: Run Flask Cognitive App
python3 app.py
```
Open **`http://localhost:5002`** to train cognitive abilities and track user progress.

---

## 🧪 Testing & Validation

Synaptia includes automated unit and integration test suites covering difficulty engines, hint providers, analytics, and memory aggregators:

```bash
python3 -m pytest tests/ -v
```

---

## 🔒 Security & Privacy

* **Edge Processing**: Face landmark detection, acoustic voiceprints, and primary local summarization occur on the client's local device.
* **Firestore Security Rules**: User progress and personal profiles are strictly isolated with user-scoped authentication rules (`request.auth.uid == userId`).

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
