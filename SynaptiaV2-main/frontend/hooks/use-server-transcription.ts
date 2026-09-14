"use client"

import { useEffect, useRef, useState, useCallback } from "react"
import { summarizeTranscriptLocally, formatLocalHighlights } from "@/lib/local-summarizer"
import { extractVoiceDescriptor, enrollVoiceprint, getEnrolledVoiceprint } from "@/lib/voice-descriptor"


const INFERENCE_BACKEND_URL = "http://localhost:8002"

type TranscriptionStatus = "idle" | "listening" | "transcribing" | "error"

interface UseServerTranscriptionOptions {
  enabled: boolean
  stream: MediaStream | null
  personName?: string
}

function supportedMimeType() {
  return ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((type) => MediaRecorder.isTypeSupported(type))
}

async function encodeBase64(audio: Blob) {
  const bytes = new Uint8Array(await audio.arrayBuffer())
  let binary = ""
  const blockSize = 0x8000
  for (let offset = 0; offset < bytes.length; offset += blockSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + blockSize))
  }
  return btoa(binary)
}

export function deduplicateSummary(summary: string | null): string | null {
  if (!summary) return null
  const formatted = formatLocalHighlights(summary)
  return formatted || null
}

export function useServerTranscription({ enabled, stream, personName }: UseServerTranscriptionOptions) {
  const [status, setStatus] = useState<TranscriptionStatus>("idle")
  const [memorySummary, setMemorySummary] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isVoiceVerified, setIsVoiceVerified] = useState(false)
  const [lastSpeakerStatus, setLastSpeakerStatus] = useState<string | null>(null)
  const personNameRef = useRef(personName)
  const latestVoiceDescriptorRef = useRef<number[] | null>(null)

  useEffect(() => {
    personNameRef.current = personName
    setMemorySummary(null)
    setIsVoiceVerified(false)
  }, [personName])

  // Explicit voice enrollment function
  const enrollVoice = useCallback(async (nameToEnroll?: string) => {
    const targetName = nameToEnroll || personNameRef.current
    if (!targetName) return { success: false, error: "Person name required for voice enrollment." }
    const descriptor = latestVoiceDescriptorRef.current
    if (!descriptor || descriptor.length === 0) {
      return { success: false, error: "No voice sample detected yet. Speak a short phrase to calibrate." }
    }
    const success = await enrollVoiceprint(targetName, descriptor)
    if (success) {
      setIsVoiceVerified(true)
      setLastSpeakerStatus(`Voiceprint enrolled for ${targetName}`)
    }
    return { success, error: success ? null : "Failed to persist voiceprint." }
  }, [])

  useEffect(() => {
    if (!enabled || !stream) {
      setStatus("idle")
      return
    }
    if (typeof MediaRecorder === "undefined") {
      setStatus("error")
      setError("This browser cannot record microphone audio.")
      return
    }

    const mimeType = supportedMimeType()
    if (!mimeType) {
      setStatus("error")
      setError("This browser does not support a compatible microphone format.")
      return
    }
    const audioStream = new MediaStream(stream.getAudioTracks())
    if (audioStream.getAudioTracks().length === 0) {
      setStatus("error")
      setError("No microphone track is available. Enable microphone access and restart the camera.")
      return
    }

    let disposed = false
    let recorder: MediaRecorder | null = null
    let recordedChunks: BlobPart[] = []
    let isRecordingUtterance = false
    let speechStartTime = 0
    let consecutiveSpeechFrames = 0
    let consecutiveSilenceMs = 0
    let vadInterval: ReturnType<typeof setInterval> | null = null
    let maxUtteranceTimer: ReturnType<typeof setTimeout> | null = null

    // VAD & Voiceprint Audio Engine
    let audioContext: AudioContext | null = null
    let analyser: AnalyserNode | null = null
    let audioSource: MediaStreamAudioSourceNode | null = null
    let highPassFilter: BiquadFilterNode | null = null
    let vadDataArray: Float32Array | null = null
    let freqDataArray: Float32Array | null = null
    let accumulatedFrequencyBins: Float32Array | null = null
    let frequencySampleCount = 0
    let voicedFramesCount = 0

    try {
      const AudioContextClass = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      if (AudioContextClass) {
        audioContext = new AudioContextClass()
        analyser = audioContext.createAnalyser()
        analyser.fftSize = 512
        audioSource = audioContext.createMediaStreamSource(audioStream)

        // 180Hz High-Pass filter removes breath puffs, desk rumble, and low air whoosh
        highPassFilter = audioContext.createBiquadFilter()
        highPassFilter.type = "highpass"
        highPassFilter.frequency.value = 180
        highPassFilter.Q.value = 0.707

        audioSource.connect(highPassFilter)
        highPassFilter.connect(analyser)

        vadDataArray = new Float32Array(analyser.fftSize)
        freqDataArray = new Float32Array(analyser.frequencyBinCount)
        accumulatedFrequencyBins = new Float32Array(analyser.frequencyBinCount)
      }
    } catch (vadErr) {
      console.warn("[TranscriptionVAD] Web Audio Analyser init:", vadErr)
    }

    const processUtteranceAudio = async (audioBlob: Blob, voiceDescriptor: number[]) => {
      if (disposed || !audioBlob || audioBlob.size === 0) return

      setStatus("transcribing")
      try {
        const audioBase64 = await encodeBase64(audioBlob)
        const response = await fetch(`${INFERENCE_BACKEND_URL}/api/transcriptions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            audio_base64: audioBase64,
            filename: `speech.${mimeType.includes("mp4") ? "m4a" : "webm"}`,
            content_type: mimeType,
            person_name: personNameRef.current,
            voice_descriptor: voiceDescriptor,
          }),
        })
        const payload = await response.json().catch(() => ({}))
        if (!response.ok) {
          throw new Error(payload.detail ?? `Transcription returned ${response.status}`)
        }

        // Handle speaker verification rejection or silence/breath drop
        if (payload.reason && (payload.reason.includes("Bystander") || payload.reason.includes("No speech"))) {
          console.log("[VoiceVerifier] Audio rejected:", payload.reason)
          setLastSpeakerStatus(payload.reason.includes("Bystander") ? "Bystander voice filtered" : "Silence ignored")
          setIsVoiceVerified(false)
          return
        }

        if (payload.text && payload.text.trim()) {
          setIsVoiceVerified(true)
          setLastSpeakerStatus(personNameRef.current ? `Voice verified: ${personNameRef.current}` : "Voice active")

          // Prioritize high-quality synthesized backend memory summary
          if (payload.summary && !disposed) {
            setMemorySummary(deduplicateSummary(payload.summary))
          } else {
            // Local Hugging Face ONNX summarizer fallback for offline operation
            const localSummary = await summarizeTranscriptLocally(payload.text, personNameRef.current)
            if (localSummary && !disposed) {
              setMemorySummary(deduplicateSummary(localSummary))
            }
          }
        } else if (payload.summary && !disposed) {
          setIsVoiceVerified(true)
          setMemorySummary(deduplicateSummary(payload.summary))
        }


        setError(null)
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : "Speech transcription failed"
        if (!disposed) {
          setStatus("error")
          setError(`Speech transcription: ${message}`)
        }
      } finally {
        if (!disposed) {
          setStatus("listening")
        }
      }
    }

    const stopUtteranceRecording = () => {
      if (!isRecordingUtterance || !recorder) return
      isRecordingUtterance = false
      if (maxUtteranceTimer) {
        clearTimeout(maxUtteranceTimer)
        maxUtteranceTimer = null
      }
      try {
        if (recorder.state === "recording") {
          recorder.stop()
        }
      } catch (err) {
        console.warn("[TranscriptionVAD] Error stopping recorder:", err)
      }
    }

    const startUtteranceRecording = () => {
      if (isRecordingUtterance || disposed) return
      isRecordingUtterance = true
      speechStartTime = Date.now()
      consecutiveSilenceMs = 0
      voicedFramesCount = 0
      recordedChunks = []
      frequencySampleCount = 0
      if (accumulatedFrequencyBins) {
        accumulatedFrequencyBins.fill(0)
      }

      try {
        recorder = new MediaRecorder(audioStream, { mimeType })
        recorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            recordedChunks.push(event.data)
          }
        }
        recorder.onstop = () => {
          const duration = Date.now() - speechStartTime
          const completedBlob = new Blob(recordedChunks, { type: mimeType })
          recordedChunks = []

          // Compute average frequency spectrum across speech duration
          let voiceDesc: number[] = []
          if (accumulatedFrequencyBins && frequencySampleCount > 0) {
            const avgFreq = new Float32Array(accumulatedFrequencyBins.length)
            for (let i = 0; i < avgFreq.length; i++) {
              avgFreq[i] = accumulatedFrequencyBins[i] / frequencySampleCount
            }
            voiceDesc = extractVoiceDescriptor(avgFreq)
            latestVoiceDescriptorRef.current = voiceDesc
          }

          // Speech check: must be at least 500ms duration and have valid audio data
          if (duration >= 500 && completedBlob.size >= 800) {
            console.log(`[TranscriptionVAD] Utterance completed (${duration}ms, ${completedBlob.size} bytes), sending to transcription...`)
            void processUtteranceAudio(completedBlob, voiceDesc)
          } else {
            console.log(`[TranscriptionVAD] Chunk discarded (${duration}ms, size=${completedBlob.size}).`)
            if (!disposed) setStatus("listening")
          }
        }
        recorder.start(100)
        setStatus("listening")

        maxUtteranceTimer = setTimeout(() => {
          if (isRecordingUtterance) {
            stopUtteranceRecording()
          }
        }, 10_000)
      } catch (err) {
        console.error("[TranscriptionVAD] Failed to start recorder:", err)
        isRecordingUtterance = false
      }
    }

    // Voice Activity Detection Loop: runs every 50ms
    // With 180Hz High-Pass Filter active, silence is < 0.008, speaking is 0.018 - 0.20
    const SPEECH_START_RMS = 0.018
    const SPEECH_CONTINUE_RMS = 0.010
    const SILENCE_TO_STOP_MS = 800

    setStatus("listening")

    vadInterval = setInterval(() => {
      if (disposed || !analyser || !vadDataArray) return

      analyser.getFloatTimeDomainData(vadDataArray)
      let sumSquares = 0
      for (let i = 0; i < vadDataArray.length; i++) {
        sumSquares += vadDataArray[i] * vadDataArray[i]
      }
      const rms = Math.sqrt(sumSquares / vadDataArray.length)

      const isVoiced = rms >= SPEECH_START_RMS

      if (!isRecordingUtterance) {
        // Listening for speech onset: requires 2 consecutive frames (100ms)
        if (isVoiced) {
          consecutiveSpeechFrames++
          if (consecutiveSpeechFrames >= 2) {
            consecutiveSpeechFrames = 0
            startUtteranceRecording()
          }
        } else {
          consecutiveSpeechFrames = 0
        }
      } else {
        // Actively capturing speech utterance & accumulating spectral frequency
        if (freqDataArray && accumulatedFrequencyBins) {
          analyser.getFloatFrequencyData(freqDataArray)
          for (let i = 0; i < freqDataArray.length; i++) {
            accumulatedFrequencyBins[i] += freqDataArray[i]
          }
          frequencySampleCount++
        }

        if (isVoiced) {
          voicedFramesCount++
        }

        if (rms >= SPEECH_CONTINUE_RMS) {
          consecutiveSilenceMs = 0
        } else {
          consecutiveSilenceMs += 50
          if (consecutiveSilenceMs >= SILENCE_TO_STOP_MS) {
            stopUtteranceRecording()
          }
        }
      }
    }, 50)



    setError(null)

    return () => {
      disposed = true
      if (vadInterval) clearInterval(vadInterval)
      if (maxUtteranceTimer) clearTimeout(maxUtteranceTimer)
      if (recorder && recorder.state === "recording") {
        try {
          recorder.stop()
        } catch {}
      }
      if (audioContext && audioContext.state !== "closed") {
        void audioContext.close()
      }
    }
  }, [enabled, stream])

  return {
    isSupported: typeof MediaRecorder !== "undefined",
    isListening: status === "listening" || status === "transcribing",
    status,
    memorySummary,
    error,
    isVoiceVerified,
    lastSpeakerStatus,
    enrollVoice,
  }
}


