"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { Mic, MicOff, Video, VideoOff } from "lucide-react"
import { FaceNotification } from "@/components/face-notification"
import { useFaceDetection } from "@/hooks/use-face-detection"
import { useFaceRecognition } from "@/hooks/use-face-recognition"
import { useServerTranscription, deduplicateSummary } from "@/hooks/use-server-transcription"
import type { DetectedFace } from "@/lib/face-tracker"
import { RayBanOverlay } from "@/components/rayban-overlay"
import { cn } from "@/lib/utils"
import {
  calculateVideoTransform,
  mapBoundingBoxToOverlay,
  calculateNotificationPosition,
} from "@/lib/coordinate-mapper"

// Track when faces were first seen (for unidentified timeout)
type FaceTimestampMap = Map<string, number>

const UNIDENTIFIED_TIMEOUT_MS = 4000 // Show "Unidentified" after 4 seconds with no SSE data
const INFERENCE_BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE ?? (typeof window !== "undefined" && window.location.port !== "3000" ? "" : "http://localhost:8002")

function captureFaceCrop(video: HTMLVideoElement, face: DetectedFace): string | null {
  const { originX, originY, width, height } = face.boundingBox
  const sourceX = Math.max(0, Math.floor(originX))
  const sourceY = Math.max(0, Math.floor(originY))
  const sourceWidth = Math.min(Math.ceil(width), video.videoWidth - sourceX)
  const sourceHeight = Math.min(Math.ceil(height), video.videoHeight - sourceY)

  if (sourceWidth <= 0 || sourceHeight <= 0) {
    return null
  }

  const canvas = document.createElement("canvas")
  const targetWidth = Math.min(192, sourceWidth)
  canvas.width = targetWidth
  canvas.height = Math.max(1, Math.round((sourceHeight / sourceWidth) * targetWidth))
  const context = canvas.getContext("2d")
  if (!context) {
    return null
  }

  context.drawImage(
    video,
    sourceX,
    sourceY,
    sourceWidth,
    sourceHeight,
    0,
    0,
    canvas.width,
    canvas.height
  )
  return canvas.toDataURL("image/jpeg", 0.8)
}

export default function WebcamStream() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const [isStreaming, setIsStreaming] = useState(false)
  const [isVideoReady, setIsVideoReady] = useState(false)
  const [unidentifiedFaceIds, setUnidentifiedFaceIds] = useState<Set<string>>(new Set())
  const [isRayBanMode, setIsRayBanMode] = useState(false)
  const [isMuted, setIsMuted] = useState(false)
  const [faceStorageError, setFaceStorageError] = useState<string | null>(null)
  const [enrollmentName, setEnrollmentName] = useState("")
  const [enrollmentMessage, setEnrollmentMessage] = useState<string | null>(null)
  const [memorySummary, setMemorySummary] = useState<string | null>(null)
  const faceFirstSeenRef = useRef<FaceTimestampMap>(new Map())
  const faceSessionIdRef = useRef("")
  const storedFaceIdsRef = useRef<Set<string>>(new Set())
  const storingFaceIdsRef = useRef<Set<string>>(new Set())
  const faceStorageRetryAtRef = useRef<Map<string, number>>(new Map())

  const { detectedFaces, isLoading: isFaceDetectionLoading, error: faceDetectionError } = useFaceDetection(
    videoRef.current,
    {
      enabled: isStreaming && isVideoReady && !isRayBanMode,
      minDetectionConfidence: 0.35,
      targetFps: 12,
      useWorker: false,
    }
  )
  const {
    isLoading: isFaceRecognitionLoading,
    error: faceRecognitionError,
    match: faceMatch,
    enroll,
  } = useFaceRecognition(videoRef.current, isStreaming && isVideoReady && !isRayBanMode)
  const {
    isListening,
    status: speechStatus,
    memorySummary: newMemorySummary,
    error: speechError,
    isVoiceVerified,
    lastSpeakerStatus,
    enrollVoice,
  } = useServerTranscription({
    enabled: isStreaming && !isMuted,
    stream: streamRef.current,
    personName: faceMatch?.name ?? "Wearer",
  })

  const enrollCurrentFace = async () => {
    const trimmed = enrollmentName.trim()
    const result = await enroll(trimmed)
    if (result.success) {
      await enrollVoice(trimmed)
      setEnrollmentMessage(`Enrolled ${trimmed} (Face + Voiceprint saved).`)
    } else {
      setEnrollmentMessage(result.error ?? "Enrollment failed.")
    }
  }


  const clearAllProfiles = async () => {
    try {
      if (typeof window !== "undefined") {
        window.localStorage.clear()
      }
      await fetch(`${INFERENCE_BACKEND_URL}/api/reset-all`, { method: "POST" })
      setEnrollmentMessage("All face data, voiceprints, and memories wiped clean.")
      setMemorySummary(null)
      setTimeout(() => {
        if (typeof window !== "undefined") window.location.reload()
      }, 500)
    } catch (err) {
      setEnrollmentMessage("Failed to reset all data.")
    }
  }

  useEffect(() => {
    const name = faceMatch?.name
    if (!name) {
      setMemorySummary(null)
      return
    }

    let cancelled = false
    void fetch(`${INFERENCE_BACKEND_URL}/api/person-memories/${encodeURIComponent(name)}`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`Memory lookup returned ${response.status}`)
        return response.json() as Promise<{ memory?: { summary?: string } | null }>
      })
      .then((body) => {
        if (!cancelled) setMemorySummary(deduplicateSummary(body.memory?.summary ?? null))
      })
      .catch((cause) => {
        console.error("[PersonMemory] Failed to load saved memory:", cause)
        if (!cancelled) setMemorySummary(null)
      })
      .catch((cause) => {
        console.error("[PersonMemory] Failed to load saved memory:", cause)
        if (!cancelled) setMemorySummary(null)
      })

    return () => {
      cancelled = true
    }
  }, [faceMatch?.name])

  // Persist one downscaled crop only after a face survives several frames.
  // This filters one-frame false positives and avoids uploading video frames.
  useEffect(() => {
    const video = videoRef.current
    if (!video || !isStreaming || !isVideoReady) {
      return
    }

    detectedFaces
      .filter(
        (face) =>
          face.frameCount >= 5 &&
          !storedFaceIdsRef.current.has(face.id) &&
          !storingFaceIdsRef.current.has(face.id) &&
          (faceStorageRetryAtRef.current.get(face.id) ?? 0) <= Date.now()
      )
      .forEach((face) => {
        storingFaceIdsRef.current.add(face.id)

        void (async () => {
          try {
            if (!faceSessionIdRef.current) {
              faceSessionIdRef.current = crypto.randomUUID()
            }

            const imageDataUrl = captureFaceCrop(video, face)
            if (!imageDataUrl) {
              throw new Error("Unable to capture the detected face crop")
            }

            const response = await fetch(`${INFERENCE_BACKEND_URL}/api/faces/observations`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                session_id: faceSessionIdRef.current,
                face_id: face.id,
                confidence: face.confidence,
                bounding_box: face.boundingBox,
                image_data_url: imageDataUrl,
              }),
            })

            if (!response.ok) {
              throw new Error(`Face storage returned ${response.status}`)
            }

            storedFaceIdsRef.current.add(face.id)
            faceStorageRetryAtRef.current.delete(face.id)
            setFaceStorageError(null)
            console.info(`[FaceDetection] Stored observation for ${face.id}`)
          } catch (cause) {
            const message = cause instanceof Error ? cause.message : "Face storage failed"
            console.error("[FaceDetection] Failed to store face observation:", cause)
            setFaceStorageError(message)
            faceStorageRetryAtRef.current.set(face.id, Date.now() + 5_000)
          } finally {
            storingFaceIdsRef.current.delete(face.id)
          }
        })()
      })
  }, [detectedFaces, isStreaming, isVideoReady])

  // Record first-seen timestamps for new faces (runs on every detection update)
  useEffect(() => {
    if (detectedFaces.length === 0) return
    const now = Date.now()
    detectedFaces.forEach((face) => {
      if (!faceFirstSeenRef.current.has(face.id)) {
        faceFirstSeenRef.current.set(face.id, now)
      }
    })
  }, [detectedFaces])

  // Periodically check if any faces should be marked as unidentified
  // Uses an interval instead of setTimeout to avoid being cancelled by detectedFaces updates
  useEffect(() => {
    const interval = setInterval(() => {
      const now = Date.now()
      setUnidentifiedFaceIds((prev) => {
        const next = new Set(prev)
        for (const [faceId, firstSeen] of faceFirstSeenRef.current.entries()) {
          if (now - firstSeen >= UNIDENTIFIED_TIMEOUT_MS) {
            next.add(faceId)
          }
        }
        return next
      })
    }, 1000) // Check every second

    return () => clearInterval(interval)
  }, [])

  // Clean up lost faces
  useEffect(() => {
    const currentFaceIds = new Set(detectedFaces.map(f => f.id))
    setUnidentifiedFaceIds((prev) => {
      const next = new Set(prev)
      for (const faceId of next) {
        if (!currentFaceIds.has(faceId)) {
          next.delete(faceId)
        }
      }
      return next
    })
    // Clean up first-seen timestamps for lost faces
    for (const faceId of faceFirstSeenRef.current.keys()) {
      if (!currentFaceIds.has(faceId)) {
        faceFirstSeenRef.current.delete(faceId)
      }
    }
  }, [detectedFaces])

  useEffect(() => {
    startWebcam()

    return () => {
      stopWebcam()
    }
  }, [])

  const startWebcam = async () => {
    try {
      console.log('[Webcam] Requesting media access...')
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: true,
      })

      if (videoRef.current) {
        const video = videoRef.current

        const handleMetadataLoaded = () => {
          console.log('[Webcam] Video metadata loaded:', {
            width: video.videoWidth,
            height: video.videoHeight
          })
          setIsVideoReady(true)
        }

        video.addEventListener('loadedmetadata', handleMetadataLoaded)

        if (video.videoWidth > 0 && video.videoHeight > 0) {
          handleMetadataLoaded()
        }

        video.srcObject = stream
        streamRef.current = stream
        stream.getAudioTracks().forEach((track) => {
          track.enabled = !isMuted
        })
        setIsStreaming(true)
        console.log('[Webcam] Stream started')

      }
    } catch (err) {
      console.error("[Webcam] Error accessing webcam:", err)
    }
  }

  const stopWebcam = () => {
    console.log('[Webcam] Stopping stream')

    if (videoRef.current?.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream
      stream.getTracks().forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }

    streamRef.current = null
    setIsStreaming(false)
    setIsVideoReady(false)
    setIsRayBanMode(false)
    setIsMuted(false)
    storedFaceIdsRef.current.clear()
    storingFaceIdsRef.current.clear()
    faceStorageRetryAtRef.current.clear()
    faceSessionIdRef.current = ""
  }

  const toggleMute = useCallback(() => {
    const stream = streamRef.current
    if (!stream) {
      return
    }
    setIsMuted((prev) => {
      const next = !prev
      stream.getAudioTracks().forEach((track) => {
        track.enabled = !next
      })
      return next
    })
  }, [])

  const faceNotifications = detectedFaces.map((face) => {
    const video = videoRef.current
    const overlay = overlayRef.current

    if (!video || !overlay) return null

    const videoWidth = video.videoWidth
    const videoHeight = video.videoHeight
    const overlayWidth = overlay.clientWidth
    const overlayHeight = overlay.clientHeight

    if (videoWidth === 0 || videoHeight === 0) return null

    const transform = calculateVideoTransform(
      videoWidth,
      videoHeight,
      overlayWidth,
      overlayHeight
    )

    const overlayBox = mapBoundingBoxToOverlay(
      face.boundingBox,
      transform,
      overlayWidth,
      true
    )

    const position = calculateNotificationPosition(
      overlayBox,
      overlayWidth,
      overlayHeight
    )

    return {
      face,
      position,
      overlayBox,
    }
  }).filter((n) => n !== null)

  const primaryFaceId = detectedFaces.length
    ? detectedFaces.reduce((largest, face) => {
        const largestArea = largest.boundingBox.width * largest.boundingBox.height
        const area = face.boundingBox.width * face.boundingBox.height
        return area > largestArea ? face : largest
      }).id
    : null

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black select-none" style={{ fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', 'Helvetica Neue', Arial, sans-serif", WebkitFontSmoothing: "antialiased" } as React.CSSProperties}>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className={cn(
          "absolute inset-0 h-full w-full object-cover transition-all duration-300",
          isRayBanMode ? "scale-[1.08] blur-[13px]" : "scale-100"
        )}
        style={{ transform: 'scaleX(-1)' }}
      />

      {!isRayBanMode && (
        <>
          {/* ── Face overlays ── */}
          <div ref={overlayRef} className="absolute inset-0 pointer-events-none">
            {faceNotifications.map((notification) => {
              const isRecognized = notification!.face.id === primaryFaceId && Boolean(faceMatch)
              const isUnidentified = !isRecognized && unidentifiedFaceIds.has(notification!.face.id)
              return (
                <FaceNotification
                  key={notification!.face.id}
                  faceId={notification!.face.id}
                  box={notification!.overlayBox}
                  left={notification!.position.left}
                  top={notification!.position.top}
                  confidence={notification!.face.confidence}
                  name={isRecognized ? faceMatch?.name : undefined}
                  description={newMemorySummary ?? memorySummary ?? undefined}
                  isUnidentified={isUnidentified}
                />
              )
            })}
          </div>

          {/* ── Top-right: camera status pill ── */}
          <div style={{
            position: "absolute", top: 16, right: 16,
            display: "flex", alignItems: "center", gap: 7,
            padding: "6px 14px",
            borderRadius: 24,
            background: "rgba(0,0,0,0.55)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            border: "1px solid rgba(255,255,255,0.10)",
            boxShadow: "0 2px 14px rgba(0,0,0,0.4)",
            zIndex: 50,
          } as React.CSSProperties}>
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              background: isStreaming ? "#30d158" : "#ff453a",
              boxShadow: isStreaming ? "0 0 8px #30d158" : "0 0 8px #ff453a",
              flexShrink: 0,
              ...(isStreaming ? { animation: "pulse 2s ease-in-out infinite" } : {}),
            }} />
            <span style={{ fontSize: 12, fontWeight: 500, color: "rgba(255,255,255,0.80)", letterSpacing: "-0.1px", textShadow: "0 1px 4px rgba(0,0,0,0.8)" }}>
              {isStreaming ? "Camera Live" : "Camera Off"}
            </span>
          </div>

          {/* ── Top-right: speech/mic status ── */}
          <div style={{
            position: "absolute", top: 52, right: 16,
            maxWidth: 240,
            padding: "6px 12px",
            borderRadius: 12,
            background: "rgba(0,0,0,0.48)",
            backdropFilter: "blur(16px)",
            WebkitBackdropFilter: "blur(16px)",
            border: "1px solid rgba(255,255,255,0.08)",
            zIndex: 50,
          } as React.CSSProperties}>
            <span style={{ fontSize: 11.5, fontWeight: 400, color: "rgba(255,255,255,0.58)", textShadow: "0 1px 3px rgba(0,0,0,0.9)", lineHeight: 1.4 }}>
              {!faceMatch?.name
                ? "Recognize or enroll a face to save memory"
                : speechStatus === "transcribing"
                  ? "✦ Updating memory…"
                  : isListening
                    ? "● Listening…"
                    : isMuted ? "Mic paused" : "Ready"}
            </span>
          </div>

          {/* ── Top-left: enrollment panel ── */}
          <div style={{
            position: "absolute", top: 80, left: 16,
            width: 268,
            borderRadius: 18,
            background: "rgba(0,0,0,0.60)",
            backdropFilter: "blur(24px)",
            WebkitBackdropFilter: "blur(24px)",
            border: "1px solid rgba(255,255,255,0.10)",
            boxShadow: "0 8px 32px rgba(0,0,0,0.45), 0 0 0 0.5px rgba(255,255,255,0.06)",
            padding: "14px 15px",
            zIndex: 50,
          } as React.CSSProperties}>
            {/* Panel header */}
            <div style={{ marginBottom: 10 }}>
              <p style={{ margin: 0, fontSize: 13.5, fontWeight: 700, color: "#fff", letterSpacing: "-0.2px", textShadow: "0 1px 6px rgba(0,0,0,0.8)" }}>
                {faceMatch ? `👤 ${faceMatch.name}` : "Enroll Face"}
              </p>
              <p style={{ margin: "3px 0 0", fontSize: 11, fontWeight: 400, color: "rgba(255,255,255,0.48)", textShadow: "0 1px 3px rgba(0,0,0,0.8)", lineHeight: 1.35 }}>
                {isFaceRecognitionLoading ? "Loading recognition model…" : "Enter a name and keep your face visible."}
              </p>
            </div>

            {/* Divider */}
            <div style={{ height: 1, background: "linear-gradient(to right, transparent, rgba(255,255,255,0.09), transparent)", marginBottom: 10 }} />

            {/* Input + button row */}
            <div style={{ display: "flex", gap: 7 }}>
              <input
                value={enrollmentName}
                onChange={(e) => setEnrollmentName(e.target.value)}
                placeholder="Your name"
                style={{
                  flex: 1, minWidth: 0,
                  padding: "8px 11px",
                  borderRadius: 11,
                  background: "rgba(255,255,255,0.07)",
                  border: "1px solid rgba(255,255,255,0.13)",
                  color: "#fff",
                  fontSize: 12.5,
                  fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                  fontWeight: 400,
                  outline: "none",
                  letterSpacing: "-0.1px",
                } as React.CSSProperties}
                onFocus={(e) => { e.currentTarget.style.border = "1px solid rgba(10,132,255,0.7)" }}
                onBlur={(e) => { e.currentTarget.style.border = "1px solid rgba(255,255,255,0.13)" }}
              />
              <button
                onClick={enrollCurrentFace}
                disabled={isFaceRecognitionLoading || !isStreaming || !enrollmentName.trim()}
                style={{
                  padding: "8px 14px",
                  borderRadius: 11,
                  background: isFaceRecognitionLoading || !isStreaming || !enrollmentName.trim()
                    ? "rgba(10,132,255,0.3)"
                    : "rgba(10,132,255,0.85)",
                  border: "none",
                  color: isFaceRecognitionLoading || !isStreaming || !enrollmentName.trim()
                    ? "rgba(255,255,255,0.4)"
                    : "#fff",
                  fontSize: 12.5,
                  fontWeight: 600,
                  cursor: isFaceRecognitionLoading || !isStreaming || !enrollmentName.trim() ? "not-allowed" : "pointer",
                  fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                  letterSpacing: "-0.1px",
                  whiteSpace: "nowrap",
                  transition: "background 200ms, color 200ms",
                } as React.CSSProperties}
              >
                Enroll
              </button>
            </div>

            <div style={{ display: "flex", justifyContent: "flex-end", marginTop: 8 }}>
              <button
                onClick={clearAllProfiles}
                style={{
                  background: "none",
                  border: "none",
                  color: "rgba(255,107,107,0.75)",
                  fontSize: 11,
                  fontWeight: 500,
                  cursor: "pointer",
                  textDecoration: "underline",
                  padding: "2px 4px",
                }}
              >
                Reset / Clear All Memory & Face Data
              </button>
            </div>

            {enrollmentMessage && (
              <p style={{
                margin: "7px 0 0",
                fontSize: 11.5, fontWeight: 500,
                color: enrollmentMessage.startsWith("Enrolled") ? "#30d158" : "#ff9f0a",
                textShadow: "0 1px 3px rgba(0,0,0,0.8)",
              }}>
                {enrollmentMessage}
              </p>
            )}

          </div>

          {/* ── Top-left: AI init / error pills ── */}
          {isFaceDetectionLoading && (
            <div style={{
              position: "absolute", top: 16, left: 16,
              display: "flex", alignItems: "center", gap: 7,
              padding: "6px 14px",
              borderRadius: 24,
              background: "rgba(0,0,0,0.55)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              border: "1px solid rgba(10,132,255,0.25)",
              boxShadow: "0 2px 14px rgba(10,132,255,0.15)",
              zIndex: 50,
            } as React.CSSProperties}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#0a84ff", boxShadow: "0 0 8px #0a84ff", animation: "pulse 1.5s infinite", flexShrink: 0 }} />
              <span style={{ fontSize: 12, fontWeight: 500, color: "#64b5f6", textShadow: "0 1px 4px rgba(0,0,0,0.9)" }}>Initializing Vision AI…</span>
            </div>
          )}
          {faceDetectionError && (
            <div style={{
              position: "absolute", top: 16, left: 16,
              padding: "6px 13px", borderRadius: 24,
              background: "rgba(255,59,48,0.12)",
              backdropFilter: "blur(16px)",
              border: "1px solid rgba(255,59,48,0.25)",
              zIndex: 50,
            } as React.CSSProperties}>
              <span style={{ fontSize: 11.5, color: "#ff6b6b", textShadow: "0 1px 4px rgba(0,0,0,0.9)" }}>⚠ {faceDetectionError}</span>
            </div>
          )}
          {faceStorageError && (
            <div style={{
              position: "absolute", top: 54, left: 16,
              padding: "6px 13px", borderRadius: 24,
              background: "rgba(255,59,48,0.10)",
              backdropFilter: "blur(16px)",
              border: "1px solid rgba(255,59,48,0.20)",
              zIndex: 50,
            } as React.CSSProperties}>
              <span style={{ fontSize: 11.5, color: "rgba(255,107,107,0.85)", textShadow: "0 1px 4px rgba(0,0,0,0.9)" }}>⚠ {faceStorageError}</span>
            </div>
          )}
          {faceRecognitionError && (
            <div style={{
              position: "absolute", top: faceStorageError ? 90 : 54, left: 16,
              padding: "6px 13px", borderRadius: 24,
              background: "rgba(255,59,48,0.10)",
              backdropFilter: "blur(16px)",
              border: "1px solid rgba(255,59,48,0.20)",
              zIndex: 50,
            } as React.CSSProperties}>
              <span style={{ fontSize: 11.5, color: "rgba(255,107,107,0.85)", textShadow: "0 1px 4px rgba(0,0,0,0.9)" }}>⚠ {faceRecognitionError}</span>
            </div>
          )}
          {speechError && (
            <div style={{
              position: "absolute", top: 88, right: 16,
              padding: "6px 13px", borderRadius: 24,
              background: "rgba(255,59,48,0.10)",
              backdropFilter: "blur(16px)",
              border: "1px solid rgba(255,59,48,0.20)",
              zIndex: 50,
            } as React.CSSProperties}>
              <span style={{ fontSize: 11.5, color: "rgba(255,107,107,0.85)", textShadow: "0 1px 4px rgba(0,0,0,0.9)" }}>⚠ {speechError}</span>
            </div>
          )}

          {lastSpeakerStatus && (
            <div style={{
              position: "absolute", top: 16, right: 16,
              display: "flex", alignItems: "center", gap: 7,
              padding: "6px 14px",
              borderRadius: 24,
              background: isVoiceVerified ? "rgba(48,209,88,0.15)" : "rgba(255,159,10,0.15)",
              backdropFilter: "blur(20px)",
              WebkitBackdropFilter: "blur(20px)",
              border: isVoiceVerified ? "1px solid rgba(48,209,88,0.35)" : "1px solid rgba(255,159,10,0.35)",
              boxShadow: isVoiceVerified ? "0 2px 14px rgba(48,209,88,0.2)" : "0 2px 14px rgba(255,159,10,0.2)",
              zIndex: 50,
            } as React.CSSProperties}>
              <span style={{ width: 6, height: 6, borderRadius: "50%", background: isVoiceVerified ? "#30d158" : "#ff9f0a", boxShadow: `0 0 8px ${isVoiceVerified ? "#30d158" : "#ff9f0a"}`, flexShrink: 0 }} />
              <span style={{ fontSize: 12, fontWeight: 600, color: isVoiceVerified ? "#30d158" : "#ff9f0a", textShadow: "0 1px 4px rgba(0,0,0,0.9)" }}>
                {lastSpeakerStatus}
              </span>
            </div>
          )}
        </>
      )}


      <RayBanOverlay stream={streamRef.current} videoRef={videoRef} visible={isRayBanMode} />

      {/* ── Bottom dock ─────────────────────────────────────────────────── */}
      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0,
        display: "flex", justifyContent: "center", alignItems: "center",
        padding: "20px 24px 28px",
        background: "linear-gradient(to top, rgba(0,0,0,0.75) 0%, rgba(0,0,0,0.2) 70%, transparent 100%)",
        zIndex: 50,
        pointerEvents: "auto",
      } as React.CSSProperties}>
        <div style={{
          display: "flex", alignItems: "center", gap: 10,
          padding: "10px 18px",
          borderRadius: 40,
          background: "rgba(0,0,0,0.62)",
          backdropFilter: "blur(28px)",
          WebkitBackdropFilter: "blur(28px)",
          border: "1px solid rgba(255,255,255,0.10)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.5), 0 0 0 0.5px rgba(255,255,255,0.05)",
        } as React.CSSProperties}>

          {/* Mic button */}
          <button
            onClick={toggleMute}
            disabled={!isStreaming}
            style={{
              width: 44, height: 44, borderRadius: "50%",
              background: isMuted ? "rgba(255,59,48,0.75)" : "rgba(255,255,255,0.10)",
              border: isMuted ? "1px solid rgba(255,59,48,0.5)" : "1px solid rgba(255,255,255,0.14)",
              display: "flex", alignItems: "center", justifyContent: "center",
              cursor: !isStreaming ? "not-allowed" : "pointer",
              opacity: !isStreaming ? 0.4 : 1,
              transition: "background 200ms, border 200ms, transform 150ms",
              boxShadow: isMuted ? "0 0 16px rgba(255,59,48,0.3)" : "none",
            } as React.CSSProperties}
            onMouseEnter={(e) => { if (isStreaming) (e.currentTarget as HTMLButtonElement).style.transform = "scale(1.08)" }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(1)" }}
            aria-pressed={isMuted}
          >
            {isMuted
              ? <MicOff style={{ width: 18, height: 18, color: "#fff" }} />
              : <Mic style={{ width: 18, height: 18, color: "rgba(255,255,255,0.9)" }} />}
          </button>
          <span style={{ fontSize: 11.5, fontWeight: 500, color: "rgba(255,255,255,0.55)", letterSpacing: "-0.1px", textShadow: "0 1px 3px rgba(0,0,0,0.8)", minWidth: 76 }}>
            {isMuted ? "Mic off" : !faceMatch?.name ? "Waiting" : isListening ? "Listening…" : "Ready"}
          </span>

          {/* Separator */}
          <div style={{ width: 1, height: 24, background: "rgba(255,255,255,0.12)" }} />

          {/* Camera button */}
          <button
            onClick={isStreaming ? stopWebcam : startWebcam}
            style={{
              width: 44, height: 44, borderRadius: "50%",
              background: isStreaming ? "rgba(10,132,255,0.75)" : "rgba(255,255,255,0.10)",
              border: isStreaming ? "1px solid rgba(10,132,255,0.5)" : "1px solid rgba(255,255,255,0.14)",
              display: "flex", alignItems: "center", justifyContent: "center",
              cursor: "pointer",
              transition: "background 200ms, border 200ms, transform 150ms",
              boxShadow: isStreaming ? "0 0 18px rgba(10,132,255,0.35)" : "none",
            } as React.CSSProperties}
            onMouseEnter={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(1.08)" }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(1)" }}
          >
            {isStreaming
              ? <Video style={{ width: 18, height: 18, color: "#fff" }} />
              : <VideoOff style={{ width: 18, height: 18, color: "rgba(255,255,255,0.75)" }} />}
          </button>
          <span style={{ fontSize: 11.5, fontWeight: 500, color: "rgba(255,255,255,0.55)", letterSpacing: "-0.1px", textShadow: "0 1px 3px rgba(0,0,0,0.8)", minWidth: 60 }}>
            {isStreaming ? "Camera On" : "Start"}
          </span>

          {/* Separator */}
          <div style={{ width: 1, height: 24, background: "rgba(255,255,255,0.12)" }} />

          {/* Ray-Ban toggle */}
          <button
            onClick={() => setIsRayBanMode((p) => !p)}
            disabled={!isStreaming}
            style={{
              padding: "9px 18px",
              borderRadius: 22,
              background: isRayBanMode ? "rgba(10,132,255,0.70)" : "rgba(255,255,255,0.07)",
              border: isRayBanMode ? "1px solid rgba(10,132,255,0.5)" : "1px solid rgba(255,255,255,0.12)",
              color: isRayBanMode ? "#fff" : "rgba(255,255,255,0.65)",
              fontSize: 12, fontWeight: 600,
              fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
              letterSpacing: "-0.1px",
              cursor: !isStreaming ? "not-allowed" : "pointer",
              opacity: !isStreaming ? 0.4 : 1,
              transition: "background 200ms, border 200ms, color 200ms, transform 150ms",
              whiteSpace: "nowrap",
              textShadow: "0 1px 3px rgba(0,0,0,0.8)",
            } as React.CSSProperties}
            onMouseEnter={(e) => { if (isStreaming) (e.currentTarget as HTMLButtonElement).style.transform = "scale(1.04)" }}
            onMouseLeave={(e) => { (e.currentTarget as HTMLButtonElement).style.transform = "scale(1)" }}
          >
            👓 {isRayBanMode ? "Exit Glasses View" : "Ray-Ban View"}
          </button>
        </div>
      </div>
    </div>
  )
}
