"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import * as faceapi from "face-api.js"

export interface FaceIdentity {
  name: string
  descriptor: number[]
}

export interface FaceMatch {
  name: string
  distance: number
}

interface EnrollmentResult {
  success: boolean
  error?: string
}

const MODEL_URL = "/models/face-api"
const MATCH_THRESHOLD = 0.52

function descriptorMatch(descriptor: Float32Array, identities: FaceIdentity[]): FaceMatch | null {
  let closest: FaceMatch | null = null

  for (const identity of identities) {
    if (identity.descriptor.length !== descriptor.length) {
      continue
    }
    const distance = faceapi.euclideanDistance(descriptor, new Float32Array(identity.descriptor))
    if (!closest || distance < closest.distance) {
      closest = { name: identity.name, distance }
    }
  }

  return closest && closest.distance <= MATCH_THRESHOLD ? closest : null
}

const LOCAL_STORAGE_KEY = "synaptia_v2_face_identities"

function getLocalIdentities(): FaceIdentity[] {
  if (typeof window === "undefined") return []
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_KEY)
    return raw ? (JSON.parse(raw) as FaceIdentity[]) : []
  } catch {
    return []
  }
}

function saveLocalIdentities(identities: FaceIdentity[]) {
  if (typeof window === "undefined") return
  try {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(identities))
  } catch (err) {
    console.warn("[FaceRecognition] Failed to save to localStorage:", err)
  }
}

// Number of consecutive missed detections required before clearing the match.
// This prevents the name card from flickering on single bad frames.
const MISS_THRESHOLD = 4

export function useFaceRecognition(video: HTMLVideoElement | null, enabled: boolean) {
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [match, setMatch] = useState<FaceMatch | null>(null)
  const identitiesRef = useRef<FaceIdentity[]>(getLocalIdentities())
  const recognitionInFlightRef = useRef(false)
  const missCountRef = useRef(0)

  const loadIdentities = useCallback(async () => {
    // 1. First load immediately from local storage
    const local = getLocalIdentities()
    if (local.length > 0) {
      identitiesRef.current = local
    }

    // 2. Fetch from backend and merge
    try {
      const response = await fetch("http://localhost:8002/api/face-identities")
      if (response.ok) {
        const body = (await response.json()) as { identities?: FaceIdentity[] }
        const serverIdentities = body.identities ?? []
        
        // Merge without duplicates by name
        const map = new Map<string, FaceIdentity>()
        for (const id of local) map.set(id.name.toLowerCase(), id)
        for (const id of serverIdentities) map.set(id.name.toLowerCase(), id)

        const merged = Array.from(map.values())
        identitiesRef.current = merged
        saveLocalIdentities(merged)
      }
    } catch (err) {
      console.warn("[FaceRecognition] Could not sync identities with server, using local copy:", err)
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        setIsLoading(true)
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
          loadIdentities(),
        ])
        if (!cancelled) {
          setError(null)
          setIsLoading(false)
        }
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : "Unable to load face recognition"
        console.error("[FaceRecognition] Initialization error:", cause)
        if (!cancelled) {
          setError(message)
          setIsLoading(false)
        }
      }
    }

    void load()
    return () => {
      cancelled = true
    }
  }, [loadIdentities])

  const detectDescriptor = useCallback(async () => {
    if (!video || video.videoWidth === 0 || video.videoHeight === 0) {
      return null
    }

    return faceapi
      .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.5 }))
      .withFaceLandmarks()
      .withFaceDescriptor()
  }, [video])

  useEffect(() => {
    if (!enabled || isLoading || !video) {
      return
    }

    const interval = window.setInterval(() => {
      if (recognitionInFlightRef.current) {
        return
      }
      recognitionInFlightRef.current = true

      void (async () => {
        try {
          const detection = await detectDescriptor()
          const candidate = detection
            ? descriptorMatch(detection.descriptor, identitiesRef.current)
            : null

          if (candidate) {
            // Got a positive match — reset miss counter and update immediately
            missCountRef.current = 0
            setMatch(candidate)
          } else {
            // No match this frame — only clear after MISS_THRESHOLD consecutive misses
            missCountRef.current += 1
            if (missCountRef.current >= MISS_THRESHOLD) {
              setMatch(null)
            }
          }
        } catch (cause) {
          console.error("[FaceRecognition] Matching error:", cause)
        } finally {
          recognitionInFlightRef.current = false
        }
      })()
    }, 1200)

    return () => window.clearInterval(interval)
  }, [detectDescriptor, enabled, isLoading, video])

  const enroll = useCallback(async (name: string): Promise<EnrollmentResult> => {
    const trimmedName = name.trim()
    if (!trimmedName) {
      return { success: false, error: "Enter your name before enrolling." }
    }

    try {
      const detection = await detectDescriptor()
      if (!detection) {
        return { success: false, error: "Keep one face clearly in view, then try again." }
      }

      const newIdentity: FaceIdentity = {
        name: trimmedName,
        descriptor: Array.from(detection.descriptor),
      }
      const currentLocal = getLocalIdentities().filter(
        (id) => id.name.toLowerCase() !== trimmedName.toLowerCase()
      )
      currentLocal.push(newIdentity)
      saveLocalIdentities(currentLocal)
      identitiesRef.current = currentLocal

      try {
        await fetch("http://localhost:8002/api/face-identities", {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(newIdentity),
        })
      } catch (err) {
        console.warn("[FaceRecognition] Could not sync enrolled identity to backend:", err)
      }

      setMatch({ name: trimmedName, distance: 0 })
      return { success: true }
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : "Face enrollment failed"
      console.error("[FaceRecognition] Enrollment error:", cause)
      return { success: false, error: message }
    }
  }, [detectDescriptor, loadIdentities])

  return { isLoading, error, match, enroll }
}
