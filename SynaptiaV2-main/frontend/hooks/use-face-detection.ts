"use client"

import { useEffect, useRef, useState } from "react"
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision"
import type { DetectedFace, Detection } from "@/lib/face-tracker"
import { FaceTracker } from "@/lib/face-tracker"

export interface UseFaceDetectionOptions {
  enabled?: boolean
  minDetectionConfidence?: number
  targetFps?: number
  /** Kept for API compatibility. MediaPipe's detector must receive a video element. */
  useWorker?: boolean
}

export interface UseFaceDetectionResult {
  detectedFaces: DetectedFace[]
  isLoading: boolean
  error: string | null
}

const MEDIAPIPE_VERSION = "0.10.22-rc.20250304"
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

/**
 * Detect faces directly from the HTMLVideoElement.
 *
 * The previous worker implementation copied every frame to ImageData and then
 * attempted to initialise MediaPipe's GPU/WebGL path in a module worker. That
 * combination is not reliable across browsers and made failures hard to surface.
 * Supplying the video element in VIDEO mode is the supported live-camera API and
 * preserves the exact coordinates used to draw the camera preview.
 */
export function useFaceDetection(
  videoElement: HTMLVideoElement | null,
  options: UseFaceDetectionOptions = {}
): UseFaceDetectionResult {
  const {
    enabled = true,
    minDetectionConfidence = 0.5,
    targetFps = 12,
    useWorker: _useWorker = false,
  } = options

  const [detectedFaces, setDetectedFaces] = useState<DetectedFace[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const detectorRef = useRef<FaceDetector | null>(null)
  const faceTrackerRef = useRef(new FaceTracker())
  const rafIdRef = useRef<number | null>(null)
  const lastDetectionTimeRef = useRef(0)

  useEffect(() => {
    let cancelled = false
    let detector: FaceDetector | null = null

    const initialize = async () => {
      try {
        setIsLoading(true)
        setError(null)

        const vision = await FilesetResolver.forVisionTasks(WASM_ROOT)
        detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: FACE_MODEL,
            // CPU works across browsers and avoids a WebGL delegate dependency.
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          minDetectionConfidence,
        })

        if (cancelled) {
          detector.close()
          return
        }

        detectorRef.current = detector
        setIsLoading(false)
      } catch (cause) {
        const message = cause instanceof Error ? cause.message : "Failed to load face detection"
        console.error("[FaceDetection] Initialization error:", cause)
        if (!cancelled) {
          setError(message)
          setIsLoading(false)
        }
      }
    }

    void initialize()

    return () => {
      cancelled = true
      if (detectorRef.current === detector) {
        detectorRef.current = null
      }
      detector?.close()
    }
  }, [minDetectionConfidence])

  useEffect(() => {
    if (!enabled || !videoElement || isLoading || !detectorRef.current) {
      return
    }

    const frameIntervalMs = 1000 / Math.max(targetFps, 1)

    const detectFrame = () => {
      const detector = detectorRef.current
      if (!detector || !enabled || !videoElement) {
        return
      }

      const now = performance.now()
      if (
        now - lastDetectionTimeRef.current >= frameIntervalMs &&
        videoElement.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        videoElement.videoWidth > 0 &&
        videoElement.videoHeight > 0
      ) {
        try {
          const result = detector.detectForVideo(videoElement, now)
          const detections: Detection[] = (result.detections ?? []).map((detection) => ({
            boundingBox: {
              originX: detection.boundingBox?.originX ?? 0,
              originY: detection.boundingBox?.originY ?? 0,
              width: detection.boundingBox?.width ?? 0,
              height: detection.boundingBox?.height ?? 0,
            },
            confidence: detection.categories?.[0]?.score ?? 0,
          }))

          setDetectedFaces(faceTrackerRef.current.update(detections))
          lastDetectionTimeRef.current = now
        } catch (cause) {
          const message = cause instanceof Error ? cause.message : "Face detection failed"
          console.error("[FaceDetection] Detection error:", cause)
          setError(message)
        }
      }

      rafIdRef.current = requestAnimationFrame(detectFrame)
    }

    rafIdRef.current = requestAnimationFrame(detectFrame)
    return () => {
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current)
        rafIdRef.current = null
      }
    }
  }, [enabled, isLoading, targetFps, videoElement])

  useEffect(() => {
    if (!enabled) {
      setDetectedFaces([])
      faceTrackerRef.current.clear()
      lastDetectionTimeRef.current = 0
    }
  }, [enabled])

  return { detectedFaces, isLoading, error }
}
