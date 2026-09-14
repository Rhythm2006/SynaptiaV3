"use client"

const INFERENCE_BACKEND_URL = "http://localhost:8002"

/**
 * Extracts a normalized 128-dimensional acoustic spectral voice descriptor
 * from Web Audio Analyser frequency data.
 */
export function extractVoiceDescriptor(frequencyData: Float32Array): number[] {
  const descriptorLength = 128
  const step = Math.max(1, Math.floor(frequencyData.length / descriptorLength))
  const descriptor: number[] = new Array(descriptorLength).fill(0)

  for (let i = 0; i < descriptorLength; i++) {
    const idx = Math.min(i * step, frequencyData.length - 1)
    // Decibels typically range from -100 dB (quiet) to 0 dB (loud peak)
    // Normalize dB values to a positive [0, 1] range:
    const db = frequencyData[idx]
    descriptor[i] = Math.max(0, (db + 100) / 100)
  }

  // L2 normalize the descriptor vector
  const norm = Math.sqrt(descriptor.reduce((sum, val) => sum + val * val, 0))
  if (norm === 0) return descriptor
  return descriptor.map((val) => Number((val / norm).toFixed(5)))
}

/**
 * Computes cosine similarity between two voice descriptor vectors.
 */
export function computeVoiceSimilarity(vecA: number[], vecB: number[]): number {
  if (!vecA || !vecB || vecA.length === 0 || vecB.length === 0) return 0
  const len = Math.min(vecA.length, vecB.length)
  let dot = 0
  let normA = 0
  let normB = 0
  for (let i = 0; i < len; i++) {
    dot += vecA[i] * vecB[i]
    normA += vecA[i] * vecA[i]
    normB += vecB[i] * vecB[i]
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB)
  return denom === 0 ? 0 : dot / denom
}

/**
 * Saves the enrolled acoustic voice descriptor for a named user.
 */
export async function enrollVoiceprint(personName: string, descriptor: number[]): Promise<boolean> {
  const cleanName = personName.trim()
  if (!cleanName || descriptor.length === 0) return false

  // Store in browser localStorage for offline capability
  try {
    localStorage.setItem(`synaptia_voice_${cleanName.toLowerCase()}`, JSON.stringify(descriptor))
  } catch {}

  // Persist to inference service backend
  try {
    const res = await fetch(`${INFERENCE_BACKEND_URL}/api/voice-identities`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: cleanName,
        descriptor,
      }),
    })
    return res.ok
  } catch (err) {
    console.warn("[VoiceDescriptor] Backend voice enrollment failed (cached locally):", err)
    return true
  }
}

/**
 * Loads the enrolled voice descriptor for a person from backend or local storage.
 */
export async function getEnrolledVoiceprint(personName: string): Promise<number[] | null> {
  const cleanName = personName.trim().toLowerCase()
  if (!cleanName) return null

  // 1. Check localStorage first
  try {
    const local = localStorage.getItem(`synaptia_voice_${cleanName}`)
    if (local) {
      const parsed = JSON.parse(local)
      if (Array.isArray(parsed) && parsed.length > 0) return parsed
    }
  } catch {}

  // 2. Fetch from backend
  try {
    const res = await fetch(`${INFERENCE_BACKEND_URL}/api/voice-identities/${encodeURIComponent(cleanName)}`)
    if (res.ok) {
      const data = await res.json()
      if (data.descriptor && Array.isArray(data.descriptor)) {
        try {
          localStorage.setItem(`synaptia_voice_${cleanName}`, JSON.stringify(data.descriptor))
        } catch {}
        return data.descriptor
      }
    }
  } catch {}

  return null
}
