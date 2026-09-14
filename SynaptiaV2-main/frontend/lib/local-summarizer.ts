"use client"

let summarizerPipeline: any = null
let loadingPromise: Promise<any> | null = null

/**
 * Initializes and returns the in-browser Hugging Face Transformers.js summarization pipeline.
 * Uses ONNX-quantized Xenova/distilbart-cnn-6-6 running locally in WebAssembly/WebGPU.
 */
export async function getLocalSummarizer() {
  if (typeof window === "undefined") return null
  if (summarizerPipeline) return summarizerPipeline
  if (loadingPromise) return loadingPromise

  loadingPromise = (async () => {
    try {
      const { pipeline, env } = await import("@huggingface/transformers")
      env.allowLocalModels = false
      env.useBrowserCache = true

      console.log("[LocalSummarizer] Initializing in-browser Hugging Face neural pipeline...")
      summarizerPipeline = await pipeline("summarization", "Xenova/distilbart-cnn-6-6", {
        dtype: "q8",
      })
      console.log("[LocalSummarizer] Hugging Face in-browser summarizer ready.")
      return summarizerPipeline
    } catch (err) {
      console.warn("[LocalSummarizer] Could not load in-browser Hugging Face pipeline, using fallback:", err)
      return null
    } finally {
      loadingPromise = null
    }
  })()

  return loadingPromise
}

/**
 * Transforms first-person dialogue into third-person factual memory summaries.
 */
function toThirdPersonSummary(text: string, personName?: string): string {
  let cleaned = text
    .replace(/\b(i am|i'm)\b/gi, `${personName || "They"} is`)
    .replace(/\b(i have|i've)\b/gi, `${personName || "They"} has`)
    .replace(/\b(i will|i'll)\b/gi, `${personName || "They"} will`)
    .replace(/\b(i was)\b/gi, `${personName || "They"} was`)
    .replace(/\b(i|me|my)\b/gi, personName ? `${personName}'s` : "their")
    .replace(/\b(you are|you're)\b/gi, "you are")
    .replace(/\b(we are|we're)\b/gi, "Both discussed")
    .replace(/\b(we have|we've)\b/gi, "Both have")
    .replace(/\b(we)\b/gi, "Both")
    .replace(/\s+/g, " ")
    .trim()

  // Capitalize first letter
  if (cleaned) {
    cleaned = cleaned.charAt(0).toUpperCase() + cleaned.slice(1)
  }

  return cleaned.endsWith(".") ? cleaned : `${cleaned}.`
}

/**
 * Formats and caps raw memory text into maximum 2-3 concise highlight bullet points (<=25 words).
 */
export function formatLocalHighlights(
  text: string,
  personName?: string,
  maxPoints: number = 3,
  maxWords: number = 25
): string {
  if (!text) return ""

  // Split by explicit bullets, newlines, or sentence boundaries
  let rawLines = text.split(/[\n\r]+|[•\-\*]\s+/).map((l) => l.trim()).filter(Boolean)
  if (rawLines.length <= 1) {
    rawLines = text.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean)
  }

  const uniquePoints: string[] = []
  const seenNorms = new Set<string>()

  for (const line of rawLines) {
    const cleaned = line.replace(/^[•\-\*\s]+/, "").trim()
    if (!cleaned || cleaned.length < 3) continue
    const thirdPerson = toThirdPersonSummary(cleaned, personName)
    const norm = thirdPerson.toLowerCase().replace(/[^\w\s]/g, "").replace(/\s+/g, " ")
    if (!norm || seenNorms.has(norm)) continue

    let isSub = false
    for (const s of seenNorms) {
      if (s.includes(norm) || norm.includes(s)) {
        isSub = true
        break
      }
    }
    if (isSub) continue

    seenNorms.add(norm)
    uniquePoints.push(thirdPerson)
    if (uniquePoints.length >= maxPoints) break
  }

  if (uniquePoints.length === 0) return ""

  const finalPoints: string[] = []
  let totalWords = 0

  for (const pt of uniquePoints) {
    const words = pt.split(/\s+/).filter(Boolean)
    if (totalWords + words.length > maxWords) {
      const remaining = maxWords - totalWords
      if (remaining >= 3) {
        finalPoints.push(`${words.slice(0, remaining).join(" ")}…`)
      }
      break
    }
    finalPoints.push(pt)
    totalWords += words.length
  }

  const resultPoints = finalPoints.length > 0 ? finalPoints : [uniquePoints[0]]
  return resultPoints.map((p) => `• ${p.replace(/^[•\s]+/, "")}`).join("\n")
}

const FILLER_PHRASES = [
  "hello", "hi there", "hey", "how are you", "can you hear me",
  "test 1 2 3", "testing testing", "good morning", "good evening", "bye bye",
  "thank you", "thanks", "thank you for watching", "thanks for watching",
  "you", "ok", "okay", "yes", "no", "bye", "music", "subtitles", "subscribe"
]

const HALLUCINATION_REGEX = /^(thank you|thanks|thank you for watching|thanks for watching|you|ok|okay|yes|no|bye|goodbye|music|subtitles by|subscribe|like and subscribe)[.!?\s]*$/i

/**
 * Summarizes conversation transcripts locally inside the browser.
 * Zero external API calls to Groq. Always produces third-person factual highlight points.
 */
export async function summarizeTranscriptLocally(
  transcript: string,
  personName?: string
): Promise<string | null> {
  const clean = transcript.trim()
  if (!clean || clean.length < 10) return null

  // Require at least 3 conversational words to avoid summarizing background noise or single-word hallucinations
  const wordCount = clean.split(/\s+/).filter(Boolean).length
  if (wordCount < 3) return null

  // Ignore small talk filler & Whisper silence hallucinations
  if (HALLUCINATION_REGEX.test(clean)) return null

  const lower = clean.toLowerCase().replace(/[^\w\s]/g, "").trim()
  if (FILLER_PHRASES.some((f) => lower === f || (lower.startsWith(f) && lower.length < 25))) {
    return null
  }

  try {
    const pipe = await getLocalSummarizer()
    if (pipe) {
      const result = await pipe(clean, {
        max_new_tokens: 40,
        min_new_tokens: 8,
      })
      if (Array.isArray(result) && result[0]?.summary_text) {
        let summary = result[0].summary_text.trim()
        if (HALLUCINATION_REGEX.test(summary)) return null
        return formatLocalHighlights(summary, personName)
      }
    }
  } catch (err) {
    console.warn("[LocalSummarizer] Local neural inference warning:", err)
  }

  // Synthesize a third-person highlight summary from the action
  return formatLocalHighlights(clean, personName)
}


