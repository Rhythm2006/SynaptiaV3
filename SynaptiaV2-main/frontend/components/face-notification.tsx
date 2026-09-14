"use client"

import { useEffect, useRef, useState } from "react"
import { GlassContainer } from "@/components/glass-container"
import type { BoundingBox } from "@/lib/coordinate-mapper"

interface FaceNotificationProps {
  faceId: string
  name?: string
  description?: string
  relationship?: string
  left: number
  top: number
  box?: BoundingBox
  confidence: number
  autoDismiss?: boolean
  dismissDelay?: number
  isUnidentified?: boolean
  onClose?: () => void
}

// Apple SF Pro system font stack applied inline so it works inside any parent
const SF = `font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Helvetica Neue", Arial, sans-serif; -webkit-font-smoothing: antialiased;`

const THEME = {
  identified:   { accent: "#30d158", glow: "rgba(48,209,88,0.30)",  dim: "rgba(48,209,88,0.12)",  label: "RECOGNIZED"   },
  unidentified: { accent: "#ff9f0a", glow: "rgba(255,159,10,0.30)", dim: "rgba(255,159,10,0.12)", label: "UNKNOWN"       },
  scanning:     { accent: "#0a84ff", glow: "rgba(10,132,255,0.30)", dim: "rgba(10,132,255,0.12)", label: "SCANNING"      },
}

/** Smooth-in animated number (counts up once on mount) */
function ConfidencePct({ value }: { value: number }) {
  const [display, setDisplay] = useState(0)
  const frame = useRef<number>(0)
  useEffect(() => {
    const target = Math.round(value * 100)
    const start = performance.now()
    const dur = 600
    const tick = (now: number) => {
      const p = Math.min((now - start) / dur, 1)
      setDisplay(Math.round(p * target))
      if (p < 1) frame.current = requestAnimationFrame(tick)
    }
    frame.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame.current)
  }, [value])
  return <>{display}%</>
}

/** Parses multiline or bulleted highlight summaries into individual points */
function parseHighlights(text: string): string[] {
  if (!text) return []
  const lines = text.split(/[\n\r]+/).map((l) => l.trim()).filter(Boolean)
  if (lines.length > 1) {
    return lines.map((l) => l.replace(/^[•\-\*\s]+/, "").trim()).filter(Boolean)
  }
  if (text.includes("•")) {
    return text.split("•").map((l) => l.trim()).filter(Boolean)
  }
  return [text.trim()]
}

export function FaceNotification({
  faceId,
  name,
  description,
  relationship,
  left,
  top,
  box,
  confidence,
  autoDismiss = false,
  dismissDelay = 8000,
  isUnidentified = false,
  onClose,
}: FaceNotificationProps) {
  const [dismissed, setDismissed] = useState(false)
  const [entered, setEntered] = useState(false)

  useEffect(() => {
    const t = setTimeout(() => setEntered(true), 30)
    return () => clearTimeout(t)
  }, [])

  useEffect(() => {
    if (autoDismiss && onClose) {
      const t = setTimeout(() => { setDismissed(true); onClose() }, dismissDelay)
      return () => clearTimeout(t)
    }
  }, [autoDismiss, dismissDelay, onClose])

  if (dismissed) return null

  const state = name ? "identified" : isUnidentified ? "unidentified" : "scanning"
  const { accent, glow, dim, label } = THEME[state]

  // ── Reticle corner SVG paths ───────────────────────────────────────────
  const Corner = ({ position }: { position: "tl" | "tr" | "bl" | "br" }) => {
    const size = 18
    const stroke = 1.8
    const r = 5
    const paths: Record<string, string> = {
      tl: `M ${size} ${stroke/2} L ${r} ${stroke/2} Q ${stroke/2} ${stroke/2} ${stroke/2} ${r} L ${stroke/2} ${size}`,
      tr: `M ${stroke/2} ${stroke/2+0} L ${size-r} ${stroke/2} Q ${size-stroke/2} ${stroke/2} ${size-stroke/2} ${r} L ${size-stroke/2} ${size}`,
      bl: `M ${size} ${size-stroke/2} L ${r} ${size-stroke/2} Q ${stroke/2} ${size-stroke/2} ${stroke/2} ${size-r} L ${stroke/2} ${stroke/2}`,
      br: `M ${stroke/2} ${size-stroke/2} L ${size-r} ${size-stroke/2} Q ${size-stroke/2} ${size-stroke/2} ${size-stroke/2} ${size-r} L ${size-stroke/2} ${stroke/2}`,
    }
    const offsets: Record<string, React.CSSProperties> = {
      tl: { top: -2, left: -2 },
      tr: { top: -2, right: -2 },
      bl: { bottom: -2, left: -2 },
      br: { bottom: -2, right: -2 },
    }
    return (
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        style={{ position: "absolute", ...offsets[position] }}
      >
        <path
          d={paths[position]}
          fill="none"
          stroke="rgba(255,255,255,0.9)"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
      </svg>
    )
  }

  return (
    <>
      {/* ── AR Face Reticle ─────────────────────────────────────────────── */}
      {box && (
        <div
          className="absolute z-30 pointer-events-none"
          style={{
            left:   `${box.originX}px`,
            top:    `${box.originY}px`,
            width:  `${box.width}px`,
            height: `${box.height}px`,
            transition: "left 80ms ease-out, top 80ms ease-out, width 80ms ease-out, height 80ms ease-out",
          }}
        >
          {/* Main border — thin, rounded, semi-transparent */}
          <div
            style={{
              position: "absolute", inset: 0,
              borderRadius: 14,
              border: `1px solid ${accent}55`,
              boxShadow: `0 0 0 0.5px rgba(255,255,255,0.08), 0 0 20px ${glow}, inset 0 0 14px ${dim}`,
            }}
          />

          {/* Scan-line shimmer */}
          <div
            style={{
              position: "absolute", inset: 0,
              borderRadius: 14,
              overflow: "hidden",
              background: `linear-gradient(180deg, ${accent}10 0%, transparent 60%)`,
              animation: "pulse 2.4s ease-in-out infinite",
            }}
          />

          {/* Corner brackets */}
          <Corner position="tl" />
          <Corner position="tr" />
          <Corner position="bl" />
          <Corner position="br" />

          {/* Status pill — sits above the box */}
          <div
            style={{
              position: "absolute",
              top: -34,
              left: "50%",
              transform: "translateX(-50%)",
              display: "flex",
              alignItems: "center",
              gap: 6,
              padding: "4px 10px",
              borderRadius: 20,
              background: "rgba(0,0,0,0.62)",
              backdropFilter: "blur(18px)",
              WebkitBackdropFilter: "blur(18px)",
              border: `1px solid ${accent}30`,
              boxShadow: `0 2px 12px rgba(0,0,0,0.5)`,
              whiteSpace: "nowrap",
            }}
          >
            {/* Pulsing dot */}
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: accent,
              boxShadow: `0 0 6px ${accent}`,
              flexShrink: 0,
              animation: "pulse 1.8s ease-in-out infinite",
            }} />
            <span style={{
              ...Object.fromEntries(SF.split(";").filter(Boolean).map(s => {
                const [k, v] = s.split(":"); return [k.trim().replace(/-([a-z])/g, (_,c) => c.toUpperCase()), v?.trim()]
              })),
              fontSize: 10,
              fontWeight: 600,
              letterSpacing: "0.08em",
              color: accent,
              textShadow: `0 0 8px ${glow}`,
            }}>
              {label}
            </span>
            <span style={{
              fontSize: 10,
              fontWeight: 400,
              color: "rgba(255,255,255,0.45)",
              letterSpacing: "0.02em",
              fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
            }}>
              <ConfidencePct value={confidence} />
            </span>
          </div>
        </div>
      )}

      {/* ── Info Card ─────────────────────────────────────────────────────── */}
      <div
        className="absolute z-40 pointer-events-auto"
        style={{
          left: `${left}px`,
          top:  `${top}px`,
          opacity:   entered ? 1 : 0,
          transform: entered ? "translateY(0) scale(1)" : "translateY(6px) scale(0.97)",
          transition: "opacity 380ms cubic-bezier(0.34,1.56,0.64,1), transform 380ms cubic-bezier(0.34,1.56,0.64,1)",
        }}
      >
        {name ? (
          /* ── IDENTIFIED ─────────────────────────────────────────────── */
          <GlassContainer
            variant="prominent"
            tint="cool"
            distortion="subtle"
            hover={false}
            blur={28}
            opacity={0.18}
            specularIntensity={0.35}
            innerGlowColor={glow}
            className="w-[21rem] max-w-[90vw] p-0"
            style={{
              boxShadow: `0 12px 48px rgba(0,0,0,0.55), 0 0 0 0.5px rgba(255,255,255,0.12), 0 0 24px ${glow}`,
              borderRadius: 20,
            }}
          >
            {/* Accent top strip */}
            <div style={{
              height: 3,
              borderRadius: "20px 20px 0 0",
              background: `linear-gradient(90deg, transparent, ${accent}90, transparent)`,
            }} />

            <div style={{ padding: "14px 16px 16px" }}>
              {/* Header */}
              <div style={{ display: "flex", alignItems: "center", gap: 11, marginBottom: description ? 12 : 0 }}>
                {/* Avatar */}
                <div style={{
                  width: 40, height: 40, borderRadius: "50%", flexShrink: 0,
                  background: `radial-gradient(circle at 30% 30%, ${accent}40, ${accent}10)`,
                  border: `1.5px solid ${accent}55`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif",
                  fontSize: 17, fontWeight: 700, color: accent,
                  textShadow: `0 0 10px ${glow}`,
                  boxShadow: `0 0 16px ${dim}`,
                }}>
                  {name.charAt(0).toUpperCase()}
                </div>

                <div style={{ flex: 1, minWidth: 0 }}>
                  <p style={{
                    margin: 0,
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                    fontSize: 10.5, fontWeight: 500,
                    color: accent,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    marginBottom: 2,
                    textShadow: `0 0 8px ${glow}`,
                  }}>
                    Recognized
                  </p>
                  <h4 style={{
                    margin: 0,
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif",
                    fontSize: 20, fontWeight: 700,
                    color: "#fff",
                    letterSpacing: "-0.4px",
                    lineHeight: 1.1,
                    textShadow: "0 1px 8px rgba(0,0,0,0.8)",
                    overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>
                    {name}
                  </h4>
                </div>

                {relationship && (
                  <span style={{
                    flexShrink: 0,
                    padding: "3px 9px",
                    borderRadius: 20,
                    background: dim,
                    border: `1px solid ${accent}35`,
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                    fontSize: 10.5, fontWeight: 600,
                    color: accent,
                    letterSpacing: "0.02em",
                    whiteSpace: "nowrap",
                  }}>
                    {relationship}
                  </span>
                )}
              </div>

              {/* Divider + summary highlights */}
              {description && (
                <>
                  <div style={{
                    height: 1,
                    background: `linear-gradient(to right, transparent, rgba(255,255,255,0.1), transparent)`,
                    marginBottom: 10,
                  }} />
                  <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
                    {parseHighlights(description).map((item, idx) => (
                      <div key={idx} style={{ display: "flex", alignItems: "flex-start", gap: 7 }}>
                        <span style={{
                          width: 5,
                          height: 5,
                          borderRadius: "50%",
                          background: accent,
                          boxShadow: `0 0 6px ${accent}`,
                          marginTop: 6,
                          flexShrink: 0,
                        }} />
                        <p style={{
                          margin: 0,
                          fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                          fontSize: 13,
                          fontWeight: 400,
                          lineHeight: 1.45,
                          color: "rgba(255,255,255,0.85)",
                          textShadow: "0 1px 4px rgba(0,0,0,0.7)",
                          letterSpacing: "-0.1px",
                        }}>
                          {item}
                        </p>
                      </div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </GlassContainer>

        ) : isUnidentified ? (
          /* ── UNIDENTIFIED ─────────────────────────────────────────────── */
          <GlassContainer
            variant="regular"
            tint="warm"
            distortion="subtle"
            hover={false}
            blur={24}
            opacity={0.15}
            innerGlowColor={glow}
            className="w-[18rem] max-w-[88vw] p-0"
            style={{
              boxShadow: `0 10px 40px rgba(0,0,0,0.5), 0 0 0 0.5px rgba(255,255,255,0.10), 0 0 18px ${glow}`,
              borderRadius: 18,
            }}
          >
            <div style={{ height: 3, borderRadius: "18px 18px 0 0", background: `linear-gradient(90deg, transparent, ${accent}80, transparent)` }} />
            <div style={{ padding: "13px 15px 14px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <div style={{
                  width: 36, height: 36, borderRadius: "50%", flexShrink: 0,
                  background: `radial-gradient(circle, ${accent}25, transparent)`,
                  border: `1.5px solid ${accent}45`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontFamily: "-apple-system, sans-serif",
                  fontSize: 16, color: accent,
                }}>?</div>
                <div>
                  <p style={{
                    margin: 0,
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                    fontSize: 10, fontWeight: 500, letterSpacing: "0.07em",
                    textTransform: "uppercase", color: accent, marginBottom: 2,
                    textShadow: `0 0 8px ${glow}`,
                  }}>Unknown Face</p>
                  <h4 style={{
                    margin: 0,
                    fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Display', sans-serif",
                    fontSize: 17, fontWeight: 700,
                    color: "#fff", letterSpacing: "-0.3px",
                    textShadow: "0 1px 6px rgba(0,0,0,0.8)",
                  }}>Not Enrolled</h4>
                </div>
              </div>
              <div style={{ height: 1, background: "linear-gradient(to right, transparent, rgba(255,255,255,0.08), transparent)", marginBottom: 9 }} />
              <p style={{
                margin: 0,
                fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
                fontSize: 12.5, fontWeight: 400, lineHeight: 1.5,
                color: "rgba(255,255,255,0.62)",
                textShadow: "0 1px 4px rgba(0,0,0,0.7)",
              }}>
                Enter a name in the panel, then tap <strong style={{ color: "rgba(255,255,255,0.85)", fontWeight: 600 }}>Enroll</strong> to save and recognize this face.
              </p>
            </div>
          </GlassContainer>

        ) : (
          /* ── SCANNING ─────────────────────────────────────────────────── */
          <div style={{
            display: "inline-flex", alignItems: "center", gap: 8,
            padding: "7px 14px",
            borderRadius: 22,
            background: "rgba(0,0,0,0.58)",
            backdropFilter: "blur(20px)",
            WebkitBackdropFilter: "blur(20px)",
            border: `1px solid ${accent}30`,
            boxShadow: `0 4px 20px rgba(0,0,0,0.4), 0 0 14px ${glow}`,
            fontFamily: "-apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif",
            fontSize: 12.5, fontWeight: 500,
            color: "rgba(255,255,255,0.72)",
            textShadow: "0 1px 4px rgba(0,0,0,0.8)",
          }}>
            <span style={{
              width: 7, height: 7, borderRadius: "50%",
              background: accent,
              boxShadow: `0 0 8px ${accent}`,
              flexShrink: 0,
              animation: "pulse 1.6s ease-in-out infinite",
            }} />
            Face detected · ready to enroll
          </div>
        )}
      </div>
    </>
  )
}
