import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import {
  Box,
  Typography,
  Paper,
  Divider,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Alert,
  Stack,
  IconButton,
  Tooltip,
  CircularProgress,
  useTheme,
  ToggleButtonGroup,
  ToggleButton,
} from "@mui/material";
import DeleteRoundedIcon from "@mui/icons-material/DeleteRounded";
import AddRoundedIcon from "@mui/icons-material/AddRounded";
import {
  CACHED_NOMIC_EMBEDDINGS,
} from "./cachedNomicEmbeddings";

const MAX_SENTENCES = 5;

const COLORS = ["#26BDC0", "#FF5A00", "#7C4DFF", "#43A047", "#E91E63"];

const DEFAULT_SENTENCES = [
  "The cat sat on the mat",
  "A kitten curled up on the couch",
  "My cat loves to chase yarn",
  "Quantum computers use qubits to perform calculations",
];

const SENTENCE_STORAGE_KEY = "skinnyai.sentences.v1";

function loadStoredSentenceTexts(): string[] | null {
  try {
    const raw = localStorage.getItem(SENTENCE_STORAGE_KEY);
    if (!raw) return null;
    const parsed: unknown = JSON.parse(raw);
    const texts = Array.isArray(parsed) ? parsed : (parsed as { texts?: unknown })?.texts;
    if (!Array.isArray(texts)) return null;
    const clean = texts
      .filter((t): t is string => typeof t === "string")
      .map((t) => t.trim().slice(0, 500))
      .filter((t) => t.length > 0)
      .slice(0, MAX_SENTENCES);
    return clean.length > 0 ? clean : null;
  } catch {
    return null;
  }
}

interface OllamaModel {
  name: string;
  model: string;
  capabilities?: string[] | null;
  details?: {
    parameter_size?: string;
  } | null;
}

function parameterCount(size: string | undefined): number {
  if (!size) return Number.POSITIVE_INFINITY;
  const m = size.trim().match(/^([\d.]+)\s*([KMGT])?/i);
  if (!m) return Number.POSITIVE_INFINITY;
  const multipliers: Record<string, number> = {
    K: 1e3,
    M: 1e6,
    G: 1e9,
    B: 1e9,
    T: 1e12,
  };
  const unit = (m[2] ?? "").toUpperCase();
  return Number.parseFloat(m[1]!) * (multipliers[unit] ?? 1);
}

interface Sentence {
  id: number;
  text: string;
  color: string;
  embedding: number[] | null;
}

// True for nomic embedding models ("nomic-embed-text", "nomic-embed-text:latest", …).
function isNomicModel(name: string): boolean {
  return name.toLowerCase().includes("nomic");
}

// Bundled precomputed vectors apply only to the nomic model.
function cachedEmbeddingFor(text: string, modelName: string): number[] | null {
  if (!isNomicModel(modelName)) return null;
  return CACHED_NOMIC_EMBEDDINGS[text] ?? null;
}

// ─── PCA via Gram matrix (n ≤ 5, d = 768) ────────────────────
// G = Xc·Xcᵀ (n×n). Top-2 eigenpairs of G give PCA scores:
// coord_i = sqrt(λ) · v[i]. Cheap and exact for tiny n.

function powerIteration(matrix: number[][], iterations = 200): { value: number; vector: number[] } {
  const n = matrix.length;
  let v = Array.from({ length: n }, (_, i) => i + 1);
  let norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  v = v.map((x) => x / norm);
  for (let k = 0; k < iterations; k++) {
    const w = matrix.map((row) => row.reduce((s, x, j) => s + x * v[j]!, 0));
    norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
    if (norm < 1e-12) break;
    v = w.map((x) => x / norm);
  }
  const mv = matrix.map((row) => row.reduce((s, x, j) => s + x * v[j]!, 0));
  const value = v.reduce((s, x, i) => s + x * mv[i]!, 0);
  return { value, vector: v };
}

function pca2D(embeddings: number[][]): { x: number; y: number }[] {
  const n = embeddings.length;
  if (n === 0) return [];
  if (n === 1) return [{ x: 0, y: 0 }];
  const d = embeddings[0]!.length;

  const mean = new Array<number>(d).fill(0);
  for (const e of embeddings) for (let j = 0; j < d; j++) mean[j]! += e[j]! / n;
  const xc = embeddings.map((e) => e.map((x, j) => x - mean[j]!));

  const G = xc.map((a) => xc.map((b) => a.reduce((s, x, j) => s + x * b[j]!, 0)));

  const first = powerIteration(G);
  const deflated = G.map((row, i) =>
    row.map((x, j) => x - first.value * first.vector[i]! * first.vector[j]!),
  );
  const second = powerIteration(deflated);

  const s1 = Math.sqrt(Math.max(first.value, 0));
  const s2 = Math.sqrt(Math.max(second.value, 0));
  const pts = xc.map((_, i) => ({
    x: s1 * first.vector[i]!,
    y: s2 * second.vector[i]!,
  }));

  // If second component is degenerate (e.g. n = 2), spread along x only.
  const maxAbs = Math.max(...pts.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)]), 1e-12);
  const scale = 80 / maxAbs;
  return pts.map((p) => ({ x: p.x * scale, y: p.y * scale }));
}

function pca3D(embeddings: number[][]): { x: number; y: number; z: number }[] {
  const n = embeddings.length;
  if (n === 0) return [];
  if (n === 1) return [{ x: 0, y: 0, z: 0 }];
  if (n === 2) {
    const pts2 = pca2D(embeddings);
    return pts2.map((p) => ({ x: p.x, y: p.y, z: 0 }));
  }
  const d = embeddings[0]!.length;

  const mean = new Array<number>(d).fill(0);
  for (const e of embeddings) for (let j = 0; j < d; j++) mean[j]! += e[j]! / n;
  const xc = embeddings.map((e) => e.map((x, j) => x - mean[j]!));

  const G = xc.map((a) => xc.map((b) => a.reduce((s, x, j) => s + x * b[j]!, 0)));

  const first = powerIteration(G);
  const deflated2 = G.map((row, i) =>
    row.map((x, j) => x - first.value * first.vector[i]! * first.vector[j]!),
  );
  const second = powerIteration(deflated2);
  const deflated3 = deflated2.map((row, i) =>
    row.map((x, j) => x - second.value * second.vector[i]! * second.vector[j]!),
  );
  const third = powerIteration(deflated3);

  const s1 = Math.sqrt(Math.max(first.value, 0));
  const s2 = Math.sqrt(Math.max(second.value, 0));
  const s3 = Math.sqrt(Math.max(third.value, 0));
  const pts = xc.map((_, i) => ({
    x: s1 * first.vector[i]!,
    y: s2 * second.vector[i]!,
    z: s3 * third.vector[i]!,
  }));

  const maxAbs = Math.max(
    ...pts.flatMap((p) => [Math.abs(p.x), Math.abs(p.y), Math.abs(p.z)]),
    1e-12,
  );
  const scale = 80 / maxAbs;
  return pts.map((p) => ({ x: p.x * scale, y: p.y * scale, z: p.z * scale }));
}

function cosineSim(a: number[], b: number[]): number {
  let dot = 0,
    nA = 0,
    nB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    nA += a[i]! * a[i]!;
    nB += b[i]! * b[i]!;
  }
  return dot / (Math.sqrt(nA) * Math.sqrt(nB) || 1);
}

function euclideanDist(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i]! - b[i]!;
    sum += d * d;
  }
  return Math.sqrt(sum);
}

// ─── Scatter Chart (hand-rolled SVG) ────────────────────────

function ScatterChart({
  sentences,
  points,
  selectedIds,
  onToggle,
}: {
  sentences: Sentence[];
  points: { x: number; y: number }[];
  selectedIds: number[];
  onToggle: (id: number) => void;
}) {
  const theme = useTheme();
  const grid = [-80, -60, -40, -20, 20, 40, 60, 80];

  return (
    <svg viewBox="-110 -110 220 220" width="100%" style={{ display: "block", maxHeight: 460 }}>
      {grid.map((g) => (
        <g key={g}>
          <line x1={g} y1={-100} x2={g} y2={100} stroke={theme.palette.divider} strokeWidth={0.5} />
          <line x1={-100} y1={g} x2={100} y2={g} stroke={theme.palette.divider} strokeWidth={0.5} />
        </g>
      ))}
      <line x1={-100} y1={0} x2={100} y2={0} stroke={theme.palette.text.disabled} strokeWidth={1} />
      <line x1={0} y1={-100} x2={0} y2={100} stroke={theme.palette.text.disabled} strokeWidth={1} />
      <text x={-102} y={-104} fill={theme.palette.text.secondary} fontSize={7}>PC1 →</text>
      <text x={-102} y={-94} fill={theme.palette.text.secondary} fontSize={7}>PC2 ↓</text>
      {sentences.map((s, i) => {
        const p = points[i] ?? { x: 0, y: 0 };
        const selected = selectedIds.includes(s.id);
        return (
          <g
            key={s.id}
            onClick={() => onToggle(s.id)}
            style={{ cursor: "pointer" }}
          >
            <title>{s.text}</title>
            {selected && (
              <circle
                cx={p.x}
                cy={p.y}
                r={13}
                fill="none"
                stroke={theme.palette.text.primary}
                strokeWidth={1.5}
                strokeDasharray="4 2"
              />
            )}
            <circle
              cx={p.x}
              cy={p.y}
              r={10}
              fill={s.color}
              opacity={selected || selectedIds.length === 0 ? 0.95 : 0.55}
              stroke={theme.palette.background.paper}
              strokeWidth={2}
            />
            <text
              x={p.x}
              y={p.y + 3.5}
              fill="#fff"
              fontSize={9}
              fontWeight={700}
              textAnchor="middle"
              pointerEvents="none"
            >
              {i + 1}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Scatter Chart 3D (pseudo-3D SVG with drag-to-rotate) ──

function ScatterChart3D({
  sentences,
  points,
  selectedIds,
  onToggle,
}: {
  sentences: Sentence[];
  points: { x: number; y: number; z: number }[];
  selectedIds: number[];
  onToggle: (id: number) => void;
}) {
  const theme = useTheme();
  const [rotX, setRotX] = useState(-0.5);
  const [rotY, setRotY] = useState(0.6);
  const dragRef = useRef<{ x: number; y: number } | null>(null);

  const project = useCallback(
    (p: { x: number; y: number; z: number }) => {
      const cosY = Math.cos(rotY);
      const sinY = Math.sin(rotY);
      const cosX = Math.cos(rotX);
      const sinX = Math.sin(rotX);
      const x1 = p.x * cosY + p.z * sinY;
      const z1 = -p.x * sinY + p.z * cosY;
      const y1 = p.y * cosX - z1 * sinX;
      const z2 = p.y * sinX + z1 * cosX;
      const focal = 320;
      const scale = focal / (focal - z2);
      return { sx: x1 * scale, sy: y1 * scale, depth: z2, scale };
    },
    [rotX, rotY],
  );

  const projected = useMemo(
    () =>
      sentences.map((s, i) => {
        const raw = points[i] ?? { x: 0, y: 0, z: 0 };
        const pr = project(raw);
        return { s, i, raw, ...pr };
      }),
    [sentences, points, project],
  );
  const sorted = useMemo(
    () => [...projected].sort((a, b) => a.depth - b.depth),
    [projected],
  );

  const line3 = (
    a: { x: number; y: number; z: number },
    b: { x: number; y: number; z: number },
  ) => {
    const pa = project(a);
    const pb = project(b);
    return { x1: pa.sx, y1: pa.sy, x2: pb.sx, y2: pb.sy };
  };
  const axis = (end: { x: number; y: number; z: number }) =>
    line3({ x: 0, y: 0, z: 0 }, end);
  const axX = axis({ x: 100, y: 0, z: 0 });
  const axY = axis({ x: 0, y: 100, z: 0 });
  const axZ = axis({ x: 0, y: 0, z: 100 });
  const axXLabel = project({ x: 108, y: 0, z: 0 });
  const axYLabel = project({ x: 0, y: 108, z: 0 });
  const axZLabel = project({ x: 0, y: 0, z: 108 });

  // Eye references: bounding cube (±100) + floor grid on the bottom
  // face (y = +B, at the far end of the PC2 axis).
  const B = 100;
  const corners = useMemo(
    () =>
      [
        { x: -B, y: -B, z: -B },
        { x: B, y: -B, z: -B },
        { x: B, y: B, z: -B },
        { x: -B, y: B, z: -B },
        { x: -B, y: -B, z: B },
        { x: B, y: -B, z: B },
        { x: B, y: B, z: B },
        { x: -B, y: B, z: B },
      ] as { x: number; y: number; z: number }[],
    [],
  );
  const cubeEdges: [number, number][] = [
    [0, 1], [1, 2], [2, 3], [3, 0], // z = -B square
    [4, 5], [5, 6], [6, 7], [7, 4], // z = +B square
    [0, 4], [1, 5], [2, 6], [3, 7], // pillars (edges 10, 11 meet the bottom face)
  ];
  const gridSteps = [-60, -20, 20, 60];
  const isDark = theme.palette.mode === "dark";

  return (
    <Box>
      <svg
        viewBox="-110 -110 220 220"
        width="100%"
        style={{ display: "block", maxHeight: 460, touchAction: "none", cursor: "grab" }}
        onPointerDown={(e) => {
          (e.target as Element).setPointerCapture?.(e.pointerId);
          dragRef.current = { x: e.clientX, y: e.clientY };
        }}
        onPointerMove={(e) => {
          const last = dragRef.current;
          if (!last) return;
          const dx = e.clientX - last.x;
          const dy = e.clientY - last.y;
          dragRef.current = { x: e.clientX, y: e.clientY };
          setRotY((r) => r + dx * 0.01);
          setRotX((r) => Math.max(-1.3, Math.min(1.3, r + dy * 0.01)));
        }}
        onPointerUp={() => {
          dragRef.current = null;
        }}
        onPointerLeave={() => {
          dragRef.current = null;
        }}
        onDoubleClick={() => {
          setRotX(-0.5);
          setRotY(0.6);
        }}
      >
        <defs>
          <radialGradient id="map3d-bg" cx="50%" cy="42%" r="75%">
            <stop
              offset="0%"
              stopColor={isDark ? "#26BDC0" : "#26BDC0"}
              stopOpacity={isDark ? 0.14 : 0.1}
            />
            <stop
              offset="100%"
              stopColor={isDark ? "#000000" : "#ffffff"}
              stopOpacity={0}
            />
          </radialGradient>
        </defs>
        <rect x={-110} y={-110} width={220} height={220} rx={10} fill="url(#map3d-bg)" />
        {/* floor grid on the bottom face (y = +B) */}
        {gridSteps.map((g) => {
          const h = line3({ x: g, y: B, z: -B }, { x: g, y: B, z: B });
          const v = line3({ x: -B, y: B, z: g }, { x: B, y: B, z: g });
          return (
            <g key={g}>
              <line {...h} stroke={theme.palette.divider} strokeWidth={0.5} opacity={0.9} />
              <line {...v} stroke={theme.palette.divider} strokeWidth={0.5} opacity={0.9} />
            </g>
          );
        })}
        {/* bounding cube (bottom face y = +B drawn stronger) */}
        {cubeEdges.map(([a, b], k) => {
          const isBottom = k === 2 || k === 6 || k === 10 || k === 11;
          return (
            <line
              key={k}
              {...line3(corners[a]!, corners[b]!)}
              stroke={theme.palette.divider}
              strokeWidth={isBottom ? 1 : 0.6}
              opacity={isBottom ? 1 : 0.7}
            />
          );
        })}
        <line {...axX} stroke="#D9A400" strokeWidth={1.5} />
        <line {...axY} stroke={theme.palette.text.primary} strokeWidth={1.5} />
        <line {...axZ} stroke="#90A4AE" strokeWidth={1.5} />
        <text x={axXLabel.sx - 8} y={axXLabel.sy + 10} fill="#D9A400" fontSize={7} fontWeight={700}>
          PC1 →
        </text>
        <text x={axYLabel.sx + 4} y={axYLabel.sy} fill={theme.palette.text.primary} fontSize={7} fontWeight={700}>
          PC2 ↓
        </text>
        <text x={axZLabel.sx + 3} y={axZLabel.sy} fill="#90A4AE" fontSize={7} fontWeight={700}>
          PC3
        </text>
        {(() => {
          const o = project({ x: 0, y: 0, z: 0 });
          return (
            <circle
              cx={o.sx}
              cy={o.sy}
              r={3}
              fill="none"
              stroke={theme.palette.text.secondary}
              strokeWidth={1.25}
              opacity={0.9}
            />
          );
        })()}
        {sorted.map(({ s, i, sx, sy, scale }) => {
          const selected = selectedIds.includes(s.id);
          const r = Math.max(6, Math.min(13, 10 * scale));
          const dimmed = selectedIds.length > 0 && !selected;
          const o = project({ x: 0, y: 0, z: 0 });
          return (
            <g key={s.id} onClick={() => onToggle(s.id)} style={{ cursor: "pointer" }}>
              <title>{s.text}</title>
              {/* spoke to the origin: anchors each point in 3D space */}
              <line
                x1={o.sx}
                y1={o.sy}
                x2={sx}
                y2={sy}
                stroke={s.color}
                strokeWidth={1.25}
                strokeDasharray="3 2"
                opacity={dimmed ? 0.35 : 0.7}
              />
              {selected && (
                <circle
                  cx={sx}
                  cy={sy}
                  r={r + 3}
                  fill="none"
                  stroke={theme.palette.text.primary}
                  strokeWidth={1.5}
                  strokeDasharray="4 2"
                />
              )}
              <circle
                cx={sx}
                cy={sy}
                r={r}
                fill={s.color}
                opacity={dimmed ? 0.55 : 0.95}
                stroke={theme.palette.background.paper}
                strokeWidth={2}
              />
              <text
                x={sx}
                y={sy + 3.5}
                fill="#fff"
                fontSize={9}
                fontWeight={700}
                textAnchor="middle"
                pointerEvents="none"
              >
                {i + 1}
              </text>
            </g>
          );
        })}
      </svg>
      <Typography variant="caption" color="text.secondary" sx={{ display: "block", textAlign: "center" }}>
        Drag to rotate · double-click to reset
      </Typography>
    </Box>
  );
}

// ─── Main Component ─────────────────────────────────────────

export function VectorEmbeddings() {
  const [initialTexts] = useState<string[]>(
    () => loadStoredSentenceTexts() ?? DEFAULT_SENTENCES,
  );
  const nextId = useRef(initialTexts.length);
  const [sentences, setSentences] = useState<Sentence[]>(() =>
    initialTexts.map((text, i) => ({
      id: i,
      text,
      color: COLORS[i % COLORS.length]!,
      // Optimistic: bundled nomic vectors render instantly; corrected once
      // the model list loads if nomic isn't the resolved model.
      embedding: CACHED_NOMIC_EMBEDDINGS[text] ?? null,
    })),
  );
  const [input, setInput] = useState("");
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [model, setModel] = useState("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [embedError, setEmbedError] = useState<string | null>(null);
  const [isEmbedding, setIsEmbedding] = useState(false);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");

  // Persist sentence texts across tab switches (keep-alive) and browser
  // reloads. Vectors are NOT stored: they are re-resolved via the bundled
  // nomic cache / Ollama embed flow so they never go stale per model.
  // Also syncs texts to the server so the get_embedding_sentences agent
  // tool always sees the current tab contents.
  useEffect(() => {
    const texts = sentences.map((s) => s.text);
    try {
      localStorage.setItem(SENTENCE_STORAGE_KEY, JSON.stringify(texts));
    } catch {
      // Storage full or unavailable — sentences still persist for tab switches.
    }
    fetch("/api/embeddings/sentences", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentences: texts }),
    }).catch(() => {
      // Server unreachable — agent tool falls back to defaults.
    });
  }, [sentences]);

  // Load embedding-capable models, default to the smallest one.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/ollama/tags");
        const data = await res.json();
        if (!res.ok) throw new Error(data?.error ?? "Failed to list models.");
        const all: OllamaModel[] = data.models ?? [];
        const capable = all.filter((m) => (m.capabilities ?? []).includes("embedding"));
        const pool = capable.length > 0 ? capable : all;
        // Prefer nomic when available, otherwise fall back to the smallest model.
        const nomic = pool.find((m) => isNomicModel(m.name));
        const smallest = [...pool].sort(
          (a, b) =>
            parameterCount(a.details?.parameter_size) -
            parameterCount(b.details?.parameter_size),
        )[0];
        const resolved = nomic ?? smallest;
        if (!cancelled) {
          setModels(pool);
          setModel(resolved?.name ?? "");
          if (capable.length === 0 && all.length > 0) {
            setModelsError("No embedding-capable model found. Try `ollama pull nomic-embed-text`.");
          }
          // Drop the optimistic bundled vectors if the resolved model isn't nomic.
          if (resolved && !isNomicModel(resolved.name)) {
            setSentences((prev) => prev.map((s) => ({ ...s, embedding: null })));
          }
        }
      } catch (e) {
        if (!cancelled)
          setModelsError(e instanceof Error ? e.message : "Could not reach Ollama.");
      } finally {
        if (!cancelled) setModelsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Embed any sentences missing vectors (debounced, single batched request).
  useEffect(() => {
    if (!model) return;
    const missing = sentences.filter((s) => s.embedding === null);
    if (missing.length === 0) return;
    const timer = setTimeout(async () => {
      setIsEmbedding(true);
      setEmbedError(null);
      try {
        const res = await fetch("/api/ollama/embed", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, inputs: missing.map((s) => s.text) }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data?.error ?? "Embedding failed.");
        const vectors: number[][] = data.embeddings ?? [];
        setSentences((prev) =>
          prev.map((s) => {
            const idx = missing.findIndex((m) => m.id === s.id);
            return idx >= 0 && vectors[idx] ? { ...s, embedding: vectors[idx]! } : s;
          }),
        );
      } catch (e) {
        setEmbedError(e instanceof Error ? e.message : "Embedding failed.");
      } finally {
        setIsEmbedding(false);
      }
    }, 500);
    return () => clearTimeout(timer);
  }, [model, sentences]);

  const handleAdd = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    setSentences((prev) => {
      if (prev.length >= MAX_SENTENCES) return prev;
      const used = new Set(prev.map((s) => s.color));
      const color = COLORS.find((c) => !used.has(c)) ?? COLORS[prev.length % COLORS.length]!;
      return [...prev, { id: nextId.current++, text, color, embedding: cachedEmbeddingFor(text, model) }];
    });
    setInput("");
  }, [input, model]);

  const handleRemove = useCallback((id: number) => {
    setSentences((prev) => prev.filter((s) => s.id !== id));
    setSelectedIds((prev) => prev.filter((sid) => sid !== id));
  }, []);

  const handleTogglePoint = useCallback((id: number) => {
    setSelectedIds((prev) =>
      prev.includes(id) ? prev.filter((sid) => sid !== id) : [...prev, id].slice(-2),
    );
  }, []);

  const embedded = useMemo(() => sentences.filter((s) => s.embedding !== null), [sentences]);

  const points = useMemo(
    () => pca2D(embedded.map((s) => s.embedding!)),
    [embedded],
  );

  const points3D = useMemo(
    () => pca3D(embedded.map((s) => s.embedding!)),
    [embedded],
  );

  const similarities = useMemo(() => {
    const pair = selectedIds
      .map((id) => embedded.find((s) => s.id === id) ?? null)
      .filter((s): s is Sentence => s !== null && s.embedding !== null);
    if (pair.length !== 2) return null;
    const [a, b] = pair as [Sentence, Sentence];
    return {
      a,
      b,
      cosine: cosineSim(a.embedding!, b.embedding!),
      euclidean: euclideanDist(a.embedding!, b.embedding!),
    };
  }, [embedded, selectedIds]);

  const error = modelsError ?? embedError;
  const dimCount = embedded[0]?.embedding?.length ?? 0;

  return (
    <Box sx={{ pt: 2, pb: 1, px: 1 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Vector Embeddings
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1.5 }}>
        AI converts text into <strong>vectors</strong> — lists of numbers that
        capture meaning. Below, real embeddings from Ollama are projected down
        to 2D. Semantically similar sentences land close together.
      </Typography>

      {error && (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          onClose={() => {
            setModelsError(null);
            setEmbedError(null);
          }}
        >
          {error}{" "}
          {error.includes("11434") && "Is `ollama serve` running?"}
        </Alert>
      )}

      <Box sx={{ display: "flex", gap: 2, alignItems: "stretch", flexWrap: "wrap" }}>
        {/* ── Left: sentence list ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "1 1 280px" }}>
          <FormControl size="small" fullWidth sx={{ mb: 1.5 }}>
            <InputLabel id="embed-model-label">Embedding model</InputLabel>
            <Select
              labelId="embed-model-label"
              label="Embedding model"
              value={model}
              onChange={(e) => {
                const next = e.target.value;
                setModel(next);
                // Regenerate vectors for the new model, reusing bundled
                // vectors when switching (back) to nomic.
                setSentences((prev) =>
                  prev.map((s) => ({ ...s, embedding: cachedEmbeddingFor(s.text, next) })),
                );
              }}
              disabled={modelsLoading || models.length === 0}
            >
              {models.map((m) => (
                <MenuItem key={m.name} value={m.name}>
                  {m.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          {modelsLoading && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Loading models from Ollama…
              </Typography>
            </Box>
          )}
          {dimCount > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
              {dimCount}-dim vectors → {viewMode === "2d" ? "2D" : "3D"} via PCA
            </Typography>
          )}
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Sentences ({sentences.length}/{MAX_SENTENCES})
          </Typography>
          <Divider sx={{ mb: 1.5 }} />
          <Stack spacing={1} sx={{ mb: 1.5 }}>
            {sentences.map((s, i) => (
              <Paper
                key={s.id}
                variant="outlined"
                sx={{ p: 1, borderRadius: 2, border: 2, borderColor: s.color }}
              >
                <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                  <Box
                    sx={{
                      width: 22,
                      height: 22,
                      borderRadius: "50%",
                      bgcolor: s.color,
                      color: "#fff",
                      fontSize: 12,
                      fontWeight: 700,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                    }}
                  >
                    {i + 1}
                  </Box>
                  <Box sx={{ flex: 1, minWidth: 0 }}>
                    <Typography variant="body2" sx={{ lineHeight: 1.4 }}>
                      {s.text}
                    </Typography>
                    {s.embedding === null && (
                      <Typography variant="caption" color="text.secondary">
                        embedding…
                      </Typography>
                    )}
                  </Box>
                  <Tooltip title="Remove">
                    <span>
                      <IconButton
                        size="small"
                        onClick={() => handleRemove(s.id)}
                        disabled={sentences.length <= 1}
                      >
                        <DeleteRoundedIcon fontSize="small" />
                      </IconButton>
                    </span>
                  </Tooltip>
                </Box>
              </Paper>
            ))}
          </Stack>
          <Box sx={{ display: "flex", gap: 1 }}>
            <TextField
              label={sentences.length >= MAX_SENTENCES ? `Max ${MAX_SENTENCES} sentences` : "Add a sentence"}
              size="small"
              fullWidth
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleAdd();
              }}
              disabled={sentences.length >= MAX_SENTENCES}
            />
            <Button
              variant="contained"
              onClick={handleAdd}
              disabled={sentences.length >= MAX_SENTENCES || input.trim() === ""}
              sx={{ flexShrink: 0 }}
            >
              <AddRoundedIcon />
            </Button>
          </Box>
        </Paper>

        {/* ── Right: chart ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "2 1 340px" }}>
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 1 }}>
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom sx={{ mb: 0 }}>
              Semantic Map
            </Typography>
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              size="small"
              onChange={(_, v) => {
                if (v === "2d" || v === "3d") setViewMode(v);
              }}
              aria-label="Map dimension"
            >
              <ToggleButton value="2d" aria-label="2D view">
                2D
              </ToggleButton>
              <ToggleButton value="3d" aria-label="3D view">
                3D
              </ToggleButton>
            </ToggleButtonGroup>
          </Box>
          <Divider sx={{ mt: 1, mb: 1.5 }} />
          {isEmbedding && embedded.length === 0 ? (
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: 300 }}>
              <Stack alignItems="center" spacing={1}>
                <CircularProgress />
                <Typography variant="body2" color="text.secondary">
                  Generating embeddings with {model || "Ollama"}…
                </Typography>
              </Stack>
            </Box>
          ) : embedded.length === 0 ? (
            <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: 300 }}>
              <Typography variant="body2" color="text.secondary" textAlign="center">
                Add a sentence to see it on the map.
              </Typography>
            </Box>
          ) : (
            <>
              {viewMode === "2d" ? (
                <ScatterChart
                  sentences={embedded}
                  points={points}
                  selectedIds={selectedIds}
                  onToggle={handleTogglePoint}
                />
              ) : (
                <ScatterChart3D
                  sentences={embedded}
                  points={points3D}
                  selectedIds={selectedIds}
                  onToggle={handleTogglePoint}
                />
              )}
              <Divider sx={{ my: 1.5 }} />
              {similarities ? (
                <Box sx={{ p: 1.25, borderRadius: 1.5, bgcolor: "action.hover" }}>
                  <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
                    <Typography variant="subtitle2" fontWeight="bold">
                      <Box component="span" sx={{ color: similarities.a.color }}>●</Box>{" "}
                      #{embedded.findIndex((s) => s.id === similarities.a.id) + 1}
                      {"  ↔  "}
                      <Box component="span" sx={{ color: similarities.b.color }}>●</Box>{" "}
                      #{embedded.findIndex((s) => s.id === similarities.b.id) + 1}
                    </Typography>
                    <Button size="small" onClick={() => setSelectedIds([])}>
                      Clear
                    </Button>
                  </Box>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
                    &ldquo;{similarities.a.text}&rdquo; vs &ldquo;{similarities.b.text}&rdquo;
                  </Typography>
                  <Typography variant="body2">
                    Cosine similarity: <strong>{similarities.cosine.toFixed(4)}</strong>
                    {"  ·  "}
                    Euclidean distance: <strong>{similarities.euclidean.toFixed(4)}</strong>
                  </Typography>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary" textAlign="center">
                  {selectedIds.length === 1
                    ? "One point selected — click a second point to compare."
                    : "Click two points on the map to compare their similarity."}
                </Typography>
              )}
              <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 1 }}>
                Positions are relative — PCA recomputes each time the set changes, so
                points shift when you add or remove sentences. Watch the cat
                sentences cluster apart from the quantum one.
                {isEmbedding && " Updating…"}
              </Typography>
            </>
          )}
        </Paper>
      </Box>
    </Box>
  );
}
