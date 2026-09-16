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

// ─── Main Component ─────────────────────────────────────────

export function VectorEmbeddings() {
  const nextId = useRef(DEFAULT_SENTENCES.length);
  const [sentences, setSentences] = useState<Sentence[]>(() =>
    DEFAULT_SENTENCES.map((text, i) => ({
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
              {dimCount}-dim vectors → 2D via PCA
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

        {/* ── Right: 2D chart ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "2 1 340px" }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Semantic map (2D projection)
          </Typography>
          <Divider sx={{ mb: 1.5 }} />
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
              <ScatterChart
                sentences={embedded}
                points={points}
                selectedIds={selectedIds}
                onToggle={handleTogglePoint}
              />
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
