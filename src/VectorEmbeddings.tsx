import { useState, useMemo, useCallback } from "react";
import {
  Box,
  Typography,
  Paper,
  Divider,
  Button,
  useTheme,
  Stack,
  CircularProgress,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from "@mui/material";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";

// ─── Seeded PRNG (Mulberry32) ─────────────────────────────

function mulberry32(seed: number): () => number {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ─── Vector Utilities ─────────────────────────────────────

const DIMS = 768;
const SHOW_DIMS = 16;

function generateUnitVector(seed: number): number[] {
  const rng = mulberry32(seed);
  const v = Array.from({ length: DIMS }, () => rng() * 2 - 1);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  return v.map((x) => x / norm);
}

function mixAndNormalize(
  components: { dir: number[]; weight: number }[],
  noiseSeed: number,
  noiseScale = 0.03,
): number[] {
  const result = new Array<number>(DIMS).fill(0);
  for (const { dir, weight } of components) {
    for (let i = 0; i < DIMS; i++) result[i] += dir[i]! * weight;
  }
  const rng = mulberry32(noiseSeed);
  for (let i = 0; i < DIMS; i++) result[i] += (rng() * 2 - 1) * noiseScale;
  const norm = Math.sqrt(result.reduce((s, x) => s + x * x, 0));
  return result.map((x) => x / norm);
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
  return dot / (Math.sqrt(nA) * Math.sqrt(nB));
}

// ─── Semantic Directions (deterministic seeds) ────────────

const D_ANIMAL = generateUnitVector(42);
const D_NATURE = generateUnitVector(137);
const D_FINANCE = generateUnitVector(256);
const D_VEHICLE = generateUnitVector(389);
const D_DOMESTIC = generateUnitVector(512);

// ─── Sample Items ─────────────────────────────────────────

type ItemType = "text" | "image";

interface SampleItem {
  id: number;
  type: ItemType;
  label: string;
  short: string;
  content: string;
  embedding: number[];
}

const ITEMS: SampleItem[] = [
  {
    id: 0,
    type: "text",
    label: "The cat sat on the mat",
    short: "Cat T",
    content: "The cat sat on the mat",
    embedding: mixAndNormalize(
      [
        { dir: D_ANIMAL, weight: 1.0 },
        { dir: D_DOMESTIC, weight: 0.7 },
      ],
      1001,
    ),
  },
  {
    id: 1,
    type: "text",
    label: "A dog played in the park",
    short: "Dog T",
    content: "A dog played in the park",
    embedding: mixAndNormalize(
      [
        { dir: D_ANIMAL, weight: 0.9 },
        { dir: D_NATURE, weight: 0.6 },
      ],
      1002,
    ),
  },
  {
    id: 2,
    type: "text",
    label: "The stock market rose sharply",
    short: "Stk T",
    content: "The stock market rose sharply",
    embedding: mixAndNormalize([{ dir: D_FINANCE, weight: 1.0 }], 1003),
  },
  {
    id: 3,
    type: "text",
    label: "She drove her car to work",
    short: "Car T",
    content: "She drove her car to work",
    embedding: mixAndNormalize([{ dir: D_VEHICLE, weight: 1.0 }], 1004),
  },
  {
    id: 4,
    type: "image",
    label: "Cat",
    short: "Cat I",
    content: "cat",
    embedding: mixAndNormalize(
      [
        { dir: D_ANIMAL, weight: 1.0 },
        { dir: D_DOMESTIC, weight: 0.6 },
      ],
      2001,
    ),
  },
  {
    id: 5,
    type: "image",
    label: "Dog",
    short: "Dog I",
    content: "dog",
    embedding: mixAndNormalize(
      [
        { dir: D_ANIMAL, weight: 0.8 },
        { dir: D_NATURE, weight: 0.7 },
      ],
      2002,
    ),
  },
  {
    id: 6,
    type: "image",
    label: "Tree",
    short: "Tree I",
    content: "tree",
    embedding: mixAndNormalize([{ dir: D_NATURE, weight: 1.0 }], 2003),
  },
  {
    id: 7,
    type: "image",
    label: "Car",
    short: "Car I",
    content: "car",
    embedding: mixAndNormalize([{ dir: D_VEHICLE, weight: 1.0 }], 2004),
  },
];

// ─── Inline SVG Images ────────────────────────────────────

function CatSvg({ size = 56 }: { size?: number }) {
  return (
    <svg viewBox="0 0 80 80" width={size} height={size}>
      <polygon points="18,32 26,8 34,32" fill="#FF9800" />
      <polygon points="46,32 54,8 62,32" fill="#FF9800" />
      <polygon points="21,32 26,15 31,32" fill="#FFE0B2" />
      <polygon points="49,32 54,15 59,32" fill="#FFE0B2" />
      <ellipse cx={40} cy={48} rx={26} ry={24} fill="#FFB74D" />
      <ellipse cx={30} cy={42} rx={4} ry={4.5} fill="#FFF" />
      <circle cx={31} cy={42} r={2.5} fill="#333" />
      <ellipse cx={50} cy={42} rx={4} ry={4.5} fill="#FFF" />
      <circle cx={51} cy={42} r={2.5} fill="#333" />
      <polygon points="40,50 37,54 43,54" fill="#E91E63" />
      <path
        d="M37,54 Q40,58 43,54"
        fill="none"
        stroke="#333"
        strokeWidth={1.2}
      />
      <line x1={8} y1={48} x2={28} y2={50} stroke="#333" strokeWidth={1} />
      <line x1={8} y1={54} x2={28} y2={53} stroke="#333" strokeWidth={1} />
      <line x1={52} y1={50} x2={72} y2={48} stroke="#333" strokeWidth={1} />
      <line x1={52} y1={53} x2={72} y2={54} stroke="#333" strokeWidth={1} />
    </svg>
  );
}

function DogSvg({ size = 56 }: { size?: number }) {
  return (
    <svg viewBox="0 0 80 80" width={size} height={size}>
      <ellipse
        cx={16}
        cy={38}
        rx={11}
        ry={18}
        fill="#795548"
        transform="rotate(-10 16 38)"
      />
      <ellipse
        cx={64}
        cy={38}
        rx={11}
        ry={18}
        fill="#795548"
        transform="rotate(10 64 38)"
      />
      <ellipse cx={40} cy={45} rx={24} ry={22} fill="#A1887F" />
      <ellipse cx={31} cy={40} rx={4} ry={4.5} fill="#FFF" />
      <circle cx={32} cy={40} r={2.5} fill="#333" />
      <ellipse cx={49} cy={40} rx={4} ry={4.5} fill="#FFF" />
      <circle cx={50} cy={40} r={2.5} fill="#333" />
      <ellipse cx={40} cy={52} rx={12} ry={8} fill="#D7CCC8" />
      <ellipse cx={40} cy={49} rx={4} ry={3} fill="#333" />
      <path
        d="M36,53 Q40,58 44,53"
        fill="none"
        stroke="#333"
        strokeWidth={1.2}
      />
      <ellipse cx={40} cy={58} rx={4} ry={5} fill="#E91E63" />
    </svg>
  );
}

function TreeSvg({ size = 56 }: { size?: number }) {
  return (
    <svg viewBox="0 0 80 80" width={size} height={size}>
      <rect x={34} y={52} width={12} height={22} rx={2} fill="#795548" />
      <circle cx={40} cy={32} r={20} fill="#43A047" />
      <circle cx={28} cy={40} r={14} fill="#66BB6A" />
      <circle cx={52} cy={40} r={14} fill="#66BB6A" />
      <circle cx={40} cy={22} r={14} fill="#2E7D32" />
      <circle cx={35} cy={26} r={5} fill="#81C784" opacity={0.6} />
    </svg>
  );
}

function CarSvg({ size = 56 }: { size?: number }) {
  return (
    <svg viewBox="0 0 80 80" width={size} height={size}>
      <rect x={6} y={38} width={68} height={18} rx={4} fill="#1976D2" />
      <path d="M20,38 L28,22 L56,22 L64,38" fill="#1565C0" />
      <path
        d="M24,37 L30,24 L41,24 L41,37"
        fill="#BBDEFB"
        opacity={0.8}
      />
      <path
        d="M43,37 L43,24 L54,24 L60,37"
        fill="#BBDEFB"
        opacity={0.8}
      />
      <rect x={67} y={42} width={7} height={5} rx={2} fill="#FFF9C4" />
      <rect x={6} y={42} width={5} height={5} rx={2} fill="#EF5350" />
      <circle cx={22} cy={58} r={8} fill="#424242" />
      <circle cx={22} cy={58} r={4.5} fill="#757575" />
      <circle cx={58} cy={58} r={8} fill="#424242" />
      <circle cx={58} cy={58} r={4.5} fill="#757575" />
      <rect x={8} y={55} width={64} height={2.5} rx={1} fill="#0D47A1" />
    </svg>
  );
}

function ItemImage({
  content,
  size = 56,
}: {
  content: string;
  size?: number;
}) {
  switch (content) {
    case "cat":
      return <CatSvg size={size} />;
    case "dog":
      return <DogSvg size={size} />;
    case "tree":
      return <TreeSvg size={size} />;
    case "car":
      return <CarSvg size={size} />;
    default:
      return null;
  }
}

// ─── Sub-components ───────────────────────────────────────

function DimensionBars({ values }: { values: number[] }) {
  const theme = useTheme();
  const maxAbs = Math.max(...values.map(Math.abs), 0.001);
  const barH = 15;
  const gap = 5;
  const h = values.length * (barH + gap);
  const w = 360;
  const lbl = 28;
  const val = 58;
  const area = w - lbl - val;
  const mid = lbl + area / 2;

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${w} ${h}`}
      style={{ display: "block", marginBottom: 8 }}
    >
      {values.map((v, i) => {
        const y = i * (barH + gap);
        const bw = (Math.abs(v) / maxAbs) * (area / 2 - 4);
        const bx = v >= 0 ? mid : mid - bw;
        const color = v >= 0 ? "#1976D2" : "#F44336";
        return (
          <g key={i}>
            <text
              x={lbl - 3}
              y={y + barH / 2 + 1}
              textAnchor="end"
              dominantBaseline="central"
              fill={theme.palette.text.secondary}
              fontSize={8}
            >
              [{i}]
            </text>
            <rect
              x={bx}
              y={y + 1}
              width={Math.max(1, bw)}
              height={barH - 2}
              rx={2}
              fill={color}
              opacity={0.6}
            />
            <line
              x1={mid}
              y1={y}
              x2={mid}
              y2={y + barH}
              stroke={theme.palette.divider}
              strokeWidth={0.5}
            />
            <text
              x={w}
              y={y + barH / 2 + 1}
              textAnchor="end"
              dominantBaseline="central"
              fill={theme.palette.text.primary}
              fontSize={8}
              fontFamily="monospace"
            >
              {v >= 0 ? "\u00A0" : ""}
              {v.toFixed(4)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ─── Main Component ───────────────────────────────────────

export function VectorEmbeddings() {
  const [embedded, setEmbedded] = useState<Set<number>>(() => new Set());
  const [embedding, setEmbedding] = useState<number | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [compareA, setCompareA] = useState<number | "">("");
  const [compareB, setCompareB] = useState<number | "">("");

  const embeddedItems = useMemo(
    () => ITEMS.filter((item) => embedded.has(item.id)),
    [embedded],
  );

  const compareAId =
    typeof compareA === "number" && embedded.has(compareA)
      ? compareA
      : (embeddedItems[0]?.id ?? null);
  const compareBCandidate =
    typeof compareB === "number" && embedded.has(compareB)
      ? compareB
      : (embeddedItems.find((i) => i.id !== compareAId)?.id ?? null);
  const compareBId =
    compareBCandidate !== null && compareBCandidate === compareAId
      ? (embeddedItems.find((i) => i.id !== compareAId)?.id ?? null)
      : compareBCandidate;
  const compareAItem =
    compareAId !== null ? (ITEMS.find((i) => i.id === compareAId) ?? null) : null;
  const compareBItem =
    compareBId !== null ? (ITEMS.find((i) => i.id === compareBId) ?? null) : null;
  const pairSimilarity =
    compareAItem && compareBItem && compareAItem.id !== compareBItem.id
      ? cosineSim(compareAItem.embedding, compareBItem.embedding)
      : null;

  const handleEmbed = useCallback(
    (id: number) => {
      if (embedded.has(id)) {
        setSelected(id);
        return;
      }
      if (embedding !== null) return;
      setEmbedding(id);
      setSelected(id);
      setTimeout(() => {
        setEmbedded((prev) => new Set([...prev, id]));
        setEmbedding(null);
      }, 600);
    },
    [embedded, embedding],
  );

  const handleEmbedAll = useCallback(() => {
    if (embedding !== null) return;
    setEmbedding(-1);
    setTimeout(() => {
      setEmbedded(new Set(ITEMS.map((i) => i.id)));
      setEmbedding(null);
      setSelected((prev) => prev ?? 0);
    }, 800);
  }, [embedding]);

  const handleReset = useCallback(() => {
    setEmbedded(new Set());
    setEmbedding(null);
    setSelected(null);
    setCompareA("");
    setCompareB("");
  }, []);

  const selectedItem =
    selected !== null ? (ITEMS.find((i) => i.id === selected) ?? null) : null;

  return (
    <Box sx={{ pt: 2, pb: 1, px: 1 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Vector Embeddings
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1.5 }}>
        AI converts text and images into <strong>vectors</strong> — lists of
        numbers that capture meaning. Similar concepts produce similar vectors,
        enabling AI to understand relationships between words, sentences, and
        images.
      </Typography>

      <Box sx={{ display: "flex", gap: 2, alignItems: "stretch" }}>
        {/* ── Left Panel: Sample Items ── */}
        <Paper
          elevation={2}
          sx={{ p: 2, borderRadius: 3, flex: "0 0 210px" }}
        >
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Sample Texts
          </Typography>
          <Divider sx={{ mb: 1 }} />
          <Stack spacing={0.5} sx={{ mb: 1.5 }}>
            {ITEMS.filter((i) => i.type === "text").map((item) => (
              <Box
                key={item.id}
                onClick={() => handleEmbed(item.id)}
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 0.5,
                  p: 0.75,
                  borderRadius: 1.5,
                  cursor: "pointer",
                  border: 2,
                  borderColor:
                    selected === item.id
                      ? "primary.main"
                      : embedded.has(item.id)
                        ? "primary.light"
                        : "divider",
                  bgcolor:
                    selected === item.id ? "action.selected" : "transparent",
                  "&:hover": { bgcolor: "action.hover" },
                }}
              >
                {embedding === item.id ||
                (embedding === -1 && !embedded.has(item.id)) ? (
                  <CircularProgress size={14} sx={{ flexShrink: 0 }} />
                ) : embedded.has(item.id) ? (
                  <CheckCircleRoundedIcon
                    sx={{ fontSize: 16, color: "primary.main", flexShrink: 0 }}
                  />
                ) : (
                  <Box sx={{ width: 16, flexShrink: 0 }} />
                )}
                <Typography
                  variant="caption"
                  sx={{
                    fontStyle: "italic",
                    lineHeight: 1.3,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                  }}
                >
                  &ldquo;{item.content}&rdquo;
                </Typography>
              </Box>
            ))}
          </Stack>

          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Sample Images
          </Typography>
          <Divider sx={{ mb: 1 }} />
          <Box
            sx={{
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 1,
              mb: 1.5,
            }}
          >
            {ITEMS.filter((i) => i.type === "image").map((item) => (
              <Paper
                key={item.id}
                elevation={selected === item.id ? 3 : 0}
                onClick={() => handleEmbed(item.id)}
                sx={{
                  p: 0.75,
                  borderRadius: 2,
                  cursor: "pointer",
                  textAlign: "center",
                  border: 2,
                  borderColor:
                    selected === item.id
                      ? "primary.main"
                      : embedded.has(item.id)
                        ? "primary.light"
                        : "divider",
                  bgcolor:
                    selected === item.id ? "action.selected" : "transparent",
                  "&:hover": { bgcolor: "action.hover" },
                  position: "relative",
                }}
              >
                {(embedding === item.id ||
                  (embedding === -1 && !embedded.has(item.id))) && (
                  <CircularProgress
                    size={16}
                    sx={{ position: "absolute", top: 4, right: 4 }}
                  />
                )}
                {embedded.has(item.id) && embedding !== item.id && (
                  <CheckCircleRoundedIcon
                    sx={{
                      position: "absolute",
                      top: 4,
                      right: 4,
                      fontSize: 16,
                      color: "primary.main",
                    }}
                  />
                )}
                <ItemImage content={item.content} size={48} />
                <Typography variant="caption" display="block">
                  {item.label}
                </Typography>
              </Paper>
            ))}
          </Box>

          <Stack spacing={1}>
            <Button
              variant="contained"
              size="small"
              fullWidth
              onClick={handleEmbedAll}
              disabled={embedded.size === ITEMS.length || embedding !== null}
            >
              Embed All
            </Button>
            <Button
              variant="outlined"
              size="small"
              fullWidth
              startIcon={<RestartAltRoundedIcon />}
              onClick={handleReset}
            >
              Reset
            </Button>
          </Stack>
        </Paper>

        {/* ── Center Panel: Embedding Display ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: 1 }}>
          {selectedItem && embedded.has(selectedItem.id) ? (
            <>
              <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
                Embedding: {selectedItem.label}
                {selectedItem.type === "image" ? " (image)" : ""}
              </Typography>
              {selectedItem.type === "image" ? (
                <Box sx={{ mb: 1.5 }}>
                  <ItemImage content={selectedItem.content} size={64} />
                </Box>
              ) : (
                <Typography
                  variant="body2"
                  color="text.secondary"
                  sx={{ mb: 1.5, fontStyle: "italic" }}
                >
                  &ldquo;{selectedItem.content}&rdquo;
                </Typography>
              )}
              <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
                Vector ({DIMS} dimensions)
              </Typography>
              <DimensionBars
                values={selectedItem.embedding.slice(0, SHOW_DIMS)}
              />
              <Typography
                variant="caption"
                color="text.secondary"
                sx={{ display: "block", mb: 1 }}
              >
                Showing {SHOW_DIMS} of {DIMS} dimensions
              </Typography>
              <Typography
                variant="caption"
                component="pre"
                sx={{
                  fontFamily: "monospace",
                  fontSize: 10,
                  color: "text.secondary",
                  bgcolor: "action.hover",
                  p: 1,
                  borderRadius: 1,
                  overflowX: "auto",
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-all",
                }}
              >
                [
                {selectedItem.embedding
                  .slice(0, SHOW_DIMS)
                  .map((v) => v.toFixed(4))
                  .join(", ")}
                , ...{DIMS - SHOW_DIMS} more]
              </Typography>
            </>
          ) : embedding !== null ? (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                minHeight: 200,
              }}
            >
              <Stack alignItems="center" spacing={1}>
                <CircularProgress />
                <Typography variant="body2" color="text.secondary">
                  Generating embedding...
                </Typography>
              </Stack>
            </Box>
          ) : (
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                minHeight: 200,
              }}
            >
              <Typography
                variant="body2"
                color="text.secondary"
                textAlign="center"
              >
                Click a sample text or image to embed it.
                <br />
                The embedding vector will appear here.
              </Typography>
            </Box>
          )}
        </Paper>

        {/* ── Right Panel: Similarity ── */}
        <Paper
          elevation={2}
          sx={{ p: 2, borderRadius: 3, minWidth: 180 }}
        >
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Similarity
          </Typography>
          <Divider sx={{ mb: 1 }} />
          {embeddedItems.length >= 2 ? (
            <Stack spacing={1.25}>
              <FormControl size="small" fullWidth>
                <InputLabel id="similarity-item-a-label">Item A</InputLabel>
                <Select
                  labelId="similarity-item-a-label"
                  label="Item A"
                  value={compareAId ?? ""}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setCompareA(Number.isNaN(val) ? "" : val);
                  }}
                >
                  {embeddedItems.map((item) => (
                    <MenuItem key={`a-${item.id}`} value={item.id}>
                      {item.short}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl size="small" fullWidth>
                <InputLabel id="similarity-item-b-label">Item B</InputLabel>
                <Select
                  labelId="similarity-item-b-label"
                  label="Item B"
                  value={compareBId ?? ""}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setCompareB(Number.isNaN(val) ? "" : val);
                  }}
                >
                  {embeddedItems
                    .filter((item) => item.id !== compareAId)
                    .map((item) => (
                      <MenuItem key={`b-${item.id}`} value={item.id}>
                        {item.short}
                      </MenuItem>
                    ))}
                </Select>
              </FormControl>

              {pairSimilarity !== null ? (
                <Box
                  sx={{
                    mt: 0.5,
                    p: 1.25,
                    borderRadius: 1.5,
                    bgcolor: "action.hover",
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    {compareAItem?.short} ↔ {compareBItem?.short}
                  </Typography>
                  <Typography variant="h5" fontWeight="bold">
                    {pairSimilarity.toFixed(4)}
                  </Typography>
                </Box>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  Select two different embedded items.
                </Typography>
              )}
            </Stack>
          ) : (
            <Typography variant="body2" color="text.secondary">
              Embed 2+ items to compare their similarity.
            </Typography>
          )}
        </Paper>
      </Box>
    </Box>
  );
}
