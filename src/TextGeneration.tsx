import { useState, useEffect, useCallback, useRef } from "react";
import {
  Box,
  Typography,
  Paper,
  Divider,
  Button,
  TextField,
  Slider,
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
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import RefreshRoundedIcon from "@mui/icons-material/RefreshRounded";
import CasinoRoundedIcon from "@mui/icons-material/CasinoRounded";
import ClearRoundedIcon from "@mui/icons-material/ClearRounded";
import { useOllamaOffline } from "./useOllamaStatus";
import { OFFLINE_TEXT_PRESETS } from "./offlineFallbacks";

// ─── Types ──────────────────────────────────────────────────

interface Candidate {
  token: string;
  logprob: number;
  prob: number;
}

interface Step {
  promptSnapshot: string;
  chosenToken: string;
  candidates: Candidate[];
  // Control values used for this step, so the adjusted view stays correct
  // if the sliders move afterwards.
  temperature: number;
  topK: number;
}

interface DisplayCandidate extends Candidate {
  displayProb: number;
  excluded: boolean;
}

// Number of candidates shown in the histogram (Ollama max is 20).
const NUM_CANDIDATES = 10;

// Make whitespace in tokens visible: " hello" -> "␣hello", "\n" -> "⏎".
// The vocabulary also contains special control tokens (usually end-of-turn)
// whose text decodes to nothing at all — Ollama reports those as "".
function formatToken(token: string): string {
  if (token === "") return "⟨control⟩";
  return token
    .replaceAll(" ", "␣")
    .replaceAll("\n", "⏎")
    .replaceAll("\t", "⇥");
}

// Derive display probabilities from the raw logprobs.
//
// Ollama returns pre-temperature logprobs l_i = logit_i − logsumexp(logits).
// We reconstruct the distribution the sampler actually draws from:
// softmax(logits / T) restricted to the top-K shortlist. Since
// logit_i = l_i + LSE and the LSE/T constant cancels when renormalizing,
// this is exactly proportional to exp(l_i / T) over the kept candidates.
// T = 0 is greedy: 100% to the argmax.
function computeDisplay(
  candidates: Candidate[],
  temperature: number,
  topK: number,
): DisplayCandidate[] {
  const sorted = [...candidates].sort((a, b) => b.prob - a.prob);
  if (temperature <= 0) {
    return sorted.map((c, i) => ({
      ...c,
      displayProb: i === 0 ? 1 : 0,
      excluded: false,
    }));
  }
  // top_k <= 0 disables top-k filtering (full shortlist is eligible).
  const filtering = topK > 0;
  const weights = sorted.map((c, i) =>
    !filtering || i < topK ? Math.exp(c.logprob / temperature) : 0,
  );
  const sum = weights.reduce((a, b) => a + b, 0) || 1;
  return sorted.map((c, i) => ({
    ...c,
    displayProb: weights[i]! / sum,
    excluded: filtering && i >= topK,
  }));
}

interface OllamaModel {
  name: string;
  model: string;
  capabilities?: string[] | null;
  details?: {
    parameter_size?: string;
  } | null;
}

// Parse sizes like "3.2B", "137M", "396.3B" into a comparable number.
// Missing/unparseable sizes sort last.
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

// ─── Histogram ──────────────────────────────────────────────

function Histogram({ step }: { step: Step }) {
  const theme = useTheme();
  const rows = computeDisplay(
    step.candidates,
    step.temperature,
    step.topK,
  );
  const maxProb = rows[0]?.displayProb ?? 1;

  if (rows.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary">
        No candidates returned.
      </Typography>
    );
  }

  return (
    <Stack spacing={1}>
      {rows.map((c, i) => {
        const isChosen = c.token === step.chosenToken;
        return (
          <Box key={`${c.token}-${i}`} sx={{ opacity: c.excluded ? 0.45 : 1 }}>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                mb: 0.25,
              }}
            >
              <Tooltip
                title={
                  c.token === ""
                    ? "Special control token (usually end-of-turn): a real vocabulary entry that renders as no text."
                    : `logprob: ${c.logprob.toFixed(4)} · raw prob: ${(c.prob * 100).toFixed(2)}%${c.excluded ? " · excluded by top_k" : ""}`
                }
              >
                <Typography
                  variant="body2"
                  fontFamily="monospace"
                  fontWeight={isChosen ? "bold" : "normal"}
                  sx={{
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    maxWidth: "60%",
                  }}
                >
                  {isChosen ? "★ " : ""}
                  &ldquo;{formatToken(c.token)}&rdquo;
                  {c.excluded ? " (cut)" : ""}
                </Typography>
              </Tooltip>
              <Typography variant="body2" color="text.secondary">
                {(c.displayProb * 100).toFixed(2)}%
              </Typography>
            </Box>
            <Box
              sx={{
                height: 18,
                borderRadius: 1,
                bgcolor: theme.palette.action.hover,
                overflow: "hidden",
              }}
            >
              <Box
                sx={{
                  height: "100%",
                  width: `${Math.max(1.5, (c.displayProb / maxProb) * 100)}%`,
                  borderRadius: 1,
                  bgcolor: isChosen
                    ? theme.palette.primary.main
                    : theme.palette.text.disabled,
                  transition: "width 0.3s",
                }}
              />
            </Box>
          </Box>
        );
      })}
    </Stack>
  );
}

// ─── Main Component ─────────────────────────────────────────

export function TextGeneration() {
  const [text, setText] = useState("The cat sat on the");
  const [last, setLast] = useState<Step | null>(null);

  const [models, setModels] = useState<OllamaModel[]>([]);
  const [model, setModel] = useState("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(10);
  const [seedText, setSeedText] = useState("");

  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Offline demo mode: Ollama unreachable → fixed presets, no free typing.
  const isOffline = useOllamaOffline(modelsError);
  const [offlinePresetId, setOfflinePresetId] = useState(
    OFFLINE_TEXT_PRESETS[0]!.id,
  );
  const offlineStepRef = useRef(0);
  const activePreset =
    OFFLINE_TEXT_PRESETS.find((p) => p.id === offlinePresetId) ??
    OFFLINE_TEXT_PRESETS[0]!;

  // When offline mode engages or the preset changes, lock the text box to
  // the fixed preset prompt.
  useEffect(() => {
    if (!isOffline) return;
    setText(activePreset.prompt);
    setLast(null);
    offlineStepRef.current = 0;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOffline, offlinePresetId]);

  // Load model list, default to the smallest model capable of completion.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/ollama/tags");
        const data = await res.json();
        if (!res.ok) throw new Error(data?.error ?? "Failed to list models.");
        const all: OllamaModel[] = data.models ?? [];
        const capable = all.filter((m) =>
          (m.capabilities ?? []).includes("completion"),
        );
        const pool = capable.length > 0 ? capable : all;
        const smallest = [...pool].sort(
          (a, b) =>
            parameterCount(a.details?.parameter_size) -
            parameterCount(b.details?.parameter_size),
        )[0];
        if (!cancelled) {
          setModels(pool);
          setModel(smallest?.name ?? "");
        }
      } catch (e) {
        if (!cancelled)
          setModelsError(
            e instanceof Error ? e.message : "Could not reach Ollama.",
          );
      } finally {
        if (!cancelled) setModelsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const parseSeed = useCallback((): number | undefined => {
    const t = seedText.trim();
    if (t === "") return undefined;
    const n = Number.parseInt(t, 10);
    return Number.isNaN(n) ? undefined : n;
  }, [seedText]);

  const requestOneToken = useCallback(
    async (prompt: string) => {
      const seed = parseSeed();
      const res = await fetch("/api/ollama/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          prompt,
          temperature,
          top_k: topK,
          topLogprobs: NUM_CANDIDATES,
          ...(seed !== undefined ? { seed } : {}),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error ?? "Generation failed.");
      return data as { response: string; candidates: Candidate[] };
    },
    [model, temperature, topK, parseSeed],
  );

  // Run model: generate the next token and append it to the text box.
  // Offline: play back the next canned variant for the active preset.
  const handleRun = useCallback(async () => {
    if (text === "" || isRunning) return;
    if (isOffline) {
      const variants = activePreset.variants;
      const variant = variants[offlineStepRef.current % variants.length]!;
      offlineStepRef.current += 1;
      setText(activePreset.prompt + variant.response);
      setLast({
        promptSnapshot: activePreset.prompt,
        chosenToken: variant.response,
        candidates: variant.candidates,
        temperature,
        topK,
      });
      return;
    }
    if (!model) return;
    setIsRunning(true);
    setError(null);
    const runTemp = temperature;
    const runTopK = topK;
    try {
      const data = await requestOneToken(text);
      setText(text + data.response);
      setLast({
        promptSnapshot: text,
        chosenToken: data.response,
        candidates: data.candidates,
        temperature: runTemp,
        topK: runTopK,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed.");
    } finally {
      setIsRunning(false);
    }
  }, [model, text, isRunning, requestOneToken, temperature, topK, isOffline, activePreset]);

  // Rerun: re-sample the same prompt, replacing the last generated token.
  // Offline: cycle to the next canned variant for the same prompt.
  const handleRerun = useCallback(async () => {
    if (last === null || isRunning) return;
    if (isOffline) {
      const variants = activePreset.variants;
      const variant = variants[offlineStepRef.current % variants.length]!;
      offlineStepRef.current += 1;
      setText(last.promptSnapshot + variant.response);
      setLast({
        promptSnapshot: last.promptSnapshot,
        chosenToken: variant.response,
        candidates: variant.candidates,
        temperature,
        topK,
      });
      return;
    }
    setIsRunning(true);
    setError(null);
    const runTemp = temperature;
    const runTopK = topK;
    try {
      const data = await requestOneToken(last.promptSnapshot);
      setText(last.promptSnapshot + data.response);
      setLast({
        promptSnapshot: last.promptSnapshot,
        chosenToken: data.response,
        candidates: data.candidates,
        temperature: runTemp,
        topK: runTopK,
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed.");
    } finally {
      setIsRunning(false);
    }
  }, [last, isRunning, requestOneToken, temperature, topK, isOffline, activePreset]);

  const randomizeSeed = useCallback(() => {
    setSeedText(String(Math.floor(Math.random() * 1_000_000)));
  }, []);

  const canRun = (isOffline || model !== "") && text !== "" && !isRunning;

  return (
    <Box sx={{ pt: 2, pb: 1, px: 1 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        Next-Token Prediction
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1.5 }}>
        LLMs generate text <strong>one token at a time</strong> — each step, the
        model scores every possible next token, turns those scores into
        probabilities, and samples one. Adjust the controls to see how
        temperature and top-K reshape the odds before each pick.
      </Typography>

      {!isOffline && (modelsError || error) && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => {
          setModelsError(null);
          setError(null);
        }}>
          {modelsError ?? error}{" "}
          {((modelsError ?? error) ?? "").includes("11434") &&
            "Is `ollama serve` running?"}
        </Alert>
      )}
      {isOffline && error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* ── Input + Run controls ── */}
      <Paper elevation={2} sx={{ p: 2, borderRadius: 3, mb: 2 }}>
        {isOffline && (
          <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
            <InputLabel id="offline-preset-label">Demo example</InputLabel>
            <Select
              labelId="offline-preset-label"
              label="Demo example"
              value={offlinePresetId}
              onChange={(e) => setOfflinePresetId(e.target.value)}
            >
              {OFFLINE_TEXT_PRESETS.map((p) => (
                <MenuItem key={p.id} value={p.id}>
                  {p.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        )}
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "stretch" }}>
          <TextField
            label={isOffline ? "Input text (locked in demo mode)" : "Input text"}
            multiline
            minRows={3}
            fullWidth
            size="small"
            value={text}
            onChange={(e) => setText(e.target.value)}
            sx={{ flex: 1 }}
            disabled={isOffline}
          />
          <Stack spacing={1} sx={{ flex: "0 0 auto" }}>
            <Button
              variant="contained"
              startIcon={
                isRunning ? (
                  <CircularProgress size={16} color="inherit" />
                ) : (
                  <PlayArrowRoundedIcon />
                )
              }
              onClick={handleRun}
              disabled={!canRun}
            >
              Run model
            </Button>
            <Tooltip title="Discard the last token and sample the same position again">
              <span>
                <Button
                  variant="outlined"
                  startIcon={<RefreshRoundedIcon />}
                  onClick={handleRerun}
                  disabled={last === null || isRunning}
                  fullWidth
                >
                  Rerun
                </Button>
              </span>
            </Tooltip>
          </Stack>
        </Box>
      </Paper>

      <Box
        sx={{
          display: "flex",
          gap: 2,
          alignItems: "stretch",
          flexWrap: "wrap",
        }}
      >
        {/* ── Controls ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "1 1 280px" }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Controls
          </Typography>
          <Divider sx={{ mb: 2 }} />

          <FormControl fullWidth size="small" sx={{ mb: 2 }}>
            <InputLabel id="model-select-label">Model</InputLabel>
            <Select
              labelId="model-select-label"
              label="Model"
              value={isOffline ? "demo-mode" : model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isOffline || modelsLoading || models.length === 0}
            >
              {isOffline ? (
                <MenuItem value="demo-mode">Demo Mode</MenuItem>
              ) : (
                models.map((m) => (
                  <MenuItem key={m.name} value={m.name}>
                    {m.name}
                  </MenuItem>
                ))
              )}
            </Select>
          </FormControl>
          {modelsLoading && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Loading models from Ollama…
              </Typography>
            </Box>
          )}

          <Typography variant="body2" gutterBottom>
            Temperature: <strong>{temperature.toFixed(2)}</strong>
          </Typography>
          <Slider
            value={temperature}
            onChange={(_, v) => setTemperature(v as number)}
            min={0}
            max={2}
            step={0.05}
            valueLabelDisplay="auto"
            sx={{ mb: 1 }}
          />

          <Typography variant="body2" gutterBottom>
            Top K: <strong>{topK}</strong>
            {topK === 0 && " (off)"}
          </Typography>
          <Slider
            value={topK}
            onChange={(_, v) => setTopK(v as number)}
            min={0}
            max={10}
            step={1}
            valueLabelDisplay="auto"
            sx={{ mb: 1 }}
          />

          <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
            <TextField
              label={isOffline ? "Seed (disabled in demo mode)" : "Seed (empty = random)"}
              size="small"
              value={seedText}
              onChange={(e) => setSeedText(e.target.value)}
              placeholder="e.g. 42"
              sx={{ flex: 1 }}
              disabled={isOffline}
            />
            <Tooltip title="Randomize seed">
              <IconButton onClick={randomizeSeed} size="small" disabled={isOffline}>
                <CasinoRoundedIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Clear seed (random every run)">
              <IconButton onClick={() => setSeedText("")} size="small" disabled={isOffline}>
                <ClearRoundedIcon />
              </IconButton>
            </Tooltip>
          </Box>
        </Paper>

        {/* ── Histogram ── */}
        <Paper
          elevation={2}
          sx={{ p: 2, borderRadius: 3, flex: "2 1 340px" }}
        >
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              flexWrap: "wrap",
              gap: 1,
              mb: 1,
            }}
          >
            <Typography variant="subtitle1" fontWeight="bold">
              Next-token probabilities
            </Typography>
          </Box>
          <Divider sx={{ mb: 1.5 }} />

          {last == null ? (<></>
          ) : (
            <>
              {last.chosenToken === "" && (
                <Alert severity="info" sx={{ mb: 1.5 }}>
                  The model sampled its end-of-turn control token, so no text
                  was appended. Click <strong>Rerun</strong> to draw a different
                  token from the same distribution.
                </Alert>
              )}
              <Histogram step={last} />
            </>
          )}
        </Paper>
      </Box>
    </Box>
  );
}
