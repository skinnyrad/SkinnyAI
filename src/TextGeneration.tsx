import { useState, useMemo, useEffect, useCallback } from "react";
import {
  Box,
  Typography,
  Paper,
  Chip,
  Divider,
  Button,
  IconButton,
  useTheme,
  Stack,
} from "@mui/material";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import StopRoundedIcon from "@mui/icons-material/StopRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import SkipNextRoundedIcon from "@mui/icons-material/SkipNextRounded";
import NavigateNextRoundedIcon from "@mui/icons-material/NavigateNextRounded";
import NavigateBeforeRoundedIcon from "@mui/icons-material/NavigateBeforeRounded";

// ─── Vocabulary & Training Data ───────────────────────────

const SENTENCES = [
  ["the", "dog", "jumped", "over", "the", "fence"],
  ["the", "cat", "sat", "on", "the", "mat"],
  ["I", "like", "to", "eat", "good", "food"],
  ["she", "went", "to", "the", "store"],
];

const VOCAB: string[] = [...new Set(SENTENCES.flat())].sort();
const V = VOCAB.length;
const W2I: Record<string, number> = Object.fromEntries(
  VOCAB.map((w, i) => [w, i]),
);

// All consecutive word pairs from training data
const BIGRAMS: [number, number][] = SENTENCES.flatMap((s) =>
  s
    .slice(0, -1)
    .map((w, i) => [W2I[w]!, W2I[s[i + 1]!]!] as [number, number]),
);

const LEARNING_RATE = 0.1;
const TOP_N = 7;

// ─── Math Helpers ─────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((l) => Math.exp(l - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function initWeights(): number[][] {
  return Array.from({ length: V }, () =>
    Array.from({ length: V }, () => (Math.random() - 0.5) * 0.2),
  );
}

function trainEpoch(
  weights: number[][],
  lr: number,
): { weights: number[][]; loss: number } {
  const w = weights.map((row) => [...row]);
  let totalLoss = 0;

  for (const [inp, tgt] of BIGRAMS) {
    const probs = softmax(w[inp]!);
    totalLoss -= Math.log((probs[tgt] ?? 0) + 1e-10);
    // Gradient: softmax output minus one-hot target
    for (let j = 0; j < V; j++) {
      w[inp]![j]! -= lr * ((probs[j] ?? 0) - (j === tgt ? 1 : 0));
    }
  }

  return { weights: w, loss: totalLoss / BIGRAMS.length };
}

function weightColor(weight: number): string {
  const intensity = Math.min(Math.abs(weight), 1);
  const [tr, tg, tb] = weight >= 0 ? [25, 118, 210] : [244, 67, 54];
  const r = Math.round(200 + (tr - 200) * intensity);
  const g = Math.round(200 + (tg - 200) * intensity);
  const b = Math.round(200 + (tb - 200) * intensity);
  return `rgb(${r},${g},${b})`;
}

// ─── Types ────────────────────────────────────────────────

interface Candidate {
  wordIdx: number;
  word: string;
  prob: number;
  isTarget: boolean;
}

interface TrainState {
  weights: number[][];
  loss: number | null;
  epoch: number;
}

// ─── SVG Diagram ──────────────────────────────────────────

function BigramDiagram({
  inputWord,
  candidates,
  inputWeights,
}: {
  inputWord: string;
  candidates: Candidate[];
  inputWeights: number[];
}) {
  const theme = useTheme();
  const textColor = theme.palette.text.primary;
  const labelColor = theme.palette.text.secondary;
  const circleStroke = theme.palette.text.primary;

  const width = 500;
  const height = 360;
  const inX = 90;
  const inY = height / 2;
  const outX = 340;
  const n = candidates.length;
  const outYs = candidates.map(
    (_, i) => 40 + (i * (height - 80)) / Math.max(n - 1, 1),
  );

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      style={{ overflow: "visible" }}
    >
      {/* Connection lines */}
      {candidates.map((c, i) => (
        <line
          key={c.wordIdx}
          x1={inX + 28}
          y1={inY}
          x2={outX - 22}
          y2={outYs[i]!}
          stroke={weightColor(inputWeights[c.wordIdx]!)}
          strokeWidth={Math.max(1.5, c.prob * 8)}
          opacity={0.6 + c.prob * 0.4}
        />
      ))}

      {/* Weight labels on lines */}
      {candidates.map((c, i) => {
        const mx = (inX + 28 + outX - 22) / 2;
        const my = (inY + outYs[i]!) / 2 - 8;
        return (
          <text
            key={`w-${c.wordIdx}`}
            x={mx}
            y={my}
            textAnchor="middle"
            fill={labelColor}
            fontSize={9}
            fontWeight="600"
          >
            w={inputWeights[c.wordIdx]!.toFixed(2)}
          </text>
        );
      })}

      {/* Input node */}
      <circle
        cx={inX}
        cy={inY}
        r={28}
        fill="transparent"
        stroke={circleStroke}
        strokeWidth={3}
      />
      <text
        x={inX}
        y={inY}
        textAnchor="middle"
        dominantBaseline="central"
        fill={textColor}
        fontSize={13}
        fontWeight="bold"
      >
        {inputWord}
      </text>

      {/* Output nodes */}
      {candidates.map((c, i) => (
        <g key={`o-${c.wordIdx}`}>
          <circle
            cx={outX}
            cy={outYs[i]!}
            r={20}
            fill="transparent"
            stroke={c.isTarget ? "#2e7d32" : circleStroke}
            strokeWidth={c.isTarget ? 3.5 : 2}
          />
          <text
            x={outX}
            y={outYs[i]!}
            textAnchor="middle"
            dominantBaseline="central"
            fill={c.isTarget ? "#2e7d32" : textColor}
            fontSize={10}
            fontWeight={c.isTarget ? "bold" : "normal"}
          >
            {c.word}
          </text>

          {/* Probability bar */}
          <rect
            x={outX + 26}
            y={outYs[i]! - 7}
            width={Math.max(2, c.prob * 100)}
            height={14}
            rx={3}
            fill={c.isTarget ? "#2e7d3288" : theme.palette.primary.main + "55"}
          />
          <text
            x={outX + 32 + c.prob * 100}
            y={outYs[i]!}
            dominantBaseline="central"
            fill={labelColor}
            fontSize={10}
          >
            {(c.prob * 100).toFixed(1)}%
          </text>
        </g>
      ))}

      {/* Column labels */}
      <text
        x={inX}
        y={14}
        textAnchor="middle"
        fill={labelColor}
        fontSize={11}
      >
        Context Word
      </text>
      <text
        x={outX + 30}
        y={14}
        textAnchor="middle"
        fill={labelColor}
        fontSize={11}
      >
        Predicted Next Word
      </text>
    </svg>
  );
}

// ─── Main Component ───────────────────────────────────────

export function TextGeneration() {
  const [sentIdx, setSentIdx] = useState(0);
  const [wordPos, setWordPos] = useState(0);
  const [trainState, setTrainState] = useState<TrainState>(() => ({
    weights: initWeights(),
    loss: null,
    epoch: 0,
  }));
  const [isTraining, setIsTraining] = useState(false);

  const sentence = SENTENCES[sentIdx]!;
  const maxPos = sentence.length - 2;
  const clampedPos = Math.min(wordPos, maxPos);
  const inputWord = sentence[clampedPos]!;
  const targetWord = sentence[clampedPos + 1]!;
  const inputIdx = W2I[inputWord]!;
  const targetIdx = W2I[targetWord]!;

  const probs = useMemo(
    () => softmax(trainState.weights[inputIdx]!),
    [trainState.weights, inputIdx],
  );

  const predictedIdx = useMemo(() => {
    let best = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i]! > probs[best]!) best = i;
    }
    return best;
  }, [probs]);

  const predictedWord = VOCAB[predictedIdx]!;
  const isCorrect = predictedIdx === targetIdx;

  const candidates = useMemo((): Candidate[] => {
    const indexed = probs.map((p, i) => ({
      wordIdx: i,
      word: VOCAB[i]!,
      prob: p,
      isTarget: i === targetIdx,
    }));
    indexed.sort((a, b) => b.prob - a.prob);
    const top = indexed.slice(0, TOP_N);
    if (!top.some((c) => c.isTarget)) {
      top.pop();
      top.push(indexed.find((c) => c.isTarget)!);
      top.sort((a, b) => b.prob - a.prob);
    }
    return top;
  }, [probs, targetIdx]);

  // Auto-training loop
  useEffect(() => {
    if (!isTraining) return;
    const id = setInterval(() => {
      setTrainState((prev) => {
        const result = trainEpoch(prev.weights, LEARNING_RATE);
        if (result.loss < 0.05 || prev.epoch >= 500) {
          setIsTraining(false);
        }
        return {
          weights: result.weights,
          loss: result.loss,
          epoch: prev.epoch + 1,
        };
      });
    }, 80);
    return () => clearInterval(id);
  }, [isTraining]);

  const handleTrainStep = useCallback(() => {
    setTrainState((prev) => {
      const result = trainEpoch(prev.weights, LEARNING_RATE);
      return {
        weights: result.weights,
        loss: result.loss,
        epoch: prev.epoch + 1,
      };
    });
  }, []);

  const handleReset = useCallback(() => {
    setIsTraining(false);
    setTrainState({ weights: initWeights(), loss: null, epoch: 0 });
  }, []);

  const handleSentenceChange = (idx: number) => {
    setSentIdx(idx);
    setWordPos(0);
  };

  return (
    <Box sx={{ pt: 2, pb: 1, px: 1 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        How LLMs Predict Text
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1.5 }}>
        Large Language Models predict the <strong>next word</strong> based on
        context. This simplified bigram model learns which word typically follows
        another by adjusting weights during training — just like a real LLM.
      </Typography>

      <Box sx={{ display: "flex", gap: 2, alignItems: "stretch" }}>
        {/* ── Left Panel: Sentences & Prediction ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "0 0 220px" }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Training Sentences
          </Typography>
          <Divider sx={{ mb: 1 }} />
          <Stack spacing={0.5} sx={{ mb: 1.5 }}>
            {SENTENCES.map((s, i) => (
              <Chip
                key={i}
                label={s.join(" ")}
                onClick={() => handleSentenceChange(i)}
                variant={i === sentIdx ? "filled" : "outlined"}
                color={i === sentIdx ? "primary" : "default"}
                size="small"
                sx={{
                  justifyContent: "flex-start",
                  height: "auto",
                  py: 0.5,
                  "& .MuiChip-label": { whiteSpace: "normal" },
                }}
              />
            ))}
          </Stack>

          <Divider sx={{ mb: 1 }} />
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Context
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 1 }}>
            {sentence.map((word, i) => {
              const isInput = i === clampedPos;
              const isPredicting = i === clampedPos + 1;
              return (
                <Chip
                  key={i}
                  label={isPredicting ? "?" : word}
                  size="small"
                  onClick={
                    i <= maxPos ? () => setWordPos(i) : undefined
                  }
                  sx={{
                    cursor: i <= maxPos ? "pointer" : "default",
                    fontWeight: isInput ? "bold" : "normal",
                    borderColor: isInput
                      ? "primary.main"
                      : isPredicting
                        ? "secondary.main"
                        : undefined,
                    borderWidth: isInput || isPredicting ? 2 : 1,
                    opacity: i > clampedPos + 1 ? 0.35 : 1,
                  }}
                  variant="outlined"
                />
              );
            })}
          </Box>
          <Box
            sx={{ display: "flex", justifyContent: "center", gap: 1, mb: 1 }}
          >
            <IconButton
              size="small"
              onClick={() => setWordPos(Math.max(0, clampedPos - 1))}
              disabled={clampedPos === 0}
            >
              <NavigateBeforeRoundedIcon />
            </IconButton>
            <IconButton
              size="small"
              onClick={() => setWordPos(Math.min(maxPos, clampedPos + 1))}
              disabled={clampedPos >= maxPos}
            >
              <NavigateNextRoundedIcon />
            </IconButton>
          </Box>

          <Divider sx={{ my: 1 }} />
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Prediction
          </Typography>
          <Box sx={{ textAlign: "center", py: 1 }}>
            <Typography variant="body2" color="text.secondary">
              After &ldquo;<strong>{inputWord}</strong>&rdquo;, model predicts:
            </Typography>
            <Typography
              variant="h6"
              fontWeight="bold"
              sx={{ color: isCorrect ? "#2e7d32" : "#f44336" }}
            >
              &ldquo;{predictedWord}&rdquo; (
              {(probs[predictedIdx]! * 100).toFixed(1)}%)
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Correct: &ldquo;<strong>{targetWord}</strong>&rdquo;{" "}
              {isCorrect ? "✓" : "✗"}
            </Typography>
          </Box>
        </Paper>

        {/* ── Center Panel: Diagram ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: 1 }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Next-Word Prediction Network
          </Typography>
          <BigramDiagram
            inputWord={inputWord}
            candidates={candidates}
            inputWeights={trainState.weights[inputIdx]!}
          />
        </Paper>

        {/* ── Right Panel: Training Controls ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "0 0 180px" }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Training
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Stack spacing={1.5}>
            <Button
              variant="contained"
              startIcon={<SkipNextRoundedIcon />}
              onClick={handleTrainStep}
              disabled={isTraining}
              fullWidth
              size="small"
            >
              Train Step
            </Button>
            <Button
              variant="contained"
              color={isTraining ? "error" : "success"}
              startIcon={
                isTraining ? <StopRoundedIcon /> : <PlayArrowRoundedIcon />
              }
              onClick={() => setIsTraining((t) => !t)}
              fullWidth
              size="small"
            >
              {isTraining ? "Stop" : "Auto Train"}
            </Button>
            <Button
              variant="outlined"
              startIcon={<RestartAltRoundedIcon />}
              onClick={handleReset}
              fullWidth
              size="small"
            >
              Reset
            </Button>
          </Stack>

          <Divider sx={{ my: 2 }} />
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Metrics
          </Typography>
          <Box sx={{ mb: 1 }}>
            <Typography variant="body2" color="text.secondary">
              Epoch
            </Typography>
            <Typography variant="h6" fontWeight="bold">
              {trainState.epoch}
            </Typography>
          </Box>
          <Box>
            <Typography variant="body2" color="text.secondary">
              Avg Loss
            </Typography>
            <Typography
              variant="h6"
              fontWeight="bold"
              sx={{
                color:
                  trainState.loss !== null && trainState.loss < 0.5
                    ? "#2e7d32"
                    : undefined,
              }}
            >
              {trainState.loss !== null ? trainState.loss.toFixed(4) : "—"}
            </Typography>
          </Box>
        </Paper>
      </Box>
    </Box>
  );
}
