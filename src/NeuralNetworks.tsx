import React, { useState, useMemo } from "react";
import {
  Box,
  Typography,
  Slider,
  Paper,
  Chip,
  Divider,
  TextField,
  Tooltip,
  useTheme,
} from "@mui/material";
import LocalMoviesRoundedIcon from "@mui/icons-material/LocalMoviesRounded";
import DoNotDisturbRoundedIcon from "@mui/icons-material/DoNotDisturbRounded";

function weightColor(weight: number): string {
  const intensity = Math.abs(weight);
  const [tr, tg, tb] = weight >= 0 ? [25, 118, 210] : [244, 67, 54];
  const r = Math.round(200 + (tr - 200) * intensity);
  const g = Math.round(200 + (tg - 200) * intensity);
  const b = Math.round(200 + (tb - 200) * intensity);
  return `rgb(${r},${g},${b})`;
}

const INPUTS = [
  { label: "Reviews", description: "Score from 1 (terrible) to 10 (outstanding)" },
  { label: "Cost", description: "Score from 1 (very expensive) to 10 (very cheap)" },
  { label: "Traffic", description: "Score from 1 (terrible traffic) to 10 (clear roads)" },
];

function PerceptronDiagram({
  inputs,
  weights,
  output,
}: {
  inputs: number[];
  weights: number[];
  output: number;
}) {
  const theme = useTheme();
  const circleStroke = theme.palette.text.primary;
  const textColor = theme.palette.text.primary;
  const labelColor = theme.palette.text.secondary;

  const width = 400;
  const height = 260;
  const inputX = 150;
  const nodeX = 320;
  const nodeY = height / 2;
  const inputYs = [60, 130, 200];
  const going = output > 0;
  const nodeColor = weightColor(going ? Math.abs(output) : -Math.abs(output));

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ overflow: "visible" }}>
      {/* Connection lines */}
      {inputs.map((value, i) => (
        <line
          key={i}
          x1={inputX + 22}
          y1={inputYs[i]}
          x2={nodeX - 22}
          y2={nodeY}
          stroke={weightColor(weights[i])}
          strokeWidth={3.5}
        />
      ))}

      {/* Input nodes */}
      {inputs.map((value, i) => {
        return (
          <g key={i}>
            <circle cx={inputX} cy={inputYs[i]} r={22} fill="transparent" stroke={circleStroke} strokeWidth={3.5} />
            <text
              x={inputX}
              y={inputYs[i]}
              textAnchor="middle"
              dominantBaseline="central"
              fill={textColor}
              fontSize={13}
              fontWeight="bold"
            >
              {value}
            </text>

            {/* Weight label on line */}
            <text
              x={(inputX + 22 + nodeX - 28) / 2}
              y={((inputYs[i] + nodeY) / 2) - 8}
              textAnchor="middle"
              fill={labelColor}
              fontSize={10}
              fontWeight="600"
            >
              w={weights[i].toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* Central node */}
      <circle
        cx={nodeX}
        cy={nodeY}
        r={22}
        fill="transparent"
        stroke={nodeColor}
        strokeWidth={3.5}
      />
      <text
        x={nodeX}
        y={nodeY}
        textAnchor="middle"
        dominantBaseline="central"
        fill={textColor}
        fontSize={11}
        fontWeight="bold"
      >
        {output.toFixed(2)}
      </text>

      {/* Input labels */}
      {INPUTS.map((inp, i) => (
        <text
          key={i}
          x={inputX - 28}
          y={inputYs[i]}
          textAnchor="end"
          dominantBaseline="central"
          fill={textColor}
          fontSize={11}
        >
          {inp.label}
        </text>
      ))}

      {/* Layer labels */}
      <text x={inputX} y={10} textAnchor="middle" fill={labelColor} fontSize={12}>
        Inputs
      </text>
      <text x={nodeX} y={10} textAnchor="middle" fill={labelColor} fontSize={12}>
        Output
      </text>
    </svg>
  );
}

export function NeuralNetworks() {
  const [inputs, setInputs] = useState([5, 5, 5]);
  const [weights, setWeights] = useState([0.5, 0.5, 0.5]);

  const setInput = (i: number, val: number) => {
    const clamped = Math.min(10, Math.max(1, val));
    setInputs((prev) => prev.map((v, j) => (j === i ? clamped : v)));
  };

  const output = useMemo(() => {
    // Normalize each input from 1-10 to 0-1, multiply by weight, sum, divide by 3
    // Output ranges from -1 to 1
    const sum = inputs.reduce((acc, val, i) => acc + ((val - 1) / 9) * weights[i], 0);
    return sum / 3;
  }, [inputs, weights]);

  const going = output > 0;
  const confidence = Math.abs(output); // 0 to 1

  return (
    <Box sx={{ pt: 2, pb: 1, px: 1 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        How AI Makes Decisions
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1.5 }}>
        AI models learn by adjusting <strong>weights</strong> — numbers that control how much
        each input matters. Here, you control the weights and inputs to see how the model
        decides whether to go to the movies.
      </Typography>

      <Box sx={{ display: "flex", gap: 2, alignItems: "stretch" }}>

        {/* Inputs panel */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "0 0 180px" }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Inputs
          </Typography>
          <Divider sx={{ mb: 2 }} />
          {INPUTS.map((inp, i) => (
            <Tooltip key={i} title={inp.description} placement="right">
              <Box sx={{ mb: 1 }}>
                <Typography variant="body2" fontWeight={500} sx={{ mb: 0.5 }}>
                  {inp.label}
                </Typography>
                <TextField
                  type="number"
                  value={inputs[i]}
                  onChange={(e) => setInput(i, parseInt(e.target.value) || 1)}
                  inputProps={{ min: 1, max: 10 }}
                  size="small"
                  fullWidth
                />
              </Box>
            </Tooltip>
          ))}

          <Divider sx={{ my: 1 }} />
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Output
          </Typography>
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 0.5, pt: 1 }}>
            {going ? (
              <LocalMoviesRoundedIcon sx={{ fontSize: 48, color: "#2e7d32" }} />
            ) : (
              <DoNotDisturbRoundedIcon sx={{ fontSize: 48, color: "#9e9e9e" }} />
            )}
            <Typography variant="h5" fontWeight="bold" sx={{ color: going ? "#2e7d32" : "#757575" }}>
              {going ? "Going" : "Not Going"}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {(confidence * 100).toFixed(1)}% confident
            </Typography>
          </Box>
        </Paper>

        {/* Diagram */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: 1 }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Basic Neural Network
          </Typography>
          <PerceptronDiagram inputs={inputs} weights={weights} output={output} />
        </Paper>

        {/* Weights panel */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "0 0 220px" }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Weights
          </Typography>
          <Divider sx={{ mb: 1 }} />
          {INPUTS.map((inp, i) => (
            <Box key={i} sx={{ mb: 1 }}>
              <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
                <Typography variant="body2" fontWeight={500}>
                  {inp.label}
                </Typography>
                <Chip
                  label={weights[i].toFixed(2)}
                  size="small"
                  sx={{ fontWeight: "bold", minWidth: 60, bgcolor: weightColor(weights[i]), color: "#1a1a1a" }}
                />
              </Box>
              <Slider
                value={weights[i]}
                min={-1}
                max={1}
                step={0.01}
                onChange={(_, val) =>
                  setWeights((prev) => prev.map((w, j) => (j === i ? (val as number) : w)))
                }
                sx={{ color: weightColor(weights[i]) }}
                marks={[
                  { value: -1, label: "-1" },
                  { value: 0, label: "0" },
                  { value: 1, label: "1" },
                ]}
              />
            </Box>
          ))}
        </Paper>

      </Box>
    </Box>
  );
}
