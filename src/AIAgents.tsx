import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  Divider,
  MenuItem,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful assistant. Use available tools when needed, be concise, and explain your final answer clearly.";

type LlmId = "gemini" | "chatgpt" | "copilot" | "claude";
type ToolId =
  | "web-search"
  | "calculator"
  | "file-reader"
  | "calendar"
  | "weather-api";
type StepKind = "user" | "llm" | "tool" | "result" | "answer";

interface LlmOption {
  id: LlmId;
  name: string;
  tagline: string;
}

interface ToolOption {
  id: ToolId;
  name: string;
  description: string;
  short: string;
}

interface Scenario {
  id: string;
  label: string;
  request: string;
  preferredTool: ToolId;
}

interface RunStep {
  kind: StepKind;
  title: string;
  body: string;
}

const LLM_OPTIONS: LlmOption[] = [
  { id: "gemini", name: "Gemini", tagline: "Google LLM family" },
  { id: "chatgpt", name: "ChatGPT", tagline: "OpenAI LLM family" },
  { id: "copilot", name: "Copilot", tagline: "GitHub coding assistant LLMs" },
  { id: "claude", name: "Claude", tagline: "Anthropic LLM family" },
];

const TOOL_OPTIONS: ToolOption[] = [
  {
    id: "web-search",
    name: "Web Search",
    description: "Look up current information from the web.",
    short: "Search",
  },
  {
    id: "calculator",
    name: "Calculator",
    description: "Evaluate arithmetic precisely.",
    short: "Calc",
  },
  {
    id: "file-reader",
    name: "File Reader",
    description: "Read external notes or documents.",
    short: "Files",
  },
  {
    id: "calendar",
    name: "Calendar",
    description: "Check schedule and time slots.",
    short: "Calendar",
  },
  {
    id: "weather-api",
    name: "Weather API",
    description: "Fetch forecast data for a location.",
    short: "Weather",
  },
];

const SCENARIOS: Scenario[] = [
  {
    id: "renewable-headline",
    label: "News lookup",
    request: "Find one headline about renewable energy and summarize it in one sentence.",
    preferredTool: "web-search",
  },
  {
    id: "oscars-headline",
    label: "Entertainment lookup",
    request: "Find one headline about the Oscars and summarize it in one sentence.",
    preferredTool: "web-search",
  },
  {
    id: "math-check",
    label: "Math calculation",
    request: "Calculate 27 * 14.",
    preferredTool: "calculator",
  },
  {
    id: "notes-check",
    label: "Read project notes",
    request: "Check my notes and tell me what I worked on yesterday.",
    preferredTool: "file-reader",
  },
  {
    id: "meeting-time",
    label: "Schedule planning",
    request: "Find two open meeting times for tomorrow afternoon.",
    preferredTool: "calendar",
  },
  {
    id: "forecast",
    label: "Weather check",
    request: "What is the weather forecast in Seattle this afternoon?",
    preferredTool: "weather-api",
  },
];

function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 3)}...`;
}

function parseSimpleMathExpression(
  input: string,
): { expression: string; result: number } | null {
  const match = input.match(/(-?\d+(?:\.\d+)?)\s*([+\-*/])\s*(-?\d+(?:\.\d+)?)/);
  if (!match) return null;
  const left = Number(match[1]);
  const op = match[2];
  const right = Number(match[3]);
  if (!Number.isFinite(left) || !Number.isFinite(right)) return null;

  let result: number;
  if (op === "+") result = left + right;
  else if (op === "-") result = left - right;
  else if (op === "*") result = left * right;
  else if (op === "/") result = right === 0 ? Number.NaN : left / right;
  else return null;

  return { expression: `${left} ${op} ${right}`, result };
}

function getToolOutput(toolId: ToolId, scenarioId: string, request: string): string {
  if (toolId === "calculator") {
    const parsed = parseSimpleMathExpression(request);
    if (parsed && Number.isFinite(parsed.result)) {
      return `Computed ${parsed.expression} = ${parsed.result}.`;
    }
    return "Computed sample: 27 * 14 = 378.";
  }
  if (toolId === "web-search") {
    if (scenarioId === "renewable-headline") {
      return 'Top result: "Grid-scale battery deployments rise as solar installations expand."';
    }
    if (scenarioId === "oscars-headline") {
      return 'Top result: "Oscars 2026: Best Picture nominees announced with several first-time directors."';
    }
    return 'Top result: "Technology and business coverage highlights this week\'s market shifts."';
  }
  if (toolId === "file-reader") {
    return 'Read file project-notes.txt: "Yesterday: drafted Q2 roadmap and prioritized onboarding fixes."';
  }
  if (toolId === "calendar") {
    return "Calendar check: tomorrow has open slots at 1:00 PM and 3:30 PM.";
  }
  return "Weather lookup: Seattle forecast is 58F with light rain in the afternoon.";
}

function buildRunScript({
  modelId,
  systemPrompt,
  scenario,
  selectedToolIds,
}: {
  modelId: LlmId;
  systemPrompt: string;
  scenario: Scenario;
  selectedToolIds: ToolId[];
}): RunStep[] {
  const modelName = LLM_OPTIONS.find((m) => m.id === modelId)?.name ?? "LLM";
  const promptPreview = truncate(systemPrompt.replace(/\s+/g, " ").trim(), 120);

  const steps: RunStep[] = [
    { kind: "user", title: "User Request", body: scenario.request },
    {
      kind: "llm",
      title: `${modelName} Reads System Prompt`,
      body: `Instruction context: "${promptPreview}"`,
    },
  ];

  if (!selectedToolIds.includes(scenario.preferredTool)) {
    const neededTool =
      TOOL_OPTIONS.find((t) => t.id === scenario.preferredTool)?.name ??
      scenario.preferredTool;
    steps.push({
      kind: "llm",
      title: "Reasoning (Required Tool Missing)",
      body: `This scenario expects ${neededTool}, but it is not attached.`,
    });
    steps.push({
      kind: "answer",
      title: "Final Answer",
      body: `I cannot reliably complete this request without ${neededTool}. Attach that tool and run again.`,
    });
    return steps;
  }

  const chosenTool = scenario.preferredTool;
  const toolName = TOOL_OPTIONS.find((t) => t.id === chosenTool)?.name ?? chosenTool;
  const toolOutput = getToolOutput(chosenTool, scenario.id, scenario.request);
  steps.push({
    kind: "tool",
    title: `Tool Call: ${toolName}`,
    body: `Input: "${truncate(scenario.request, 90)}"`,
  });
  steps.push({
    kind: "result",
    title: `${toolName} Result`,
    body: toolOutput,
  });
  steps.push({
    kind: "answer",
    title: "Final Answer",
    body: `Using ${toolName}, the agent returns: ${toolOutput}`,
  });
  return steps;
}

function stepBorderColor(kind: StepKind): string {
  if (kind === "user") return "#8e24aa";
  if (kind === "llm") return "#1976d2";
  if (kind === "tool") return "#ed6c02";
  if (kind === "result") return "#0288d1";
  return "#2e7d32";
}

function VisualAgentCanvas({
  modelName,
  selectedToolIds,
}: {
  modelName: string;
  selectedToolIds: ToolId[];
}) {
  const selectedTools = TOOL_OPTIONS.filter((tool) =>
    selectedToolIds.includes(tool.id),
  );
  const width = 520;
  const height = 280;
  const centerX = 180;
  const centerY = 140;
  const llmRadius = 62;
  const toolX = 390;
  const top = 42;
  const spacing = 46;

  return (
    <svg
      width="100%"
      viewBox={`0 0 ${width} ${height}`}
      style={{ display: "block", borderRadius: 10 }}
    >
      <rect
        x={10}
        y={10}
        width={width - 20}
        height={height - 20}
        rx={14}
        fill="transparent"
        stroke="#90a4ae"
        strokeOpacity={0.35}
      />

      <circle
        cx={centerX}
        cy={centerY}
        r={llmRadius}
        fill="#ffffff"
        stroke="#1976d2"
        strokeWidth={2.5}
      />
      <text
        x={centerX}
        y={centerY - 8}
        textAnchor="middle"
        fill="#0d47a1"
        fontSize={14}
        fontWeight="bold"
      >
        LLM
      </text>
      <text
        x={centerX}
        y={centerY + 14}
        textAnchor="middle"
        fill="#0d47a1"
        fontSize={13}
        fontWeight="600"
      >
        {modelName}
      </text>

      {selectedTools.length === 0 ? (
        <text
          x={toolX - 90}
          y={centerY + 5}
          textAnchor="middle"
          fill="#607d8b"
          fontSize={11}
        >
          Attach tools to connect nodes
        </text>
      ) : (
        selectedTools.map((tool, idx) => {
          const toolY = top + idx * spacing;
          const toolW = 112;
          const toolH = 28;
          const toolMidY = toolY + toolH / 2;
          return (
            <g key={tool.id}>
              <line
                x1={centerX + llmRadius}
                y1={centerY}
                x2={toolX}
                y2={toolMidY}
                stroke="#26BDC0"
                strokeWidth={2.2}
                strokeOpacity={0.95}
              />
              <rect
                x={toolX}
                y={toolY}
                width={toolW}
                height={toolH}
                rx={14}
                fill="#ffffff"
                stroke="#1a9ea1"
                strokeWidth={1.6}
              />
              <text
                x={toolX + toolW / 2}
                y={toolMidY + 1}
                textAnchor="middle"
                dominantBaseline="central"
                fill="#0c4a52"
                fontSize={11}
                fontWeight="600"
              >
                {tool.short}
              </text>
            </g>
          );
        })
      )}
    </svg>
  );
}

export function AIAgents() {
  const [modelId, setModelId] = useState<LlmId>("copilot");
  const [selectedTools, setSelectedTools] = useState<Set<ToolId>>(
    () => new Set<ToolId>(["web-search"]),
  );
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [scenarioId, setScenarioId] = useState<string>(SCENARIOS[0]!.id);
  const [runSteps, setRunSteps] = useState<RunStep[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const timerIdsRef = useRef<number[]>([]);

  const selectedToolList = useMemo(
    () => TOOL_OPTIONS.filter((tool) => selectedTools.has(tool.id)),
    [selectedTools],
  );
  const selectedToolIds = useMemo(
    () => selectedToolList.map((tool) => tool.id),
    [selectedToolList],
  );
  const modelName =
    LLM_OPTIONS.find((model) => model.id === modelId)?.name ?? "Unknown";
  const selectedScenario = useMemo(
    () => SCENARIOS.find((scenario) => scenario.id === scenarioId) ?? SCENARIOS[0]!,
    [scenarioId],
  );

  const agentConfig = useMemo(
    () =>
      JSON.stringify(
        {
          model: modelId,
          tools: selectedToolIds,
          systemPrompt: systemPrompt.trim(),
          scenario: selectedScenario.id,
        },
        null,
        2,
      ),
    [modelId, selectedToolIds, systemPrompt, selectedScenario.id],
  );

  const clearTimers = useCallback(() => {
    for (const timerId of timerIdsRef.current) window.clearTimeout(timerId);
    timerIdsRef.current = [];
  }, []);

  useEffect(() => () => clearTimers(), [clearTimers]);

  const toggleTool = useCallback((toolId: ToolId) => {
    setSelectedTools((prev) => {
      const next = new Set(prev);
      if (next.has(toolId)) next.delete(toolId);
      else next.add(toolId);
      return next;
    });
  }, []);

  const handleRunAgent = useCallback(() => {
    clearTimers();
    setRunSteps([]);
    setIsRunning(true);

    const script = buildRunScript({
      modelId,
      systemPrompt,
      scenario: selectedScenario,
      selectedToolIds,
    });

    script.forEach((step, index) => {
      const timerId = window.setTimeout(() => {
        setRunSteps((prev) => [...prev, step]);
        if (index === script.length - 1) setIsRunning(false);
      }, index * 500);
      timerIdsRef.current.push(timerId);
    });
  }, [clearTimers, modelId, selectedScenario, selectedToolIds, systemPrompt]);

  const handleResetAgent = useCallback(() => {
    clearTimers();
    setIsRunning(false);
    setRunSteps([]);
    setModelId("copilot");
    setSelectedTools(new Set<ToolId>(["web-search"]));
    setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    setScenarioId(SCENARIOS[0]!.id);
  }, [clearTimers]);

  return (
    <Box sx={{ pt: 2, pb: 1, px: 1 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        AI Agents
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1.5 }}>
        An AI agent is an <strong>LLM + tools + instructions</strong>. This visual
        builder shows one central LLM node connected to attached tool nodes.
      </Typography>

      <Box sx={{ display: "flex", gap: 2, alignItems: "stretch" }}>
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "0 0 260px" }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Agent Builder
          </Typography>
          <Divider sx={{ mb: 1 }} />

          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            1) Choose LLM
          </Typography>
          <Stack spacing={0.75} sx={{ mb: 1.5 }}>
            {LLM_OPTIONS.map((model) => (
              <Paper
                key={model.id}
                elevation={0}
                onClick={() => setModelId(model.id)}
                sx={{
                  p: 1,
                  borderRadius: 1.5,
                  cursor: "pointer",
                  border: 2,
                  borderColor:
                    modelId === model.id ? "primary.main" : "divider",
                  bgcolor: modelId === model.id ? "action.selected" : "transparent",
                  "&:hover": { bgcolor: "action.hover" },
                }}
              >
                <Typography variant="body2" fontWeight="bold">
                  {model.name}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {model.tagline}
                </Typography>
              </Paper>
            ))}
          </Stack>

          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            2) Attach Tools (one or many)
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mb: 1.5 }}>
            {TOOL_OPTIONS.map((tool) => {
              const active = selectedTools.has(tool.id);
              return (
                <Chip
                  key={tool.id}
                  label={tool.name}
                  onClick={() => toggleTool(tool.id)}
                  color={active ? "primary" : "default"}
                  variant={active ? "filled" : "outlined"}
                  size="small"
                  sx={{ fontWeight: active ? "bold" : "normal" }}
                />
              );
            })}
          </Box>

          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            3) Configure Instructions
          </Typography>
          <TextField
            multiline
            minRows={4}
            fullWidth
            value={systemPrompt}
            onChange={(event) => setSystemPrompt(event.target.value)}
            size="small"
            placeholder="System prompt..."
            sx={{ mb: 1 }}
          />
          <TextField
            select
            fullWidth
            size="small"
            label="Scenario"
            value={scenarioId}
            onChange={(event) => setScenarioId(event.target.value)}
          >
            {SCENARIOS.map((scenario) => (
              <MenuItem key={scenario.id} value={scenario.id}>
                {scenario.label}
              </MenuItem>
            ))}
          </TextField>
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.75, display: "block" }}>
            {selectedScenario.request}
          </Typography>
        </Paper>

        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: 1 }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Visual Agent Graph
          </Typography>
          <Divider sx={{ mb: 1 }} />
          <VisualAgentCanvas modelName={modelName} selectedToolIds={selectedToolIds} />

          <Stack direction="row" spacing={1} sx={{ mt: 1, mb: 1.25 }}>
            <Button
              variant="contained"
              startIcon={<PlayArrowRoundedIcon />}
              onClick={handleRunAgent}
              disabled={isRunning}
              size="small"
            >
              {isRunning ? "Running..." : "Run Agent"}
            </Button>
            <Button
              variant="outlined"
              startIcon={<RestartAltRoundedIcon />}
              onClick={handleResetAgent}
              size="small"
            >
              Reset
            </Button>
          </Stack>

          <Divider sx={{ mb: 1 }} />
          <Typography variant="subtitle2" fontWeight="bold" gutterBottom>
            Simulated Run Timeline
          </Typography>
          {runSteps.length === 0 ? (
            <Typography variant="body2" color="text.secondary">
              Choose a scenario and click <strong>Run Agent</strong> to see a deterministic step-by-step simulation.
            </Typography>
          ) : (
            <Stack spacing={1}>
              {runSteps.map((step, index) => (
                <Paper
                  key={`${step.kind}-${index}`}
                  elevation={0}
                  sx={{
                    p: 1,
                    borderRadius: 1.5,
                    borderLeft: 4,
                    borderColor: stepBorderColor(step.kind),
                    bgcolor: "action.hover",
                  }}
                >
                  <Typography variant="body2" fontWeight="bold">
                    {step.title}
                  </Typography>
                  <Typography
                    variant="caption"
                    color="text.secondary"
                    sx={{ whiteSpace: "pre-wrap" }}
                  >
                    {step.body}
                  </Typography>
                </Paper>
              ))}
            </Stack>
          )}
        </Paper>

        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "0 0 245px" }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Built Agent
          </Typography>
          <Divider sx={{ mb: 1 }} />

          <Typography variant="body2" color="text.secondary">
            Model
          </Typography>
          <Typography variant="body1" fontWeight="bold" sx={{ mb: 1 }}>
            {modelName}
          </Typography>

          <Typography variant="body2" color="text.secondary">
            Attached Tools
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 1 }}>
            {selectedToolList.length > 0 ? (
              selectedToolList.map((tool) => (
                <Chip
                  key={`summary-${tool.id}`}
                  label={tool.name}
                  size="small"
                  color="primary"
                  variant="outlined"
                />
              ))
            ) : (
              <Typography variant="caption" color="text.secondary">
                None
              </Typography>
            )}
          </Box>

          {selectedToolList.length === 0 && (
            <Alert severity="warning" sx={{ mb: 1, py: 0 }}>
              No tools selected. This behaves like standard LLM chat.
            </Alert>
          )}

          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
            Scenario Requires
          </Typography>
          <Typography variant="caption" sx={{ display: "block", mb: 1 }}>
            {
              TOOL_OPTIONS.find((tool) => tool.id === selectedScenario.preferredTool)
                ?.name
            }
          </Typography>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 0.5 }}>
            Configuration
          </Typography>
          <Typography
            component="pre"
            variant="caption"
            sx={{
              fontFamily: "monospace",
              bgcolor: "action.hover",
              borderRadius: 1,
              p: 1,
              m: 0,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {agentConfig}
          </Typography>
        </Paper>
      </Box>
    </Box>
  );
}
