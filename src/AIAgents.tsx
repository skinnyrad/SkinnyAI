import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  FormControlLabel,
  IconButton,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Stack,
  Switch,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import PlayArrowRoundedIcon from "@mui/icons-material/PlayArrowRounded";
import RestartAltRoundedIcon from "@mui/icons-material/RestartAltRounded";
import { useOllamaOffline } from "./useOllamaStatus";
import { OFFLINE_AGENT_TRACES } from "./offlineFallbacks";

const DEFAULT_SYSTEM_PROMPT =
  "You are a helpful assistant. Use available tools when needed, be concise, and explain your final answer clearly.";

// Fixed harness loop budget: at most this many think → act → observe rounds per run.
const MAX_STEPS = 3;

type TraceKind =
  | "user"
  | "harness"
  | "llm"
  | "tool_call"
  | "tool_result"
  | "answer"
  | "error";

interface TraceStep {
  kind: TraceKind;
  title: string;
  body: string;
  raw?: string;
}

interface ToolDef {
  id: string;
  name: string;
  description: string;
  schema: Record<string, unknown>;
}

const TOOL_DEFS: ToolDef[] = [
  {
    id: "calculator",
    name: "calculator",
    description: "Evaluate arithmetic precisely.",
    schema: { expression: "string — e.g. \"27*14+8\"" },
  },
  {
    id: "clock",
    name: "clock",
    description: "Get the current time of day.",
    schema: { "(no arguments)": "" },
  },
  {
    id: "calendar",
    name: "calendar",
    description: "Get today's date and day of the week.",
    schema: { "(no arguments)": "" },
  },
  {
    id: "read_note",
    name: "read_note",
    description: "Read notes.txt / roadmap.txt.",
    schema: { name: "string — file name, default notes.txt" },
  },
  {
    id: "get_embedding_sentences",
    name: "get_embedding_sentences",
    description: "Get the current Vector Embeddings tab sentences.",
    schema: { "(no arguments)": "" },
  },
];

interface Task {
  id: string;
  label: string;
  request: string;
}

const TASKS: Task[] = [
  {
    id: "math",
    label: "Math (needs calculator)",
    request: "What is 27*14 + 8? Use the calculator tool.",
  },
  {
    id: "multistep",
    label: "Multi-step (calendar + calculator)",
    request:
      "What day of the week is it today, and what is 96 divided by 12? Use calendar, then the calculator.",
  },
  {
    id: "notes",
    label: "Notes lookup (needs read_note)",
    request: "Check notes.txt and tell me what I worked on yesterday.",
  },
  {
    id: "embeddings",
    label: "Embeddings (needs get_embedding_sentences)",
    request:
      "What sentences are currently on the Vector Embeddings tab? Use get_embedding_sentences and list them.",
  },
  {
    id: "custom",
    label: "Free chat (type below)",
    request: "",
  },
];

interface OllamaModel {
  name: string;
  model: string;
  capabilities?: string[] | null;
  details?: { parameter_size?: string } | null;
}

function parameterCount(size: string | undefined): number {
  if (!size) return Number.POSITIVE_INFINITY;
  const m = size.trim().match(/^([\d.]+)\s*([KMGT])?/i);
  if (!m) return Number.POSITIVE_INFINITY;
  const mult: Record<string, number> = { K: 1e3, M: 1e6, G: 1e9, B: 1e9, T: 1e12 };
  return Number.parseFloat(m[1]!) * (mult[(m[2] ?? "").toUpperCase()] ?? 1);
}

function stepBorderColor(kind: TraceKind): string {
  if (kind === "user") return "#8e24aa";
  if (kind === "harness") return "#78909c";
  if (kind === "llm") return "#1976d2";
  if (kind === "tool_call") return "#ed6c02";
  if (kind === "tool_result") return "#0288d1";
  if (kind === "answer") return "#2e7d32";
  return "#d32f2f";
}

function stepLabel(kind: TraceKind): string {
  if (kind === "user") return "USER";
  if (kind === "harness") return "HARNESS";
  if (kind === "llm") return "LLM";
  if (kind === "tool_call") return "TOOL CALL";
  if (kind === "tool_result") return "TOOL RESULT";
  if (kind === "answer") return "ANSWER";
  return "ERROR";
}

type ApprovalDecision = "approve" | "deny" | "stop";

interface PendingCall {
  name: string;
  argSummary: string;
  raw: string;
}

interface AssistantMessage {
  content?: string;
  tool_calls?: Array<{
    function?: { name?: string; arguments?: unknown };
  }>;
}

function parseToolArgs(rawArgs: unknown): Record<string, unknown> {
  try {
    if (typeof rawArgs === "string") {
      const parsed: unknown = JSON.parse(rawArgs);
      if (parsed && typeof parsed === "object") return parsed as Record<string, unknown>;
      return {};
    }
    if (rawArgs && typeof rawArgs === "object")
      return rawArgs as Record<string, unknown>;
  } catch {
    // fall through to empty args
  }
  return {};
}

function summarizeArgs(args: Record<string, unknown>): string {
  return Object.entries(args)
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function padToMinimum(start: number, minMs: number): Promise<void> {
  const remainder = minMs - (Date.now() - start);
  if (remainder > 0) await sleep(remainder);
}

type RunPhase =
  | "idle"
  | "staging"
  | "thinking"
  | "awaiting"
  | "executing"
  | "relaying"
  | "dispatching";

export function AIAgents() {
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [model, setModel] = useState("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState<string | null>(null);

  const [taskId, setTaskId] = useState(TASKS[0]!.id);
  const [prompt, setPrompt] = useState(TASKS[0]!.request);
  const [systemPrompt, setSystemPrompt] = useState(DEFAULT_SYSTEM_PROMPT);
  const [sysDialogOpen, setSysDialogOpen] = useState(false);
  const [sysDraft, setSysDraft] = useState(DEFAULT_SYSTEM_PROMPT);
  const [enabledTools, setEnabledTools] = useState<string[]>([
    "calculator",
    "clock",
    "calendar",
    "read_note",
    "get_embedding_sentences",
  ]);

  const [trace, setTrace] = useState<TraceStep[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [phase, setPhase] = useState<RunPhase>("idle");
  const [runError, setRunError] = useState<string | null>(null);

  // Ask-before-run: the harness pauses before each tool execution so
  // students can inspect what the model proposed. On by default.
  const [askBefore, setAskBefore] = useState(true);
  const askBeforeRef = useRef(true);
  const [pendingCall, setPendingCall] = useState<PendingCall | null>(null);
  const approvalRef = useRef<((d: ApprovalDecision) => void) | null>(null);
  const runIdRef = useRef(0);

  // Offline demo mode: Ollama unreachable → fixed tasks only, canned traces.
  const isOffline = useOllamaOffline(modelsError);
  const visibleTasks = isOffline ? TASKS.filter((t) => t.id !== "custom") : TASKS;

  // If offline engages while "Free chat" is selected, fall back to math.
  useEffect(() => {
    if (isOffline && taskId === "custom") {
      setTaskId(TASKS[0]!.id);
      setPrompt(TASKS[0]!.request);
    }
  }, [isOffline, taskId]);

  // Load chat-capable models (exclude embedding-only), default smallest.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/ollama/tags");
        const data = await res.json();
        if (!res.ok) throw new Error(data?.error ?? "Failed to list models.");
        const all: OllamaModel[] = data.models ?? [];
        const pool = all.filter((m) => {
          const caps = m.capabilities ?? [];
          if (caps.length === 0) return true;
          return !caps.every((c) => c === "embedding");
        });
        const list = pool.length > 0 ? pool : all;
        const smallest = [...list].sort(
          (a, b) =>
            parameterCount(a.details?.parameter_size) -
            parameterCount(b.details?.parameter_size),
        )[0];
        if (!cancelled) {
          setModels(list);
          setModel(smallest?.name ?? "");
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

  useEffect(() => () => {
    // Invalidate any in-flight run on unmount.
    runIdRef.current += 1;
    approvalRef.current?.("stop");
  }, []);

  const setAskBeforeBoth = useCallback((v: boolean) => {
    setAskBefore(v);
    askBeforeRef.current = v;
  }, []);

  const toggleTool = useCallback((id: string) => {
    setEnabledTools((prev) =>
      prev.includes(id) ? prev.filter((t) => t !== id) : [...prev, id],
    );
  }, []);

  const handleTaskChange = useCallback((id: string) => {
    if (isOffline && id === "custom") return;
    setTaskId(id);
    const task = TASKS.find((t) => t.id === id);
    if (task && task.id !== "custom") setPrompt(task.request);
    if (task?.id === "custom") setPrompt("");
  }, [isOffline]);

  const waitForApproval = useCallback(() => {
    return new Promise<ApprovalDecision>((resolve) => {
      approvalRef.current = resolve;
    });
  }, []);

  const handleRun = useCallback(async () => {
    if (!prompt.trim() || isRunning) return;
    if (!isOffline && !model) return;
    const runId = ++runIdRef.current;
    const alive = () => runIdRef.current === runId;
    const push = (step: TraceStep) => setTrace((prev) => [...prev, step]);

    setIsRunning(true);
    setPhase("staging");
    setRunError(null);
    setPendingCall(null);
    setTrace([]);

    // Offline demo: replay the canned trace with the same animated phases.
    if (isOffline) {
      const canned = OFFLINE_AGENT_TRACES[taskId] ?? [];
      try {
        await sleep(1000);
        if (!alive()) return;
        for (const step of canned) {
          if (!alive()) return;
          if (step.kind === "tool_call") {
            setPhase("dispatching");
            await sleep(250);
            if (!alive()) return;
            setPhase("executing");
            await sleep(1000);
          } else if (step.kind === "llm") {
            setPhase("thinking");
            await sleep(1200);
          } else {
            setPhase("relaying");
            await sleep(250);
          }
          if (!alive()) return;
          push({ ...step });
        }
      } catch (e) {
        if (!alive()) return;
        setRunError(e instanceof Error ? e.message : "Agent run failed.");
      } finally {
        if (alive()) {
          setIsRunning(false);
          setPhase("idle");
          setPendingCall(null);
        }
      }
      return;
    }

    const userPrompt = prompt.trim();
    const toolIds = [...enabledTools];
    const steps = MAX_STEPS;
    const sys = systemPrompt.trim() || DEFAULT_SYSTEM_PROMPT;

    push({ kind: "user", title: "User request", body: userPrompt });

    // Let the HARNESS chip light up for a full second before the prompt goes to the LLM.
    await sleep(1000);
    if (!alive()) return;

    try {
      const messages: Record<string, unknown>[] = [
        { role: "system", content: sys },
        { role: "user", content: userPrompt },
      ];
      let finalText = "";
      let stoppedEarly = false;

      for (let step = 0; step < steps; step++) {
        if (!alive()) return;
        setPhase("thinking");

        const llmStart = Date.now();
        const sres = await fetch("/api/ollama/agent/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, messages, tools: toolIds }),
        });
        const sdata = await sres.json();
        // Pad fast LLM responses so the LLM chip stays lit for at least two seconds.
        await padToMinimum(llmStart, 2000);
        if (!alive()) return;
        if (!sres.ok) throw new Error(sdata?.error ?? "LLM request failed.");

        const message = sdata.message as AssistantMessage;
        const content = (message?.content ?? "").trim();
        const toolCalls = message?.tool_calls ?? [];
        messages.push({
          role: "assistant",
          content: message?.content ?? "",
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        });

        if (toolCalls.length === 0) {
          if (content) {
            push({
              kind: "llm",
              title: `LLM responds`,
              body: content.slice(0, 2000),
              raw: JSON.stringify(message).slice(0, 2000),
            });
            finalText = content;
          } else {
            push({
              kind: "error",
              title: "Empty response",
              body: "The model returned no text and no tool calls.",
              raw: JSON.stringify(sdata).slice(0, 1000),
            });
          }
          break;
        }

        for (let ci = 0; ci < toolCalls.length; ci++) {
          const call = toolCalls[ci]!;
          if (!alive()) return;
          const name = call?.function?.name ?? "unknown";
          const args = parseToolArgs(call?.function?.arguments);
          const argSummary = summarizeArgs(args);
          push({
            kind: "tool_call",
            title: `Tool call proposed: ${name}${argSummary ? ` (${argSummary})` : ""}`,
            body: `Structured output from the LLM. Nothing has run yet — the harness is holding this proposal until it is approved.`,
            raw: JSON.stringify(call, null, 2),
          });

          let decision: ApprovalDecision = "approve";
          if (askBeforeRef.current) {
            setPendingCall({
              name,
              argSummary,
              raw: JSON.stringify(call, null, 2),
            });
            setPhase("awaiting");
            decision = await waitForApproval();
            setPendingCall(null);
            if (!alive()) return;
            if (decision === "stop") {
              stoppedEarly = true;
              break;
            }
            if (decision === "deny") {
              const msg = `Denied by user — the ${name} tool was NOT executed.`;
              messages.push({ role: "tool", content: `Error: ${msg}` });
              continue;
            }
          }

          if (!alive()) return;
          // When auto-approved there is no awaiting pause, so flash the
          // harness before tools to show the LLM → harness → tools handoff.
          if (!askBeforeRef.current) {
            setPhase("dispatching");
            await sleep(250);
            if (!alive()) return;
          }
          setPhase("executing");
          // Hold the TOOLS chip lit so the run is visible before the result returns.
          await sleep(1000);
          if (!alive()) return;
          const eres = await fetch("/api/ollama/agent/exec", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, args, allowedTools: toolIds }),
          });
          const edata = await eres.json();
          if (!alive()) return;
          if (edata?.output) {
            const output = String(edata.output);
            messages.push({ role: "tool", content: output });
          } else {
            const errMsg = edata?.error ?? "Tool failed.";
            messages.push({ role: "tool", content: `Error: ${errMsg}` });
          }
          // When one LLM response batches multiple tool calls, hold the LLM
          // chip between calls as a teaching-aid pause (not technically a new inference).
          if (ci < toolCalls.length - 1) {
            if (!alive()) return;
            setPhase("relaying");
            await sleep(250);
            if (!alive()) return;
            setPhase("thinking");
            await sleep(2000);
            if (!alive()) return;
          }
        }

        if (stoppedEarly) break;
        // Show the harness handing the tool result back before the next LLM call.
        if (toolCalls.length > 0) {
          if (!alive()) return;
          setPhase("relaying");
          await sleep(250);
          if (!alive()) return;
        }
        if (step === steps - 1) {
          push({
            kind: "error",
            title: "Max steps reached",
            body: `Stopped after ${steps} tool-use rounds. Increase max steps or simplify the task.`,
          });
        }
      }

      if (!alive()) return;
      if (stoppedEarly) {
        // Halt silently — partial trace above is preserved.
      } else if (!finalText) {
        // One last no-tools call so students still get a grounded summary.
        setPhase("thinking");
        try {
          const summaryStart = Date.now();
          const sres = await fetch("/api/ollama/agent/step", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model,
              messages: [
                ...messages,
                {
                  role: "user",
                  content: "Summarize the tool results above into a concise final answer.",
                },
              ],
              tools: [],
            }),
          });
          const sdata = await sres.json();
          await padToMinimum(summaryStart, 2000);
          if (!alive()) return;
          if (sres.ok && (sdata.message as AssistantMessage)?.content) {
            finalText = String((sdata.message as AssistantMessage).content);
            push({ kind: "llm", title: "LLM responds", body: finalText.slice(0, 3000) });
          }
        } catch {
          // Final summarization is best-effort.
        }
      }
    } catch (e) {
      if (!alive()) return;
      setRunError(e instanceof Error ? e.message : "Agent run failed.");
    } finally {
      if (alive()) {
        setIsRunning(false);
        setPhase("idle");
        setPendingCall(null);
      }
    }
  }, [model, prompt, isRunning, systemPrompt, enabledTools, waitForApproval, isOffline, taskId]);

  const handleStop = useCallback(() => {
    runIdRef.current += 1;
    approvalRef.current?.("stop");
    setPendingCall(null);
    setIsRunning(false);
    setPhase("idle");
  }, []);

  const handleApprove = useCallback(() => {
    approvalRef.current?.("approve");
  }, []);

  const handleDeny = useCallback(() => {
    approvalRef.current?.("deny");
  }, []);

  const handleApproveAll = useCallback(() => {
    // Approve this call and skip further prompts for the rest of the run.
    setAskBeforeBoth(false);
    approvalRef.current?.("approve");
  }, [setAskBeforeBoth]);

  const handleReset = useCallback(() => {
    runIdRef.current += 1;
    approvalRef.current?.("stop");
    setPendingCall(null);
    setIsRunning(false);
    setPhase("idle");
    setTrace([]);
    setRunError(null);
    setTaskId(TASKS[0]!.id);
    setPrompt(TASKS[0]!.request);
    setSystemPrompt(DEFAULT_SYSTEM_PROMPT);
    setEnabledTools(["calculator", "clock", "calendar", "read_note", "get_embedding_sentences"]);
    setAskBeforeBoth(true);
  }, [setAskBeforeBoth]);

  const activeTool = useMemo(() => {
    for (let i = trace.length - 1; i >= 0; i--) {
      if (trace[i]!.kind === "tool_call") {
        const m = trace[i]!.title.match(/Tool call: (\w+)/);
        return m?.[1] ?? null;
      }
    }
    return null;
  }, [trace]);

  const error = modelsError ?? runError;
  const canRun = (isOffline || model !== "") && prompt.trim() !== "" && !isRunning;
  const visibleTrace = useMemo(
    () =>
      trace.filter(
        (s) => s.kind !== "harness" && s.kind !== "tool_result" && s.kind !== "answer",
      ),
    [trace],
  );

  const isLLMActive = modelsLoading || (isRunning && phase === "thinking");
  const isHarnessActive =
    isRunning &&
    (phase === "staging" ||
      phase === "awaiting" ||
      phase === "relaying" ||
      phase === "dispatching");
  const isToolsActive = isRunning && phase === "executing";

  function chipColor(role: string): "primary" | "default" {
    if (role === "LLM" && isLLMActive) return "primary";
    if (role === "HARNESS" && isHarnessActive) return "primary";
    if (role === "TOOLS" && isToolsActive) return "primary";
    return "default";
  }

  return (
    <Box sx={{ pt: 2, pb: 1, px: 1 }}>
      <Typography variant="h5" fontWeight="bold" gutterBottom>
        AI Agents
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 1.5 }}>
        An agent is <strong>LLM (brain) + harness (nerves) + tools (hands)</strong>.
        The LLM only writes text — when it needs facts or math it emits a{" "}
        <strong>structured tool call</strong>, the harness runs real code, and the
        result goes back in. 
      </Typography>

      {!isOffline && error && (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          onClose={() => {
            setModelsError(null);
            setRunError(null);
          }}
        >
          {error} {error.includes("11434") && "Is `ollama serve` running?"}{" "}
          {error.toLowerCase().includes("tool") &&
            "Tip: pull a tool-capable model, e.g. `ollama pull qwen2.5`."}
        </Alert>
      )}
      {isOffline && runError && (
        <Alert
          severity="error"
          sx={{ mb: 2 }}
          onClose={() => setRunError(null)}
        >
          {runError}
        </Alert>
      )}

      <Box sx={{ display: "flex", gap: 2, alignItems: "flex-start", flexWrap: "wrap" }}>
        {/* ── Left column: ask + setup ── */}
        <Box
          sx={{
            flex: "0 1 300px",
            minWidth: 270,
            display: "flex",
            flexDirection: "column",
            gap: 2,
          }}
        >
          <Paper elevation={2} sx={{ p: 2, borderRadius: 3 }}>
            <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
              Ask
            </Typography>
            <Divider sx={{ mb: 1.5 }} />
            <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
              <InputLabel id="task-label">Task</InputLabel>
              <Select
                labelId="task-label"
                label="Task"
                value={taskId}
                onChange={(e) => handleTaskChange(e.target.value)}
              >
                {visibleTasks.map((t) => (
                  <MenuItem key={t.id} value={t.id}>
                    {t.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <TextField
              label={isOffline ? "Ask (locked in demo mode)" : "Ask"}
              multiline
              minRows={3}
              fullWidth
              size="small"
              value={prompt}
              onChange={(e) => {
                setPrompt(e.target.value);
                if (taskId !== "custom") setTaskId("custom");
              }}
              sx={{ mb: 1.5 }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) handleRun();
              }}
              slotProps={{ input: { readOnly: isOffline } }}
            />
            <Stack direction="row" spacing={1}>
              {isRunning ? (
                <Button variant="contained" color="warning" onClick={handleStop} fullWidth>
                  Stop
                </Button>
              ) : (
                <Button
                  variant="contained"
                  startIcon={<PlayArrowRoundedIcon />}
                  onClick={handleRun}
                  disabled={!canRun}
                  fullWidth
                >
                  Run agent
                </Button>
              )}
              <Tooltip title="Reset">
                <IconButton
                  size="small"
                  onClick={handleReset}
                  aria-label="Reset"
                  sx={{ border: 1, borderColor: "divider", borderRadius: 1.5 }}
                >
                  <RestartAltRoundedIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            </Stack>
          </Paper>

          {/* ── Setup ── */}
          <Paper elevation={2} sx={{ p: 2, borderRadius: 3 }}>
          <Typography variant="subtitle1" fontWeight="bold" gutterBottom>
            Agent setup
          </Typography>
          <Divider sx={{ mb: 1.5 }} />

          <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
            <InputLabel id="agent-model-label">Ollama model</InputLabel>
            <Select
              labelId="agent-model-label"
              label="Ollama model"
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
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
              <CircularProgress size={16} />
              <Typography variant="body2" color="text.secondary">
                Loading models from Ollama…
              </Typography>
            </Box>
          )}

          <Typography variant="body2" fontWeight="bold" gutterBottom>
            Tools
          </Typography>
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75, mb: 1 }}>
            {TOOL_DEFS.map((t) => {
              const on = enabledTools.includes(t.id);
              return (
                <Tooltip
                  key={t.id}
                  title={
                    <Box component="pre" sx={{ m: 0, fontSize: 11, whiteSpace: "pre-wrap" }}>
                      {`${t.name}(${Object.entries(t.schema)
                        .map(([k, v]) => `${k}: ${v}`)
                        .join(", ")})\n${t.description}`}
                    </Box>
                  }
                >
                  <Chip
                    label={t.name}
                    onClick={() => toggleTool(t.id)}
                    color={on ? "primary" : "default"}
                    variant={on ? "filled" : "outlined"}
                    size="small"
                    sx={{ fontWeight: on ? "bold" : "normal" }}
                  />
                </Tooltip>
              );
            })}
          </Box>

          <FormControlLabel
            control={
              <Switch
                checked={askBefore}
                onChange={(e) => setAskBeforeBoth(e.target.checked)}
                size="small"
              />
            }
            label={
              <Typography variant="body2" fontWeight="bold">
                Ask Before Tool Calls
              </Typography>
            }
            sx={{ mb: 0.25 }}
          />

          <Button
            variant="outlined"
            size="small"
            fullWidth
            onClick={() => {
              setSysDraft(systemPrompt);
              setSysDialogOpen(true);
            }}
            disabled={isOffline}
          >
            Edit system prompt
          </Button>
          <Divider sx={{ my: 1.5 }} />
        </Paper>
        </Box>

        {/* ── Live trace ── */}
        <Paper elevation={2} sx={{ p: 2, borderRadius: 3, flex: "1 1 420px", minWidth: 300 }}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: 1,
              flexWrap: "wrap",
            }}
          >
            <Typography variant="subtitle1" fontWeight="bold">
              Live Steps
            </Typography>
            {/* role strip */}
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                gap: 0.75,
                flexWrap: "wrap",
              }}
            >
              {["LLM", "HARNESS", "TOOLS"].map((role, i) => (
                <Box key={role} sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                  {i > 0 && (
                    <Typography color="text.secondary" variant="caption">
                      ⇄
                    </Typography>
                  )}
                  <Chip
                    label={role}
                    size="small"
                    color={chipColor(role)}
                    variant={chipColor(role) === "primary" ? "filled" : "outlined"}
                    sx={{ fontWeight: "bold" }}
                  />
                </Box>
              ))}
              {activeTool && (
                <Typography variant="caption" color="text.secondary">
                  last call: {activeTool}
                </Typography>
              )}
            </Box>
          </Box>
          <Divider sx={{ mb: 1.5, mt: 1 }} />

          {visibleTrace.length === 0 ? (
            <Typography variant="body2" color="text.secondary" align="center">
              Pick a task (or type your own) and click{" "}
              <strong>Run agent</strong>.
            </Typography>
          ) : (
            <Stack spacing={1}>
              {visibleTrace.map((step, i) => {
                return (
                  <Paper
                    key={i}
                    elevation={0}
                    sx={{
                      p: 1,
                      borderRadius: 1.5,
                      borderLeft: 4,
                      borderColor: stepBorderColor(step.kind),
                      bgcolor: "action.hover",
                    }}
                  >
                    <Typography variant="caption" fontWeight="bold" color="text.secondary">
                      {stepLabel(step.kind)}
                    </Typography>
                    <Typography variant="body2" fontWeight="bold">
                      {step.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" sx={{ whiteSpace: "pre-wrap" }}>
                      {step.body}
                    </Typography>
                  </Paper>
                );
              })}
              {pendingCall && (
                <Paper
                  elevation={2}
                  sx={{
                    p: 1.5,
                    borderRadius: 1.5,
                    border: 2,
                    borderColor: "warning.main",
                    bgcolor: "action.hover",
                  }}
                >
                  <Typography variant="caption" fontWeight="bold" color="warning.main">
                    APPROVAL NEEDED
                  </Typography>
                  <Typography variant="body2" fontWeight="bold" gutterBottom>
                    Run {pendingCall.name}
                    {pendingCall.argSummary ? ` (${pendingCall.argSummary})` : ""}?
                  </Typography>
                  <Typography
                    component="pre"
                    variant="caption"
                    sx={{
                      fontFamily: "monospace",
                      bgcolor: "background.paper",
                      borderRadius: 1,
                      p: 1,
                      m: 0,
                      mb: 1,
                      whiteSpace: "pre-wrap",
                      wordBreak: "break-word",
                      maxHeight: 180,
                      overflow: "auto",
                    }}
                  >
                    {pendingCall.raw}
                  </Typography>
                  <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                    <Button variant="contained" size="small" color="success" onClick={handleApprove}>
                      Approve
                    </Button>
                    <Button variant="outlined" size="small" color="error" onClick={handleDeny}>
                      Deny
                    </Button>
                    <Button size="small" onClick={handleApproveAll}>
                      Approve all for this run
                    </Button>
                  </Stack>
                </Paper>
              )}
              {isRunning && phase === "staging" && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Harness preparing prompt…
                  </Typography>
                </Box>
              )}
              {isRunning && phase === "thinking" && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Harness waiting on Ollama…
                  </Typography>
                </Box>
              )}
              {isRunning && phase === "executing" && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Harness running tool…
                  </Typography>
                </Box>
              )}
              {isRunning && phase === "relaying" && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Harness passing result to LLM…
                  </Typography>
                </Box>
              )}
              {isRunning && phase === "dispatching" && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Harness passing to tool…
                  </Typography>
                </Box>
              )}
              {isRunning && phase === "awaiting" && !pendingCall && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Harness paused — waiting…
                  </Typography>
                </Box>
              )}
            </Stack>
          )}
        </Paper>

      </Box>

      {/* ── System prompt dialog ── */}
      <Dialog
        open={sysDialogOpen}
        onClose={() => setSysDialogOpen(false)}
        fullWidth
        maxWidth="sm"
      >
        <DialogTitle>System prompt</DialogTitle>
        <DialogContent>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1.5 }}>
            Instructions sent to the LLM on every run, before the user request.
          </Typography>
          <TextField
            multiline
            minRows={5}
            fullWidth
            size="small"
            value={sysDraft}
            onChange={(e) => setSysDraft(e.target.value)}
            autoFocus
          />
        </DialogContent>
        <DialogActions>
          <Button
            size="small"
            onClick={() => {
              setSysDraft(DEFAULT_SYSTEM_PROMPT);
            }}
          >
            Restore default
          </Button>
          <Box sx={{ flexGrow: 1 }} />
          <Button size="small" onClick={() => setSysDialogOpen(false)}>
            Cancel
          </Button>
          <Button
            size="small"
            variant="contained"
            onClick={() => {
              setSystemPrompt(sysDraft);
              setSysDialogOpen(false);
            }}
          >
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
