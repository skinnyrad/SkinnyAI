import { serve } from "bun";
import index from "./index.html";
import faviconPath from "./skinny_ai.png" with { type: "file" };

// ─── AI Agent harness: real Ollama tool-calling loop ───────────────────
// The harness is the "nervous system": it sends messages + tool schemas to
// the LLM, intercepts structured-output tool_calls, executes local tools,
// and feeds results back until the LLM answers or maxSteps is reached.

interface AgentToolDef {
  id: string;
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}

const AGENT_TOOL_DEFS: AgentToolDef[] = [
  {
    id: "calculator",
    name: "calculator",
    description:
      "Evaluate an arithmetic expression precisely. Use for any math instead of mental arithmetic.",
    parameters: {
      type: "object",
      properties: {
        expression: {
          type: "string",
          description: 'Arithmetic expression, e.g. "27*14+8" or "(96/12)+3".',
        },
      },
      required: ["expression"],
    },
  },
  {
    id: "clock",
    name: "clock",
    description:
      "Get the current time of day (hours, minutes, seconds, timezone). Use when the user asks about the time. For dates or days of the week, use calendar instead.",
    parameters: { type: "object", properties: {} },
  },
  {
    id: "calendar",
    name: "calendar",
    description:
      "Get today's date and day of the week (e.g. Thursday, September 18, 2026). Use when the user asks about today, tomorrow, dates, or days of the week. For the time of day, use clock instead.",
    parameters: { type: "object", properties: {} },
  },
  {
    id: "read_note",
    name: "read_note",
    description:
      "Read a project note file. Available files: notes.txt, roadmap.txt. Use when the user asks about notes, yesterday's work, or plans.",
    parameters: {
      type: "object",
      properties: {
        name: {
          type: "string",
          description: 'Note file name, e.g. "notes.txt". Defaults to notes.txt.',
        },
      },
    },
  },
  {
    id: "get_embedding_sentences",
    name: "get_embedding_sentences",
    description:
      "Get the current sentences on the Vector Embeddings tab. The tab syncs its sentence list here whenever it changes. Use when the user asks what sentences are being embedded or compared.",
    parameters: { type: "object", properties: {} },
  },
];

const BUNDLED_NOTES: Record<string, string> = {
  "notes.txt":
    "Yesterday: drafted Q2 roadmap and prioritized onboarding fixes. Blocked on API rate limits, asked Priya for help.",
  "roadmap.txt":
    "Q2 roadmap: 1) onboarding fixes, 2) offline mode, 3) classroom demo polish. Owner: Dalton.",
};

function safeCalculate(expression: string): string {
  const expr = expression.trim().slice(0, 100);
  if (!expr) throw new Error("Empty expression.");
  if (!/^[0-9+\-*/().\s%^]*$/.test(expr)) {
    throw new Error(`Refused to evaluate "${expr}": only numbers and + - * / ( ) % . allowed.`);
  }
  // eslint-disable-next-line no-new-func
  const fn = new Function(`"use strict"; return (${expr});`);
  const result = fn() as unknown;
  if (typeof result !== "number" || !Number.isFinite(result)) {
    throw new Error(`Expression "${expr}" did not produce a finite number.`);
  }
  return `${expr} = ${result}`;
}

const DEFAULT_EMBEDDING_SENTENCES = [
  "The cat sat on the mat",
  "A kitten curled up on the couch",
  "My cat loves to chase yarn",
  "Quantum computers use qubits to perform calculations",
];

// Live sentences synced from the Vector Embeddings tab via
// POST /api/embeddings/sentences. Falls back to defaults until the
// tab has synced at least once.
let syncedEmbeddingSentences: string[] = [...DEFAULT_EMBEDDING_SENTENCES];
let embeddingSentencesSyncedAt: number | null = null;

async function executeAgentTool(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  if (name === "calculator") {
    const expression = String(args.expression ?? "");
    if (!expression) throw new Error("calculator needs {expression}.");
    return `Computed ${safeCalculate(expression)}.`;
  }
  if (name === "clock") {
    const now = new Date();
    const time = now.toLocaleTimeString("en-US", {
      hour: "numeric",
      minute: "2-digit",
      second: "2-digit",
      timeZoneName: "short",
    });
    return `Current time: ${time} (server local).`;
  }
  if (name === "calendar") {
    const now = new Date();
    const date = now.toLocaleDateString("en-US", {
      weekday: "long",
      year: "numeric",
      month: "long",
      day: "numeric",
    });
    const iso = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
    return `Today is ${date} (${iso}, server local).`;
  }
  if (name === "read_note") {
    const key = String(args.name ?? "notes.txt").trim() || "notes.txt";
    const content = BUNDLED_NOTES[key] ?? BUNDLED_NOTES["notes.txt"];
    return `Read file ${key}: "${content}"`;
  }
  if (name === "get_embedding_sentences") {
    const list = syncedEmbeddingSentences
      .map((t, i) => `${i + 1}. "${t}"`)
      .join("\n");
    const source =
      embeddingSentencesSyncedAt === null
        ? "defaults (tab has not synced yet)"
        : `live tab contents (synced ${new Date(embeddingSentencesSyncedAt).toISOString()})`;
    return `Vector Embeddings tab sentences (${source}):\n${list}`.slice(0, 2000);
  }
  throw new Error(`Unknown tool: ${name}`);
}

interface TraceStep {
  kind: "user" | "harness" | "llm" | "tool_call" | "tool_result" | "answer" | "error";
  title: string;
  body: string;
  raw?: string;
}

function toOllamaTools(ids: string[]) {
  return AGENT_TOOL_DEFS.filter((t) => ids.includes(t.id)).map((t) => ({
    type: "function",
    function: { name: t.name, description: t.description, parameters: t.parameters },
  }));
}

const server = serve({
  routes: {
    // Serve the favicon file directly — otherwise the "/*" catch-all below
    // would return index.html for /skinny_ai.png and the icon breaks.
    "/skinny_ai.png": async () => {
      const file = Bun.file(faviconPath);
      return new Response(file, {
        headers: {
          "Content-Type": "image/png",
          "Cache-Control": "public, max-age=86400",
        },
      });
    },

    // Serve index.html for all unmatched routes.
    "/*": index,

    "/api/embeddings/sentences": {
      async GET() {
        return Response.json({
          sentences: syncedEmbeddingSentences,
          syncedAt: embeddingSentencesSyncedAt,
        });
      },
      async POST(req) {
        try {
          const body = (await req.json()) as { sentences?: unknown };
          if (!Array.isArray(body.sentences)) {
            return Response.json(
              { error: "Missing required field: sentences (array of strings)." },
              { status: 400 },
            );
          }
          const clean = body.sentences
            .filter((t): t is string => typeof t === "string")
            .map((t) => t.trim().slice(0, 500))
            .filter((t) => t.length > 0)
            .slice(0, 5);
          syncedEmbeddingSentences = clean.length > 0 ? clean : [...DEFAULT_EMBEDDING_SENTENCES];
          embeddingSentencesSyncedAt = Date.now();
          return Response.json({
            sentences: syncedEmbeddingSentences,
            syncedAt: embeddingSentencesSyncedAt,
          });
        } catch {
          return Response.json({ error: "Invalid JSON body." }, { status: 400 });
        }
      },
    },

    "/api/ollama/tags": {
      async GET() {
        try {
          const res = await fetch("http://localhost:11434/api/tags");
          const data = await res.json();
          return Response.json(data, { status: res.status });
        } catch (e) {
          return Response.json(
            { error: "Could not reach Ollama on localhost:11434. Is `ollama serve` running?" },
            { status: 502 },
          );
        }
      },
    },

    "/api/ollama/generate": {
      async POST(req) {
        try {
          const body = (await req.json()) as {
            model: string;
            prompt: string;
            temperature?: number;
            top_k?: number;
            topLogprobs?: number;
            seed?: number;
          };
          if (!body.model || !body.prompt) {
            return Response.json(
              { error: "Missing required fields: model, prompt." },
              { status: 400 },
            );
          }
          const topLogprobs = Math.min(
            20,
            Math.max(1, Math.round(body.topLogprobs ?? 10)),
          );
          const options: Record<string, unknown> = {
            // num_predict is intentionally hardcoded: one token per step.
            num_predict: 1,
          };
          if (body.temperature !== undefined) options.temperature = body.temperature;
          if (body.top_k !== undefined) options.top_k = body.top_k;
          if (body.seed !== undefined) options.seed = body.seed;

          const res = await fetch("http://localhost:11434/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: body.model,
              prompt: body.prompt,
              stream: false,
              // Raw mode: no chat template. The prompt is completed verbatim,
              // so every run is a true continuation of the text box.
              raw: true,
              logprobs: true,
              top_logprobs: topLogprobs,
              options,
            }),
          });
          const data = await res.json();
          if (!res.ok) {
            return Response.json(
              { error: data?.error ?? "Ollama request failed." },
              { status: res.status },
            );
          }
          const entry = data?.logprobs?.[0];
          const candidates = (entry?.top_logprobs ?? []).map(
            (t: { token: string; logprob: number }) => ({
              token: t.token,
              logprob: t.logprob,
              prob: Math.exp(t.logprob),
            }),
          );
          return Response.json({
            response: data.response ?? "",
            candidates,
          });
        } catch (e) {
          return Response.json(
            { error: "Could not reach Ollama on localhost:11434. Is `ollama serve` running?" },
            { status: 502 },
          );
        }
      },
    },
    "/api/ollama/embed": {
      async POST(req) {
        try {
          const body = (await req.json()) as {
            model: string;
            inputs: string[];
          };
          if (!body.model || !Array.isArray(body.inputs) || body.inputs.length === 0) {
            return Response.json(
              { error: "Missing required fields: model, inputs (non-empty array)." },
              { status: 400 },
            );
          }
          if (body.inputs.length > 5) {
            return Response.json(
              { error: "Too many inputs: maximum 5 sentences." },
              { status: 400 },
            );
          }
          const res = await fetch("http://localhost:11434/api/embed", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model: body.model,
              input: body.inputs,
            }),
          });
          const data = await res.json();
          if (!res.ok) {
            return Response.json(
              { error: data?.error ?? "Ollama embed request failed." },
              { status: res.status },
            );
          }
          return Response.json({ embeddings: data.embeddings ?? [] });
        } catch {
          return Response.json(
            { error: "Could not reach Ollama on localhost:11434. Is `ollama serve` running?" },
            { status: 502 },
          );
        }
      },
    },
    "/api/ollama/agent": {
      async POST(req) {
        try {
          const body = (await req.json()) as {
            model?: string;
            prompt?: string;
            systemPrompt?: string;
            tools?: string[];
            maxSteps?: number;
          };
          const model = (body.model ?? "").trim();
          const prompt = (body.prompt ?? "").trim();
          if (!model || !prompt) {
            return Response.json(
              { error: "Missing required fields: model, prompt." },
              { status: 400 },
            );
          }
          const toolIds = Array.isArray(body.tools) ? body.tools : [];
          const enabledDefs = AGENT_TOOL_DEFS.filter((t) => toolIds.includes(t.id));
          const maxSteps = Math.min(5, Math.max(1, Math.round(body.maxSteps ?? 4)));
          const systemPrompt =
            (body.systemPrompt ?? "").trim() ||
            "You are a helpful assistant. Use available tools when needed, be concise, and explain your final answer clearly.";

          const trace: TraceStep[] = [
            { kind: "user", title: "User request", body: prompt },
            {
              kind: "harness",
              title: `Harness → ${model}`,
              body:
                enabledDefs.length === 0
                  ? `Sending prompt with NO tools attached (chat-only ablation). Max steps: ${maxSteps}.`
                  : `Sending prompt + ${enabledDefs.length} tool schema(s): ${enabledDefs.map((t) => t.name).join(", ")}. Max steps: ${maxSteps}. The model may reply with text or a structured tool_call.`,
            },
          ];

          const messages: Record<string, unknown>[] = [
            { role: "system", content: systemPrompt },
            { role: "user", content: prompt },
          ];
          const ollamaTools = toOllamaTools(toolIds);
          let finalText = "";

          for (let step = 0; step < maxSteps; step++) {
            const res = await fetch("http://localhost:11434/api/chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model,
                messages,
                stream: false,
                ...(ollamaTools.length > 0 ? { tools: ollamaTools } : {}),
              }),
            });
            const data = await res.json();
            if (!res.ok) {
              trace.push({
                kind: "error",
                title: "LLM request failed",
                body: data?.error ?? "Ollama chat request failed.",
                raw: JSON.stringify(data).slice(0, 1000),
              });
              return Response.json({ trace, final: "" }, { status: res.status });
            }
            const message = data?.message as {
              content?: string;
              tool_calls?: Array<{
                function?: { name?: string; arguments?: unknown };
              }>;
            };
            const content = (message?.content ?? "").trim();
            const toolCalls = message?.tool_calls ?? [];

            // Keep history consistent for the next loop turn.
            messages.push({
              role: "assistant",
              content: message?.content ?? "",
              ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
            });

            if (toolCalls.length === 0) {
              if (content) {
                trace.push({
                  kind: "llm",
                  title: `LLM responds`,
                  body: content.slice(0, 2000),
                  raw: JSON.stringify(message).slice(0, 2000),
                });
                finalText = content;
              } else {
                trace.push({
                  kind: "error",
                  title: "Empty response",
                  body: "The model returned no text and no tool calls.",
                  raw: JSON.stringify(data).slice(0, 1000),
                });
              }
              break;
            }

            for (const call of toolCalls) {
              const name = call?.function?.name ?? "unknown";
              let args: Record<string, unknown> = {};
              const rawArgs = call?.function?.arguments;
              try {
                if (typeof rawArgs === "string") args = JSON.parse(rawArgs);
                else if (rawArgs && typeof rawArgs === "object")
                  args = rawArgs as Record<string, unknown>;
              } catch {
                args = {};
              }
              const argSummary = Object.entries(args)
                .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                .join(", ");
              trace.push({
                kind: "tool_call",
                title: `Tool call: ${name}${argSummary ? ` (${argSummary})` : ""}`,
                body: `Structured output from the LLM. The harness intercepts it and runs real code — the LLM never runs code itself.`,
                raw: JSON.stringify(call, null, 2),
              });
              try {
                if (!toolIds.includes(name)) {
                  throw new Error(
                    `Tool "${name}" is not attached. Toggle it on and run again.`,
                  );
                }
                const output = await executeAgentTool(name, args);
                trace.push({
                  kind: "tool_result",
                  title: `${name} result`,
                  body: output.slice(0, 2000),
                });
                messages.push({ role: "tool", content: output });
              } catch (e) {
                const errMsg = e instanceof Error ? e.message : "Tool failed.";
                trace.push({ kind: "tool_result", title: `${name} error`, body: errMsg });
                messages.push({ role: "tool", content: `Error: ${errMsg}` });
              }
            }

            if (step === maxSteps - 1) {
              trace.push({
                kind: "error",
                title: "Max steps reached",
                body: `Stopped after ${maxSteps} tool-use rounds. Increase max steps or simplify the task.`,
              });
            }
          }

          if (!finalText) {
            // One last no-tools call so students still get a grounded summary.
            try {
              const res = await fetch("http://localhost:11434/api/chat", {
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
                  stream: false,
                }),
              });
              const data = await res.json();
              if (res.ok && data?.message?.content) {
                finalText = String(data.message.content);
                trace.push({
                  kind: "answer",
                  title: "Final answer",
                  body: finalText.slice(0, 3000),
                });
              }
            } catch {
              // Final summarization is best-effort.
            }
          } else {
            trace.push({ kind: "answer", title: "Final answer", body: finalText.slice(0, 3000) });
          }

          return Response.json({ trace, final: finalText });
        } catch {
          return Response.json(
            { error: "Could not reach Ollama on localhost:11434. Is `ollama serve` running?" },
            { status: 502 },
          );
        }
      },
    },
    "/api/ollama/agent/step": {
      // Single harness turn: one Ollama chat call with tool schemas.
      // Returns the assistant message (text + tool_calls) WITHOUT executing
      // anything, so the frontend can pause for user approval first.
      async POST(req) {
        try {
          const body = (await req.json()) as {
            model?: string;
            messages?: Record<string, unknown>[];
            tools?: string[];
          };
          const model = (body.model ?? "").trim();
          const messages = body.messages;
          if (!model || !Array.isArray(messages) || messages.length === 0) {
            return Response.json(
              { error: "Missing required fields: model, messages (non-empty array)." },
              { status: 400 },
            );
          }
          const toolIds = Array.isArray(body.tools) ? body.tools : [];
          const ollamaTools = toOllamaTools(toolIds);
          const res = await fetch("http://localhost:11434/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              model,
              messages,
              stream: false,
              ...(ollamaTools.length > 0 ? { tools: ollamaTools } : {}),
            }),
          });
          const data = await res.json();
          if (!res.ok) {
            return Response.json(
              { error: data?.error ?? "Ollama chat request failed." },
              { status: res.status },
            );
          }
          return Response.json({ message: data?.message ?? { content: "" } });
        } catch {
          return Response.json(
            { error: "Could not reach Ollama on localhost:11434. Is `ollama serve` running?" },
            { status: 502 },
          );
        }
      },
    },
    "/api/ollama/agent/exec": {
      // Execute one approved tool call server-side.
      async POST(req) {
        try {
          const body = (await req.json()) as {
            name?: string;
            args?: unknown;
            allowedTools?: string[];
          };
          const name = (body.name ?? "").trim();
          if (!name) {
            return Response.json({ error: "Missing required field: name." }, { status: 400 });
          }
          let args: Record<string, unknown> = {};
          try {
            if (typeof body.args === "string") args = JSON.parse(body.args);
            else if (body.args && typeof body.args === "object")
              args = body.args as Record<string, unknown>;
          } catch {
            args = {};
          }
          if (Array.isArray(body.allowedTools) && !body.allowedTools.includes(name)) {
            return Response.json(
              { output: null, error: `Tool "${name}" is not attached. Toggle it on and run again.` },
              { status: 200 },
            );
          }
          try {
            const output = await executeAgentTool(name, args);
            return Response.json({ output });
          } catch (e) {
            return Response.json(
              { output: null, error: e instanceof Error ? e.message : "Tool failed." },
              { status: 200 },
            );
          }
        } catch {
          return Response.json({ error: "Tool execution failed." }, { status: 500 });
        }
      },
    },
    "/api/hello": {
      async GET(req) {
        return Response.json({
          message: "Hello, world!",
          method: "GET",
        });
      },
      async PUT(req) {
        return Response.json({
          message: "Hello, world!",
          method: "PUT",
        });
      },
    },

    "/api/hello/:name": async req => {
      const name = req.params.name;
      return Response.json({
        message: `Hello, ${name}!`,
      });
    },
  },

  development: process.env.NODE_ENV !== "production" && {
    // Enable browser hot reloading in development
    hmr: true,

    // Echo console logs from the browser to the server
    console: true,
  },
});

console.log(`🚀 Server running at ${server.url}`);
