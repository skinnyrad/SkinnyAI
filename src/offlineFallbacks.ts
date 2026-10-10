// Hardcoded demo data used ONLY when Ollama is unreachable.
// Each tab locks free-form input in offline mode and plays back these
// fixed examples instead of calling /api/ollama/*.

export interface OfflineCandidate {
  token: string;
  logprob: number;
  prob: number;
}

function c(token: string, logprob: number): OfflineCandidate {
  return { token, logprob, prob: Math.exp(logprob) };
}

export interface OfflineTextVariant {
  response: string;
  candidates: OfflineCandidate[];
}

export interface OfflineTextPreset {
  id: string;
  label: string;
  prompt: string;
  variants: OfflineTextVariant[];
}

function controlTail(): OfflineCandidate[] {
  return [c(" hill", -3.9), c("", -4.2)];
}

// Plausible canned next-token distributions. Temperature/top-K sliders
// still reshape these via the existing computeDisplay().
export const OFFLINE_TEXT_PRESETS: OfflineTextPreset[] = [
  {
    id: "cat",
    label: 'Preset: "The cat sat on the…"',
    prompt: "The cat sat on the",
    variants: [
      {
        response: " mat",
        candidates: [
          c(" mat", -0.3),
          c(" couch", -1.2),
          c(" floor", -1.8),
          c(" rug", -2.1),
          c(" bed", -2.4),
          c(" table", -2.9),
          c(" chair", -3.2),
          c(" roof", -3.6),
          ...controlTail(),
        ],
      },
      {
        response: " couch",
        candidates: [
          c(" couch", -0.5),
          c(" mat", -0.9),
          c(" floor", -1.6),
          c(" rug", -2.0),
          c(" bed", -2.3),
          c(" table", -2.8),
          c(" chair", -3.1),
          c(" roof", -3.5),
          ...controlTail(),
        ],
      },
      {
        response: " floor",
        candidates: [
          c(" floor", -0.7),
          c(" mat", -0.8),
          c(" couch", -1.1),
          c(" rug", -1.9),
          c(" bed", -2.2),
          c(" table", -2.7),
          c(" chair", -3.0),
          c(" roof", -3.4),
          ...controlTail(),
        ],
      },
    ],
  },
  {
    id: "robot",
    label: 'Preset: "Once upon a time the robot…"',
    prompt: "Once upon a time the robot",
    variants: [
      {
        response: " learned",
        candidates: [
          c(" learned", -0.4),
          c(" dreamed", -1.0),
          c(" woke", -1.5),
          c(" paused", -2.0),
          c(" whispered", -2.3),
          c(" calculated", -2.6),
          c(" wandered", -3.0),
          c(" hummed", -3.3),
          c(" slept", -3.7),
          c("", -4.1),
        ],
      },
      {
        response: " dreamed",
        candidates: [
          c(" dreamed", -0.6),
          c(" learned", -0.8),
          c(" woke", -1.2),
          c(" paused", -1.9),
          c(" whispered", -2.2),
          c(" calculated", -2.5),
          c(" wandered", -2.9),
          c(" hummed", -3.2),
          c(" slept", -3.6),
          c("", -4.0),
        ],
      },
    ],
  },
];

export type OfflineTraceKind =
  | "user"
  | "harness"
  | "llm"
  | "tool_call"
  | "tool_result"
  | "answer"
  | "error";

export interface OfflineTraceStep {
  kind: OfflineTraceKind;
  title: string;
  body: string;
  raw?: string;
}

// Canned agent runs for the four fixed tasks. The "Free chat" task has no
// offline trace and is hidden in demo mode.
export const OFFLINE_AGENT_TRACES: Record<string, OfflineTraceStep[]> = {
  math: [
    {
      kind: "user",
      title: "User request",
      body: "What is 27*14 + 8? Use the calculator tool.",
    },
    {
      kind: "tool_call",
      title: 'Tool call proposed: calculator (expression="27*14+8")',
      body: "Structured output from the LLM. Nothing has run yet — the harness is holding this proposal until it is approved.",
      raw: '{\n  "function": {\n    "name": "calculator",\n    "arguments": {\n      "expression": "27*14+8"\n    }\n  }\n}',
    },
    {
      kind: "llm",
      title: "LLM responds",
      body: "27*14 + 8 = 386. Computed with the calculator: 27*14 = 378, plus 8 is 386.",
    },
  ],
  multistep: [
    {
      kind: "user",
      title: "User request",
      body: "What day of the week is it today, and what is 96 divided by 12? Use calendar, then the calculator.",
    },
    {
      kind: "tool_call",
      title: "Tool call proposed: calendar",
      body: "Structured output from the LLM. Nothing has run yet — the harness is holding this proposal until it is approved.",
      raw: '{\n  "function": {\n    "name": "calendar",\n    "arguments": {}\n  }\n}',
    },
    {
      kind: "tool_call",
      title: 'Tool call proposed: calculator (expression="96/12")',
      body: "Structured output from the LLM. Nothing has run yet — the harness is holding this proposal until it is approved.",
      raw: '{\n  "function": {\n    "name": "calculator",\n    "arguments": {\n      "expression": "96/12"\n    }\n  }\n}',
    },
    {
      kind: "llm",
      title: "LLM responds",
      body: "Today is Thursday (demo data), and 96/12 = 8. I used the calendar tool for the weekday and the calculator for the division.",
    },
  ],
  notes: [
    {
      kind: "user",
      title: "User request",
      body: "Check notes.txt and tell me what I worked on yesterday.",
    },
    {
      kind: "tool_call",
      title: 'Tool call proposed: read_note (name="notes.txt")',
      body: "Structured output from the LLM. Nothing has run yet — the harness is holding this proposal until it is approved.",
      raw: '{\n  "function": {\n    "name": "read_note",\n    "arguments": {\n      "name": "notes.txt"\n    }\n  }\n}',
    },
    {
      kind: "llm",
      title: "LLM responds",
      body: 'From notes.txt: "Yesterday: drafted Q2 roadmap and prioritized onboarding fixes. Blocked on API rate limits, asked Priya for help."',
    },
  ],
  embeddings: [
    {
      kind: "user",
      title: "User request",
      body: "What sentences are currently on the Vector Embeddings tab? Use get_embedding_sentences and list them.",
    },
    {
      kind: "tool_call",
      title: "Tool call proposed: get_embedding_sentences",
      body: "Structured output from the LLM. Nothing has run yet — the harness is holding this proposal until it is approved.",
      raw: '{\n  "function": {\n    "name": "get_embedding_sentences",\n    "arguments": {}\n  }\n}',
    },
    {
      kind: "llm",
      title: "LLM responds",
      body: 'The Vector Embeddings tab has 4 sentences:\n1. "The cat sat on the mat"\n2. "A kitten curled up on the couch"\n3. "My cat loves to chase yarn"\n4. "Quantum computers use qubits to perform calculations"',
    },
  ],
};
