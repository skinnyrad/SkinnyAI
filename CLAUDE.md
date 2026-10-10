# CLAUDE.md

Guidance for AI coding agents working in this repository.

## Overview

**Skinny AI** is a classroom web app for the AI For TSCM course. It teaches four AI ideas with interactive tabs: neural networks, next-token text generation, vector embeddings, and tool-using agents. A small Bun server serves the React frontend and proxies a few endpoints to a local **Ollama** (`http://localhost:11434`). When Ollama is not running, the Ollama-backed tabs fall back to a built-in **demo mode**.

Stack: Bun (runtime, bundler, package manager) · React 19 · TypeScript · MUI 7 + Emotion. No Express, Vite or Node. The app ships as a single self-contained compiled binary.

## Commands

```bash
bun install          # install dependencies
bun dev              # dev server with HMR (bun --hot src/index.ts), http://localhost:3000
bun start            # production mode, uncompiled (NODE_ENV=production bun src/index.ts)
bun run build        # bundle the frontend to dist/ (dist/ is git-ignored)
```

The server listens on port 3000; set `PORT` to change it.

Compile a standalone binary (run from the repo root, build on the OS you are targeting):

```bash
bun build src/index.ts --compile --target=bun-darwin-arm64 --outfile=SkinnyAI   # macOS Apple Silicon
bun build src/index.ts --compile --target=bun-darwin-x64   --outfile=SkinnyAI   # macOS Intel
bun build src/index.ts --compile --target=bun-linux-x64    --outfile=SkinnyAI   # Linux x64
bun build src/index.ts --compile --target=bun-windows-x64  --outfile=SkinnyAI.exe # Windows x64
```

Run the binary with `./SkinnyAI` (add `PORT=<n>` to change the port). The frontend bundle and the favicon are embedded in the binary, so it runs from any folder (no `dist/` needed). The compiled `SkinnyAI` file is untracked; do not commit it.

There is no test suite and no lint script. `tsc --noEmit` reports a handful of existing strict-mode errors (mostly `noUncheckedIndexedAccess` in `NeuralNetworks.tsx`); do not add new ones.

## Layout

```
src/index.ts              Bun.serve() backend: static routes + all /api routes (~660 lines)
src/index.html            HTML entry; Bun's bundler transpiles everything it imports
src/frontend.tsx          React root (keeps the import.meta.hot root-persistence pattern)
src/App.tsx               Theme (dark default, light toggle) and the TABS array
src/NeuralNetworks.tsx    Tab 1: perceptron visualization (pure client side)
src/TextGeneration.tsx    Tab 2: next-token probabilities via Ollama logprobs
src/VectorEmbeddings.tsx  Tab 3: real embeddings, PCA to 2D/3D, similarity
src/AIAgents.tsx          Tab 4: LLM + harness + tools tool-calling loop
src/useOllamaStatus.tsx   Detects "Ollama unreachable" from the model-list error
src/offlineFallbacks.ts   Hardcoded demo data used only when Ollama is unreachable
src/cachedNomicEmbeddings.ts  Precomputed nomic-embed-text vectors for the default sentences
src/APITester.tsx         Leftover template component, not used by any tab
src/skinny_ai.png         Favicon / branding
```

## Architecture

**Server (`src/index.ts`)** imports `index.html` and serves it for every unmatched path (`"/*"`). `Bun.serve({ routes })` defines the API inline. `development` (HMR + browser console echo) is enabled whenever `NODE_ENV !== "production"`.

**Tabs.** `App.tsx` keeps a `TABS` array of `{ label, node }`; the active tab's component renders directly (no router). Add a tab by adding an entry there.

**API routes** (the server never calls any cloud service; everything goes to local Ollama):

| Route | Purpose |
|---|---|
| `GET /api/ollama/tags` | Proxy of Ollama's model list. Returns 502 `{ error }` when Ollama is unreachable. |
| `POST /api/ollama/generate` | One-token completion (`raw: true`, `num_predict: 1`) with `logprobs`/`top_logprobs`. Returns `{ response, candidates[] }`. Takes `model, prompt, temperature, top_k, topLogprobs, seed`. |
| `POST /api/ollama/embed` | Proxy of Ollama embeddings. Takes `model, inputs[]`. |
| `POST /api/ollama/agent/step` | One LLM turn with tool schemas; returns the assistant message (text and/or `tool_calls`). Used by the UI. |
| `POST /api/ollama/agent/exec` | Runs one approved tool server-side. Takes `name, args, allowedTools`. Used by the UI. |
| `POST /api/ollama/agent` | Server-side full tool-calling loop returning `{ trace, final }`. Not used by the UI; useful for scripting. |
| `GET/POST /api/embeddings/sentences` | The Vector Embeddings tab syncs its sentence list here so the agent tool `get_embedding_sentences` can read it. |
| `GET /skinny_ai.png` | Favicon (explicit route so the `/*` catch-all does not swallow it). |
| `/api/hello`, `/api/hello/:name` | Template examples. |

**Agent harness.** The UI drives the loop: it calls `/agent/step`, shows each proposed tool call, optionally waits for the user to Approve/Deny ("Ask Before Tool Calls"), runs approved calls through `/agent/exec`, and feeds results back until the model answers or max steps is reached. Tools live in `AGENT_TOOL_DEFS` / `executeAgentTool` in `index.ts`: `calculator` (digits and `+-*/().%^` only, anything else is refused), `clock`, `calendar`, `read_note` (bundled `notes.txt` / `roadmap.txt`), and `get_embedding_sentences`. To add a tool, add its schema to `AGENT_TOOL_DEFS` and a branch in `executeAgentTool`.

**Model selection.** Each tab fetches `/api/ollama/tags` and filters by the `capabilities` Ollama reports (`completion` for Text Generation, `embedding` for Vector Embeddings). Text Generation defaults to the smallest model by parameter count. The agent needs a tool-capable model.

**Demo mode.** If the model-list request fails with the "Could not reach Ollama" error, the Ollama tabs lock free-form input and play back canned data from `offlineFallbacks.ts` / `cachedNomicEmbeddings.ts` instead of calling `/api/ollama/*`. Temperature/top-K sliders still reshape the canned distributions. Keep new Ollama features degrading the same way, and never call `/api/ollama/*` from the UI when offline.

**Neural Networks tab.** Output = `sum(((input - 1) / 9) * weight) / 3` for inputs 1-10 and weights -1 to 1; positive means "Going" and the magnitude is the confidence.

## Conventions

- **Bun-native only:** `Bun.serve()` (not Express), `Bun.file` (not `fs.readFile`), `bun:sqlite`, `Bun.$`, `bun test`, `bun build`, `bunx`. Use `bun install` / `bun run`, never npm/yarn/pnpm. Bun loads `.env` automatically (no dotenv); browser-visible env vars must be named `BUN_PUBLIC_*`.
- **Frontend:** import `.tsx`/`.css` straight from the HTML; no Vite, no CSS modules. Global styles are in `src/index.css`; component styling uses MUI's `sx` prop.
- **Theme:** created in `App.tsx`; primary `#26BDC0`, secondary `#FF5A00`; dark mode is the default.
- **TypeScript:** `strict`, `noUncheckedIndexedAccess`, `verbatimModuleSyntax` (use `import type` for types). Path alias `@/*` maps to `src/*`. Module declarations for non-code imports (`*.svg`, `*.png`, `*.module.css`) live in `bun-env.d.ts`.
- **API errors:** return `{ error: string }` JSON with a meaningful status (400 for bad input, 502 when Ollama is unreachable). The frontend's offline detection depends on the "Could not reach Ollama" message and the `11434` port, so keep that wording.
- **Ollama host** `http://localhost:11434` is hardcoded in `index.ts` in several places.

## Gotchas

- **Embedded assets in the compiled binary:** read static files through an `import ... with { type: "file" }` path (see the favicon route). `Bun.file(new URL("./x", import.meta.url))` works under `bun start` but throws `ENOENT` in a compiled binary.
- **Unused font link:** `index.html` links the Material Symbols font from `fonts.googleapis.com`, but no component uses it (all UI icons are bundled `@mui/icons-material` SVGs), so the app works fully offline. The link can be removed.
- **Dev mode:** the browser may log a failed `ws://localhost:3000/_bun/hmr` WebSocket in some environments; this is pre-existing and does not affect the app.
- **First model call is slow** because Ollama loads the model into memory (several seconds for gemma4); the UI shows a spinner/"embedding..." state meanwhile.
- The agent `calendar`/`clock` tools report the **server's** local time, not the browser's.

## Prerequisites for the live features

Ollama running locally (`ollama serve`) with at least one chat model that supports tools, and an embedding model such as `nomic-embed-text`. The course machines use `gemma4`, `granite4.2`, `nomic-embed-text` and `all-minilm`. Without Ollama the app still runs in demo mode.
