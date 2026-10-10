# Skinny AI

Skinny AI is an interactive learning app for the AI For TSCM course. Four tabs let you explore how AI works by experimenting with it:

| Tab | What you learn |
|---|---|
| **Neural Networks** | How inputs and weights combine into a decision (a tiny perceptron deciding whether to go to the movies). |
| **Text Generation** | How an LLM picks the next token. See the real probability of every candidate, and watch temperature, top-K and seed reshape the odds. |
| **Vector Embeddings** | How text becomes vectors. Real embeddings are projected to a 2D or 3D map; click two sentences to compare their cosine similarity and Euclidean distance. |
| **AI Agents** | How an agent works: an LLM (brain) + a harness (nerves) + tools (hands). Watch each tool call, approve or deny it, and see the result go back to the model. |

Everything runs locally. The text, embedding and agent tabs talk to [Ollama](https://ollama.com) on your own machine; nothing is sent to a cloud service.

## Requirements

- **Ollama** installed and running, for the live features (the Neural Networks tab and demo mode work without it). Models used in class:
  - a chat model with tool support, such as `gemma4` or `granite4.2` (Text Generation and AI Agents)
  - an embedding model, such as `nomic-embed-text` (Vector Embeddings)

  ```bash
  ollama pull gemma4
  ollama pull granite4.2
  ollama pull nomic-embed-text
  ```
- No internet connection is needed once everything is installed; the app and Ollama both run offline.
- **[Bun](https://bun.sh)** only if you want to run or build from source. Using the compiled app needs nothing else.

## Running the compiled app

The compiled app is a single file with the web interface built in, so it runs from any folder.

```bash
# macOS / Linux
./SkinnyAI
```

```powershell
# Windows
.\SkinnyAI.exe
```

Then open **http://localhost:3000** in your browser. Set the `PORT` environment variable to use a different port (for example `PORT=3100 ./SkinnyAI`).

If macOS blocks the file the first time because it was downloaded, clear the quarantine flag and run it again:

```bash
xattr -d com.apple.quarantine ./SkinnyAI
```

## Demo mode (no Ollama)

If Ollama is not running, the app still works. The Text Generation, Vector Embeddings and AI Agents tabs switch to **Demo Mode**: free-form input is locked and each tab plays back built-in example data so you can still see how the ideas work. Start Ollama and reload the page to get the live version.

## Running from source

```bash
bun install      # install dependencies
bun dev          # dev server with hot reload at http://localhost:3000
bun start        # production mode, uncompiled
```

## Building

Bundle the frontend only (output goes to `dist/`):

```bash
bun run build
```

Compile a standalone executable. Run these from the repo root and build on the same OS you are targeting (cross-compiling may not work):

```bash
# macOS (Apple Silicon)
bun build src/index.ts --compile --target=bun-darwin-arm64 --outfile=SkinnyAI

# macOS (Intel)
bun build src/index.ts --compile --target=bun-darwin-x64 --outfile=SkinnyAI

# Linux (x64)
bun build src/index.ts --compile --target=bun-linux-x64 --outfile=SkinnyAI

# Windows (x64)
bun build src/index.ts --compile --target=bun-windows-x64 --outfile=SkinnyAI.exe
```

The executable includes the frontend, so no `dist/` folder needs to ship with it.

## Troubleshooting

- **The model dropdowns are empty or the tabs say Demo Mode:** Ollama isn't reachable. Start it (open the Ollama app, or run `ollama serve`) and reload.
- **The first request takes a while:** Ollama is loading the model into memory. Later requests are much faster.
- **The AI Agents tab doesn't call tools:** the selected model must support tool calling. Try `gemma4` or `granite4.2`.
- **Port 3000 is already in use:** run with a different port, e.g. `PORT=3100 ./SkinnyAI`.

## Project notes

Developer and AI-agent notes (architecture, API routes, conventions) are in [CLAUDE.md](CLAUDE.md).
