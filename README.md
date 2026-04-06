# Skinny AI Learning Application

To install dependencies:

```bash
bun install
```

To start a development server:

```bash
bun dev
```

To run for production:

```bash
bun start
```

## Compilation

Here are the **Bun compile commands** for each OS (run from the repo root).

```bash
# Windows (x64)
bun build src/index.ts --compile --target=bun-windows-x64 --outfile=SkinnyAI.exe
```

```bash
# Linux (x64)
bun build src/index.ts --compile --target=bun-linux-x64 --outfile=SkinnyAI
```

```bash
# macOS (Apple Silicon / M1-M3)
bun build src/index.ts --compile --target=bun-darwin-arm64 --outfile=SkinnyAI
```

```bash
# macOS (Intel x64)
bun build src/index.ts --compile --target=bun-darwin-x64 --outfile=SkinnyAI
```

Notes:
- You generally need to **build on the same OS** you’re targeting (cross-compiling may not work depending on your setup).
- The resulting executable will still need your built frontend assets (e.g., a `dist/` folder) alongside it, unless you embed assets separately.
