# Copilot Instructions

## Commands

```bash
bun install          # install dependencies
bun dev              # dev server with HMR at http://localhost:3000
bun start            # production server
bun build ./src/index.html --outdir=dist  # production build
bun test             # run tests
bun test src/foo.test.ts  # run a single test file
```

## Architecture

This is a full-stack Bun + React 19 app. The Bun server (`src/index.ts`) uses `Bun.serve()` with native file-based routing — it imports `src/index.html` directly and serves it for all unmatched routes (`"/*"`). Bun's bundler auto-transpiles any `.tsx`/`.ts` files referenced from HTML.

**Request flow:**
`src/index.ts` (Bun server) → `src/index.html` → `src/frontend.tsx` (React entry) → `src/App.tsx` (root component)

**Adding a new tab/view:** `App.tsx` maintains a `TABS` array. Add an entry with `{ label, icon, component }` — no routing or lazy-loading needed, the active tab's component renders directly.

**API routes** are defined inline in `src/index.ts` using `Bun.serve({ routes: { "/api/...": { GET, POST, ... } } })`.

## Key Conventions

- **Bun-native APIs only**: use `Bun.serve()` (not express), `bun:sqlite` (not better-sqlite3), `Bun.file` (not fs.readFile), `Bun.$` (not execa)
- **No dotenv**: Bun loads `.env` automatically
- **Public env vars** must be prefixed `BUN_PUBLIC_*` to be exposed to the browser
- **Path alias**: `@/*` resolves to `./src/*`
- **TypeScript**: strict mode + `verbatimModuleSyntax` — always use `import type` for type-only imports
- **UI library**: MUI v7 with Emotion. Theme is created in `App.tsx` via `createTheme`. Brand colors: primary `#26BDC0`, secondary `#FF5A00`
- **HMR pattern**: `frontend.tsx` uses `import.meta.hot` to persist the React root across hot reloads — preserve this pattern in the entry file
- **No CSS modules**: global styles in `src/index.css`, component styles via MUI `sx` prop
