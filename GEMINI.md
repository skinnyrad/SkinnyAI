# GEMINI.md - SkinnyAI Project Context

## Project Overview
**SkinnyAI** is a high-performance React application built using the [Bun](https://bun.sh) runtime and toolkit. It serves as an educational platform or template for exploring AI basics, such as neural network visualizations and interactive decision-making models.

### Core Technologies
- **Runtime & Tooling:** [Bun](https://bun.sh) (v1.3.9+) - Acts as the JavaScript runtime, package manager, bundler, and test runner.
- **Frontend Framework:** [React 19](https://react.dev) - Utilizes modern React features and concurrent rendering.
- **UI Components:** [Material UI (MUI)](https://mui.com) - Provides a consistent, accessible design system.
- **Styling:** Emotion (via MUI) and Vanilla CSS.
- **Language:** TypeScript for type safety and enhanced developer experience.

## Architecture
The project follows a "Bun-native" full-stack architecture, leveraging Bun's built-in server and bundling capabilities without external tools like Express or Vite.

- **Backend / Entry Point (`src/index.ts`):** Uses `Bun.serve()` to handle both static file serving (`index.html`) and API routes (e.g., `/api/hello`).
- **Frontend Shell (`src/index.html`):** A lightweight HTML entry point that imports the React application directly.
- **React Entry (`src/frontend.tsx`):** Bootstraps the React application into the DOM with Hot Module Replacement (HMR) support.
- **Main App (`src/App.tsx`):** Manages global state, theming (light/dark mode), and navigation (tabs).
- **Core Components:**
    - `NeuralNetworks.tsx`: An interactive visualization of a basic neural network (perceptron) showing inputs, weights, and decision logic.

## Building and Running

### Prerequisites
- [Bun](https://bun.sh) installed on your system.

### Key Commands
- **Install Dependencies:**
  ```bash
  bun install
  ```
- **Start Development Server:**
  ```bash
  bun dev
  ```
  *Runs `bun --hot src/index.ts` with live reloading.*
- **Production Build:**
  ```bash
  bun build ./src/index.html --outdir=dist --sourcemap --target=browser --minify --define:process.env.NODE_ENV='"production"' --env='BUN_PUBLIC_*'
  ```
- **Start Production Server:**
  ```bash
  bun start
  ```

## Development Conventions

### General Principles
- **Prefer Bun Built-ins:** Always use Bun's native APIs over external libraries where possible (e.g., `Bun.serve` instead of Express, `Bun.file` instead of `fs`).
- **No Vite/Webpack:** The project uses Bun's internal bundler for both development and production builds.
- **TypeScript First:** All new files should be written in TypeScript (`.ts` or `.tsx`).

### Styling & UI
- Use **Material UI (MUI)** for components to maintain visual consistency.
- Prefer the defined theme palette (Primary: `#26BDC0`, Secondary: `#FF5A00`).
- Use the **Material Symbols Rounded** icon set via the `material-symbols-rounded` CSS class.

### API Development
- Add new API routes directly in the `routes` object within `src/index.ts`.
- Routes support dynamic parameters (e.g., `/api/hello/:name`) and various HTTP methods.

### Testing
- Use `bun test` for running test suites. (Note: Currently, no tests are defined in the repository; follow the pattern in `CLAUDE.md` to add them).

## Project Structure
- `src/index.ts`: Bun server and API routing.
- `src/index.html`: Main HTML entry point.
- `src/frontend.tsx`: React mounting logic and HMR.
- `src/App.tsx`: Root React component and theming.
- `src/NeuralNetworks.tsx`: Neural network simulation component.
- `CLAUDE.md`: Contains detailed technical guidelines and "Cheat Sheet" for Bun-specific development.
