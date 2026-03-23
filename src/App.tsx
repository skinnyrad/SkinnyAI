import React, { useState, useMemo } from "react";
import {
  Box,
  Tabs,
  Tab,
  AppBar,
  Toolbar,
  Typography,
  Container,
  CssBaseline,
  IconButton,
  Tooltip,
} from "@mui/material";
import { ThemeProvider, createTheme } from "@mui/material/styles";
import LightModeRoundedIcon from "@mui/icons-material/LightModeRounded";
import DarkModeRoundedIcon from "@mui/icons-material/DarkModeRounded";
import { NeuralNetworks } from "./NeuralNetworks";
import { TextGeneration } from "./TextGeneration";
import { VectorEmbeddings } from "./VectorEmbeddings";
import "./index.css";

const TABS = [
  { label: "Neural Networks",  component: <NeuralNetworks /> },
  { label: "Text Generation",  component: <TextGeneration /> },
  { label: "VECTOR EMBEDDINGS", component: <VectorEmbeddings /> },
];

export function App() {
  const [tab, setTab] = useState(0);
  const [mode, setMode] = useState<"light" | "dark">("dark");

  const theme = useMemo(() => createTheme({
    palette: {
      mode,
      primary: { main: "#26BDC0" },
    secondary: { main: "#FF5A00" },
      ...(mode === "light" && { background: { default: "#f4f6fa" } }),
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    },
    shape: { borderRadius: 10 },
  }), [mode]);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static" elevation={0} sx={{ background: "linear-gradient(90deg, #1a9ea1, #26BDC0)", color: "#000" }}>
        <Toolbar variant="dense" sx={{ gap: 2 }}>
          <Typography variant="h6" fontWeight="bold" sx={{ letterSpacing: 0.5, whiteSpace: "nowrap", color: "#000" }}>
            AI Basics
          </Typography>
          <Tabs
            value={tab}
            onChange={(_, v) => setTab(v)}
            textColor="inherit"
            indicatorColor="secondary"
            sx={{ "& .MuiTab-root": { fontWeight: 600, opacity: 0.7, color: "#000", "&.Mui-selected": { opacity: 1, color: "#000" } } }}
          >
            {TABS.map((t, i) => (
              <Tab key={i} label={t.label} icon={t.icon} iconPosition="start" />
            ))}
          </Tabs>
          <Box sx={{ flexGrow: 1 }} />
          <Tooltip title={mode === "dark" ? "Switch to light mode" : "Switch to dark mode"}>
            <IconButton onClick={() => setMode(m => m === "dark" ? "light" : "dark")} sx={{ color: "#000" }}>
              {mode === "dark" ? <LightModeRoundedIcon /> : <DarkModeRoundedIcon />}
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Box sx={{ bgcolor: "background.default" }}>
        <Container maxWidth="lg">
          {TABS[tab].component}
        </Container>
      </Box>
    </ThemeProvider>
  );
}

export default App;
