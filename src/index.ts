import { serve } from "bun";
import index from "./index.html";

const server = serve({
  routes: {
    // Serve index.html for all unmatched routes.
    "/*": index,

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
