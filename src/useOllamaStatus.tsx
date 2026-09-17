// Ollama is considered unreachable when the model-list request fails with
// the backend's 502 "Could not reach Ollama on localhost:11434" error.
export function isOllamaUnreachableError(message: string | null): boolean {
  if (!message) return false;
  return message.includes("11434") || message.includes("Could not reach Ollama");
}

export function useOllamaOffline(modelsError: string | null): boolean {
  return isOllamaUnreachableError(modelsError);
}
