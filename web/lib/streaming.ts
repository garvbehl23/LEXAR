import type { Settings, StreamEvent, MessageMeta, ThinkingPhase, Evidence } from "@/types";

export interface StreamCallbacks {
  onToken:  (token: string) => void;
  onMeta:   (meta: MessageMeta) => void;
  onPhase:  (phase: ThinkingPhase, message: string) => void;
  onDone:   () => void;
  onError:  (err: string) => void;
}

// ── Error mapping ─────────────────────────────────────────────────────────────

function mapNetwork(err: Error): string {
  if (err.name === "AbortError") return "";
  return "Unable to connect to server. Is the backend running?";
}

function mapHttp(status: number): string {
  if (status === 429) return "Service is busy. Please try again shortly.";
  if (status === 503) return "Backend not available. Start the server.";
  if (status >= 500)  return "Something went wrong on the server.";
  return "An unexpected error occurred.";
}

function mapBackend(msg: string): string {
  const l = msg.toLowerCase();
  if (l.includes("quota") || l.includes("429"))
    return "Gemini quota exceeded. Switching to Ollama...";
  if (l.includes("api key") || l.includes("invalid_api_key"))
    return "Invalid API key. Check GEMINI_API_KEY in .env.";
  if (l.includes("ollama") || l.includes("connection refused"))
    return "Local model unavailable. Trying fallback...";
  if (l.includes("not ready") || l.includes("knowledge base"))
    return "Knowledge base not ready. Run data preparation.";
  if (l.includes("timeout"))
    return "Request timed out. Please try again.";
  // If it's already a friendly sentence, pass through
  if (msg.length < 120 && !msg.includes("Traceback") && !msg.includes("\n"))
    return msg;
  return "Something went wrong. Please try again.";
}

// ── Main ─────────────────────────────────────────────────────────────────────

export async function streamQuery(
  query:     string,
  settings:  Settings,
  callbacks: StreamCallbacks,
  signal?:   AbortSignal
): Promise<void> {
  const { onToken, onMeta, onPhase, onDone, onError } = callbacks;

  let response: Response;
  try {
    response = await fetch("/api/stream", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        index_name:   settings.indexName,
        model:        settings.model,
        ollama_model: settings.ollamaModel ?? "",
        top_k:        10,
        rerank_k:     5,
        has_user_docs: false,
      }),
      signal,
    });
  } catch (err) {
    const msg = mapNetwork(err as Error);
    if (msg) onError(msg);
    return;
  }

  if (!response.ok) {
    onError(mapHttp(response.status));
    return;
  }

  const reader = response.body?.getReader();
  if (!reader) {
    onError("Streaming not supported.");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;

        let event: StreamEvent;
        try { event = JSON.parse(raw); }
        catch  { continue; }

        switch (event.type) {
          case "phase":
            onPhase(event.phase ?? null, event.message ?? "");
            break;
          case "token":
            if (event.text) onToken(event.text);
            break;
          case "meta":
            onMeta({
              evidence_count: event.evidence_count,
              confidence:     event.confidence,
              evidence_ids:   event.evidence_ids,
              evidence:       event.evidence as Evidence[] | undefined,
            });
            break;
          case "error":
            onError(mapBackend(event.message ?? "Generation failed"));
            return;
          case "done":
            onDone();
            return;
        }
      }
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      onError("Connection interrupted. Please try again.");
    }
  } finally {
    try { reader.releaseLock(); } catch { /* ignore */ }
  }

  onDone();
}
