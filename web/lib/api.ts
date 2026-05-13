import type { OllamaStatus, UploadResult } from "@/types";

// ── Upload ────────────────────────────────────────────────────────────────────

export async function uploadFile(file: File): Promise<UploadResult> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch("/api/upload", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail ?? `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Ollama ────────────────────────────────────────────────────────────────────

export async function getOllamaStatus(): Promise<OllamaStatus> {
  try {
    const res = await fetch("/api/ollama-models", { cache: "no-store" });
    if (!res.ok) return { available: false, models: [], selected: null };
    return res.json();
  } catch {
    return { available: false, models: [], selected: null };
  }
}

// ── Health ────────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch("/api/health", { cache: "no-store" });
    return res.ok;
  } catch {
    return false;
  }
}
