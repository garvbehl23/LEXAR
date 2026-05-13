import type { OllamaStatus } from "@/types";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8001";

export async function GET() {
  try {
    const res = await fetch(`${BACKEND}/ollama/models`, {
      cache: "no-store",
      signal: AbortSignal.timeout(6000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = (await res.json()) as OllamaStatus;
    return Response.json(data);
  } catch {
    const fallback: OllamaStatus = { available: false, models: [], selected: null };
    return Response.json(fallback);
  }
}

export const dynamic = "force-dynamic";
