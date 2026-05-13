import { type NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

const SSE_HEADERS = {
  "Content-Type": "text/event-stream",
  "Cache-Control": "no-cache",
  "X-Accel-Buffering": "no",
  Connection: "keep-alive",
};

function makeErrorSSE(message: string): ReadableStream<Uint8Array> {
  const enc = new TextEncoder();
  return new ReadableStream({
    start(ctrl) {
      ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ type: "error", message })}\n\n`));
      ctrl.enqueue(enc.encode(`data: ${JSON.stringify({ type: "done" })}\n\n`));
      ctrl.close();
    },
  });
}

function mapHttpStatus(status: number): string {
  if (status === 429) return "Service is busy. Please try again shortly.";
  if (status === 503) return "Backend is not available. Is the server running?";
  if (status === 401 || status === 403) return "Authentication error. Check your API key.";
  if (status >= 500) return "Something went wrong on the server. Please try again.";
  return "An unexpected error occurred. Please try again.";
}

export async function POST(req: NextRequest) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return new Response(makeErrorSSE("Invalid request format."), {
      status: 200,
      headers: SSE_HEADERS,
    });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND}/stream/`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      // Node.js 18 needs this for request body streaming
      // @ts-expect-error -- not in @types/node yet
      duplex: "half",
    });
  } catch {
    // Backend completely unreachable
    return new Response(
      makeErrorSSE("Unable to connect to backend. Is the server running?"),
      { status: 200, headers: SSE_HEADERS }
    );
  }

  if (!upstream.ok) {
    return new Response(makeErrorSSE(mapHttpStatus(upstream.status)), {
      status: 200,
      headers: SSE_HEADERS,
    });
  }

  // Pass the SSE stream through directly
  return new Response(upstream.body, { status: 200, headers: SSE_HEADERS });
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
