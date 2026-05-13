const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const res = await fetch(`${BACKEND}/health/`, { cache: "no-store" });
    const data = await res.json();
    return Response.json(data, { status: res.status });
  } catch {
    return Response.json({ status: "backend_unreachable" }, { status: 503 });
  }
}

export const dynamic = "force-dynamic";
