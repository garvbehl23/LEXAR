import { type NextRequest } from "next/server";

const BACKEND = process.env.BACKEND_URL ?? "http://localhost:8000";

export async function POST(req: NextRequest) {
  const formData = await req.formData();

  const upstream = await fetch(`${BACKEND}/upload/`, {
    method: "POST",
    body: formData,
  });

  const data = await upstream.json();
  return Response.json(data, { status: upstream.status });
}

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
