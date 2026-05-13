import type { IndexName, ModelType } from "@/types";

export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export const INDEX_OPTIONS: { value: IndexName; label: string }[] = [
  { value: "ipc", label: "IPC (Indian Penal Code)" },
  { value: "ipc_crpc", label: "IPC + CrPC" },
  { value: "ipc_crpc_iea", label: "IPC + CrPC + IEA" },
  { value: "lexar_medium", label: "LEXAR Medium (All Laws)" },
];

export const MODEL_OPTIONS: {
  value: ModelType;
  label: string;
  description: string;
}[] = [
  { value: "gemini", label: "Gemini", description: "Google · fast & accurate" },
  { value: "ollama", label: "Ollama", description: "Local · private, no API key" },
  { value: "flan-t5", label: "Flan-T5", description: "Offline fallback model" },
];

export const WELCOME_SUGGESTIONS = [
  "What is the punishment for theft under IPC?",
  "Explain the difference between murder and culpable homicide",
  "What are the rights of an arrested person under CrPC?",
  "What constitutes sexual harassment under Indian law?",
];
