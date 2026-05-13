export type Role          = "user" | "assistant";
export type MessageStatus = "pending" | "streaming" | "complete" | "error";
export type ThinkingPhase = "thinking" | "retrieving" | "generating" | null;
export type IndexName     = "ipc" | "ipc_crpc" | "ipc_crpc_iea" | "lexar_medium";
export type ModelType     = "gemini" | "ollama" | "flan-t5";

export interface Evidence {
  chunk_id?:    string;
  text:         string;
  section?:     string;
  statute?:     string;
  rerank_score?: number;
  score?:       number;
  metadata?:    Record<string, unknown>;
}

export interface MessageMeta {
  evidence_count?: number;
  confidence?:     number;
  evidence_ids?:   string[];
  evidence?:       Evidence[];
}

export interface Message {
  id:             string;
  role:           Role;
  content:        string;
  status:         MessageStatus;
  timestamp:      number;
  meta?:          MessageMeta;
  thinkingPhase?: ThinkingPhase;
  thinkingMsg?:   string;
}

export interface Chat {
  id:        string;
  title:     string;
  messages:  Message[];
  createdAt: number;
  updatedAt: number;
}

export interface Settings {
  model:       ModelType;
  indexName:   IndexName;
  ollamaModel: string;   // specific resolved Ollama model name
}

export interface StreamEvent {
  type:            "meta" | "token" | "done" | "error" | "phase";
  text?:           string;
  message?:        string;
  phase?:          ThinkingPhase;
  evidence_count?: number;
  confidence?:     number;
  evidence_ids?:   string[];
  evidence?:       Evidence[];
}

export interface OllamaStatus {
  available: boolean;
  models:    string[];
  selected:  string | null;
}

export interface UploadResult {
  document_id:       string;
  original_filename: string;
  size_mb:           number;
  text_length:       number;
  num_chunks:        number;
  chunks_path:       string;
  status:            string;
}
