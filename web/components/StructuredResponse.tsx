"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, Scale, BookOpen, Lightbulb } from "lucide-react";
import type { MessageMeta } from "@/types";
import { SourceCard } from "./SourceCard";

// ── Simple regex-based parser for the Gemini answer template ─────────────────
function parseAnswer(raw: string) {
  const numbered = raw.match(/[（(]?[123][）)][\s.:]?[\s\S]+?(?=[（(]?[234][）)]|$)/g);
  if (numbered && numbered.length >= 2) {
    const clean = (s: string) => s.replace(/^[（(]?[123][）)][\s.:]*/, "").trim();
    return {
      answer:      clean(numbered[0] ?? ""),
      sections:    clean(numbered[1] ?? ""),
      explanation: clean(numbered[2] ?? ""),
    };
  }
  return { answer: raw, sections: "", explanation: "" };
}

function extractCitations(text: string): string[] {
  const re =
    /\b(?:Section|Sec\.?|s\.)\s*\d+[A-Z]?(?:\(\d+\))?(?:\s+(?:of\s+)?(?:IPC|CrPC|IEA))?|\b(?:IPC|CrPC|IEA)\s+(?:Section|Sec\.?|s\.)\s*\d+[A-Z]?(?:\(\d+\))?/gi;
  return Array.from(new Set(text.match(re) ?? [])).slice(0, 8);
}

interface Props {
  content: string;
  meta?:   MessageMeta;
}

export function StructuredResponse({ content, meta }: Props) {
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const parsed    = parseAnswer(content);
  const citations = extractCitations(content);

  const hasEvidence     = (meta?.evidence?.length ?? 0) > 0;
  const hasEvidenceIds  = (meta?.evidence_ids?.length ?? 0) > 0;
  const evidenceCount   = meta?.evidence?.length ?? meta?.evidence_count ?? 0;

  return (
    <div className="space-y-3 text-sm text-gray-800 leading-relaxed">
      {/* Answer */}
      <p className="whitespace-pre-wrap">{parsed.answer}</p>

      {/* Citation tags */}
      {citations.length > 0 && (
        <div className="flex flex-wrap gap-1.5 pt-1">
          {citations.map((c) => (
            <span
              key={c}
              className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 text-xs font-medium border border-gray-200"
            >
              <Scale className="w-3 h-3 text-gray-500" />
              {c}
            </span>
          ))}
        </div>
      )}

      {/* Explanation */}
      {parsed.explanation && (
        <div className="pt-2 border-t border-gray-100">
          <div className="flex items-center gap-1.5 mb-1.5 text-xs font-medium text-gray-500">
            <Lightbulb className="w-3.5 h-3.5" />
            Explanation
          </div>
          <p className="whitespace-pre-wrap text-gray-700">{parsed.explanation}</p>
        </div>
      )}

      {/* Applicable sections */}
      {parsed.sections && parsed.sections !== parsed.answer && (
        <div className="pt-1">
          <div className="flex items-center gap-1.5 mb-1 text-xs font-medium text-gray-500">
            <BookOpen className="w-3.5 h-3.5" />
            Applicable sections
          </div>
          <p className="whitespace-pre-wrap text-gray-700">{parsed.sections}</p>
        </div>
      )}

      {/* Sources */}
      {(hasEvidence || hasEvidenceIds) && (
        <div className="pt-2 border-t border-gray-100">
          <button
            onClick={() => setSourcesOpen((o) => !o)}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-700 transition-colors"
          >
            {sourcesOpen ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
            {evidenceCount} source{evidenceCount !== 1 ? "s" : ""} retrieved
            {meta?.confidence !== undefined && (
              <span className="ml-2 px-1.5 py-px rounded bg-gray-100 text-gray-600">
                {Math.round(meta.confidence * 100)}% confidence
              </span>
            )}
          </button>

          {sourcesOpen && (
            <div className="mt-2 space-y-2">
              {hasEvidence
                ? meta!.evidence!.slice(0, 6).map((ev, i) => (
                    <SourceCard key={ev.chunk_id ?? i} evidence={ev} index={i} />
                  ))
                : meta?.evidence_ids?.slice(0, 5).map((id, i) => (
                    <div
                      key={id || i}
                      className="text-xs text-gray-500 pl-3 border-l-2 border-gray-200 py-0.5"
                    >
                      {id || `Source ${i + 1}`}
                    </div>
                  ))
              }
            </div>
          )}
        </div>
      )}
    </div>
  );
}
