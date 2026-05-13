"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, BookOpen } from "lucide-react";
import { clsx } from "clsx";
import type { Evidence } from "@/types";

interface Props {
  evidence: Evidence;
  index:    number;
}

export function SourceCard({ evidence, index }: Props) {
  const [open, setOpen] = useState(false);

  const statute = evidence.statute ?? evidence.metadata?.statute as string ?? "";
  const section = evidence.section ?? evidence.metadata?.section as string ?? "";
  const score   = evidence.rerank_score ?? evidence.score;
  const snippet = evidence.text.slice(0, open ? 500 : 160);
  const truncated = evidence.text.length > 160 && !open;

  return (
    <div className="rounded-lg border border-gray-200 bg-gray-50 overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-start gap-2.5 px-3 py-2.5 text-left hover:bg-gray-100 transition-colors"
      >
        {/* Index badge */}
        <span className="flex-shrink-0 mt-0.5 w-5 h-5 rounded bg-gray-200 text-gray-600 text-[10px] font-semibold flex items-center justify-center">
          {index + 1}
        </span>

        <div className="flex-1 min-w-0">
          {/* Statute + section row */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {statute && (
              <span className="inline-flex items-center gap-1 text-xs font-semibold text-gray-700">
                <BookOpen className="w-3 h-3 text-gray-500" />
                {statute}
              </span>
            )}
            {section && (
              <span className="text-xs text-gray-500 font-medium">{section}</span>
            )}
            {score !== undefined && (
              <span className={clsx(
                "ml-auto text-[10px] px-1.5 py-px rounded font-medium",
                score >= 0.7 ? "bg-green-50 text-green-700" :
                score >= 0.4 ? "bg-yellow-50 text-yellow-700" :
                               "bg-gray-100 text-gray-500"
              )}>
                {Math.round(score * 100)}%
              </span>
            )}
          </div>

          {/* Text snippet */}
          <p className="mt-1 text-xs text-gray-600 leading-relaxed line-clamp-2">
            {snippet}{truncated && "…"}
          </p>
        </div>

        {/* Toggle icon */}
        <span className="flex-shrink-0 mt-0.5 text-gray-400">
          {open ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </span>
      </button>

      {/* Expanded text */}
      {open && (
        <div className="px-4 pb-3 pt-0 border-t border-gray-200">
          <p className="text-xs text-gray-700 leading-relaxed whitespace-pre-wrap">
            {evidence.text.slice(0, 500)}
            {evidence.text.length > 500 && "…"}
          </p>
        </div>
      )}
    </div>
  );
}
