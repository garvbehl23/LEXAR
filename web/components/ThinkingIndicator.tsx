"use client";

import { clsx } from "clsx";
import type { ThinkingPhase } from "@/types";

const PHASES: { key: NonNullable<ThinkingPhase>; label: string }[] = [
  { key: "thinking",   label: "Analyzing" },
  { key: "retrieving", label: "Retrieving" },
  { key: "generating", label: "Generating" },
];

interface Props {
  phase: NonNullable<ThinkingPhase>;
  message?: string;
}

export function ThinkingIndicator({ phase, message }: Props) {
  const phaseIndex = PHASES.findIndex((p) => p.key === phase);

  return (
    <div className="space-y-3 py-1">
      {/* Phase stepper */}
      <div className="flex items-center gap-0">
        {PHASES.map((p, i) => {
          const done    = i < phaseIndex;
          const active  = i === phaseIndex;
          const pending = i > phaseIndex;
          return (
            <div key={p.key} className="flex items-center">
              {/* Node */}
              <div className="flex flex-col items-center gap-1">
                <div
                  className={clsx(
                    "w-2 h-2 rounded-full transition-all duration-500",
                    done    && "bg-gray-400",
                    active  && "bg-gray-700 ring-2 ring-gray-300 ring-offset-1 animate-pulse",
                    pending && "bg-gray-200"
                  )}
                />
                <span
                  className={clsx(
                    "text-[10px] font-medium whitespace-nowrap",
                    done    && "text-gray-400",
                    active  && "text-gray-700",
                    pending && "text-gray-300"
                  )}
                >
                  {p.label}
                </span>
              </div>
              {/* Connector */}
              {i < PHASES.length - 1 && (
                <div
                  className={clsx(
                    "w-8 h-px mb-3 transition-all duration-500",
                    i < phaseIndex ? "bg-gray-400" : "bg-gray-200"
                  )}
                />
              )}
            </div>
          );
        })}
      </div>

      {/* Shimmer message bar */}
      {message && (
        <div className="flex items-center gap-2">
          <div className="flex gap-0.5">
            {[0, 1, 2].map((i) => (
              <span
                key={i}
                className="block w-1 h-1 rounded-full bg-gray-400 animate-bounce"
                style={{ animationDelay: `${i * 120}ms` }}
              />
            ))}
          </div>
          <span className="text-xs text-gray-500">{message}</span>
        </div>
      )}

      {/* Progress shimmer bar */}
      <div className="h-0.5 rounded-full bg-gray-100 overflow-hidden w-48">
        <div className="h-full rounded-full progress-shimmer" />
      </div>
    </div>
  );
}
