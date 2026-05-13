"use client";

import { Scale } from "lucide-react";
import { WELCOME_SUGGESTIONS } from "@/lib/constants";

interface Props {
  onSuggestion: (s: string) => void;
}

export function WelcomeScreen({ onSuggestion }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 py-16 text-center">
      <div className="w-14 h-14 rounded-2xl bg-gray-900 flex items-center justify-center mb-5 shadow-lg">
        <Scale className="w-7 h-7 text-white" />
      </div>

      <h1 className="text-3xl font-semibold text-gray-900 tracking-tight mb-2">
        LEXAR Legal AI
      </h1>
      <p className="text-gray-500 max-w-md text-sm leading-relaxed mb-10">
        Ask any question about Indian law. Answers are grounded in the Indian Penal
        Code, CrPC, IEA, and more — with citations.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
        {WELCOME_SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onSuggestion(s)}
            className="text-left text-sm text-gray-700 bg-white border border-gray-200 rounded-xl px-4 py-3.5 hover:border-gray-400 hover:bg-gray-50 transition-all duration-150 leading-snug shadow-sm"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}
