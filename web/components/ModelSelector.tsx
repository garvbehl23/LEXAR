"use client";

import { useState, useRef, useEffect } from "react";
import { ChevronDown, Check, Cpu, Loader2 } from "lucide-react";
import { clsx } from "clsx";
import { getOllamaStatus } from "@/lib/api";
import { useChatContext } from "@/context/ChatContext";
import type { ModelType } from "@/types";

interface ModelOption {
  value:       ModelType;
  label:       string;
  description: string;
  subLabel?:   string;   // resolved Ollama model name
  unavailable?: boolean;
}

export function ModelSelector({ disabled }: { disabled?: boolean }) {
  const { state, setModel, setOllamaModel } = useChatContext();
  const [open,         setOpen]         = useState(false);
  const [ollamaStatus, setOllamaStatus] = useState<{ available: boolean; model: string }>({
    available: false,
    model: "",
  });
  const [loading, setLoading] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Probe Ollama whenever dropdown opens or model switches to ollama
  useEffect(() => {
    if (!open && state.settings.model !== "ollama") return;

    let cancelled = false;
    setLoading(true);
    getOllamaStatus().then((s) => {
      if (cancelled) return;
      setOllamaStatus({ available: s.available, model: s.selected ?? "" });
      if (s.available && s.selected) {
        setOllamaModel(s.selected);
      }
      setLoading(false);
    });
    return () => { cancelled = true; };
  }, [open, state.settings.model, setOllamaModel]);

  // Close on outside click
  useEffect(() => {
    function h(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const options: ModelOption[] = [
    {
      value:       "gemini",
      label:       "Gemini",
      description: "Google · fast & accurate",
    },
    {
      value:       "ollama",
      label:       "Ollama",
      description: loading ? "Checking..." :
                   ollamaStatus.available ? "Local · private" : "Not running",
      subLabel:    ollamaStatus.available ? ollamaStatus.model : undefined,
      unavailable: !loading && !ollamaStatus.available,
    },
    {
      value:       "flan-t5",
      label:       "Flan-T5",
      description: "Offline fallback",
    },
  ];

  const current = options.find((o) => o.value === state.settings.model) ?? options[0];

  return (
    <div ref={ref} className="relative flex-shrink-0">
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
        className={clsx(
          "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium",
          "border border-gray-200 bg-white text-gray-600",
          "hover:bg-gray-50 hover:border-gray-300 hover:text-gray-900",
          "transition-all duration-100 whitespace-nowrap",
          "disabled:opacity-40 disabled:cursor-not-allowed",
          open && "border-gray-400 bg-gray-50 text-gray-900"
        )}
      >
        {loading && state.settings.model === "ollama"
          ? <Loader2 className="w-3 h-3 animate-spin" />
          : <Cpu className="w-3 h-3" />
        }
        {current.label}
        {current.subLabel && (
          <span className="text-gray-400 font-normal">· {current.subLabel.split(":")[0]}</span>
        )}
        <ChevronDown className={clsx("w-3 h-3 transition-transform", open && "rotate-180")} />
      </button>

      {open && (
        <div className="absolute bottom-full mb-2 left-0 w-56 bg-white rounded-xl border border-gray-200 shadow-xl py-1.5 z-50 animate-fade-in">
          <p className="px-3 pt-0.5 pb-1.5 text-[10px] text-gray-400 uppercase tracking-widest font-medium">
            Model
          </p>
          {options.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => {
                setModel(opt.value);
                if (opt.value === "ollama" && ollamaStatus.model) {
                  setOllamaModel(ollamaStatus.model);
                }
                setOpen(false);
              }}
              className={clsx(
                "w-full flex items-start gap-2.5 px-3 py-2 text-left transition-colors",
                state.settings.model === opt.value ? "bg-gray-50" : "hover:bg-gray-50",
                opt.unavailable && "opacity-50"
              )}
            >
              <span className="mt-0.5 w-3.5 flex-shrink-0">
                {state.settings.model === opt.value && (
                  <Check className="w-3.5 h-3.5 text-gray-700" />
                )}
              </span>
              <span>
                <span className="block text-sm font-medium text-gray-900 leading-none mb-0.5">
                  {opt.label}
                  {opt.value === "ollama" && loading && (
                    <Loader2 className="inline ml-1.5 w-3 h-3 animate-spin text-gray-400" />
                  )}
                </span>
                <span className="block text-xs text-gray-400">{opt.description}</span>
                {opt.subLabel && (
                  <span className="block text-xs text-gray-500 font-medium mt-0.5">
                    {opt.subLabel}
                  </span>
                )}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
