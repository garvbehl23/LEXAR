"use client";

import { useState, useRef, type KeyboardEvent } from "react";
import { Send, Paperclip, StopCircle } from "lucide-react";
import { ModelSelector } from "./ModelSelector";

interface Props {
  onSubmit: (query: string) => void;
  onUpload: () => void;
  isStreaming: boolean;
  onStop: () => void;
  disabled?: boolean;
}

export function InputBar({ onSubmit, onUpload, isStreaming, onStop, disabled }: Props) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);


  function submit() {
    const q = value.trim();
    if (!q || isStreaming) return;
    onSubmit(q);
    setValue("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function autoResize() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }

  return (
    <div className="border-t border-gray-200 bg-white px-4 py-3">
      <div className="max-w-3xl mx-auto">
        {/* Main input row */}
        <div className="flex items-end gap-2 bg-gray-50 border border-gray-200 rounded-2xl px-3 py-2 focus-within:border-gray-400 focus-within:bg-white transition-all">

          {/* ① Upload icon */}
          <button
            type="button"
            onClick={onUpload}
            disabled={isStreaming || disabled}
            title="Upload PDF"
            className="flex-shrink-0 mb-0.5 w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-gray-700 hover:bg-gray-100 transition-colors disabled:opacity-40"
          >
            <Paperclip className="w-4 h-4" />
          </button>

          {/* ② Model selector pill */}
          <div className="flex-shrink-0 mb-1 self-end">
            <ModelSelector disabled={isStreaming || disabled} />
          </div>

          {/* ③ Text input */}
          <textarea
            ref={textareaRef}
            value={value}
            onChange={(e) => { setValue(e.target.value); autoResize(); }}
            onKeyDown={onKeyDown}
            rows={1}
            placeholder="Ask a legal question…"
            disabled={disabled}
            className="flex-1 resize-none bg-transparent text-sm text-gray-900 placeholder-gray-400 outline-none leading-relaxed py-1.5 max-h-40 self-end"
          />

          {/* ④ Send / Stop */}
          {isStreaming ? (
            <button
              type="button"
              onClick={onStop}
              title="Stop"
              className="flex-shrink-0 mb-0.5 w-8 h-8 flex items-center justify-center rounded-lg bg-gray-900 text-white hover:bg-gray-700 transition-colors"
            >
              <StopCircle className="w-4 h-4" />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!value.trim() || disabled}
              title="Send"
              className="flex-shrink-0 mb-0.5 w-8 h-8 flex items-center justify-center rounded-lg bg-gray-900 text-white hover:bg-gray-700 disabled:opacity-30 transition-colors"
            >
              <Send className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        <p className="text-center text-[11px] text-gray-400 mt-2">
          LEXAR may make mistakes. Verify important legal information.
        </p>
      </div>
    </div>
  );
}
