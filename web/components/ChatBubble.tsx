import { AlertCircle } from "lucide-react";
import { StructuredResponse } from "./StructuredResponse";
import { ThinkingIndicator } from "./ThinkingIndicator";
import type { Message } from "@/types";

interface Props {
  message: Message;
}

export function ChatBubble({ message }: Props) {
  const isUser = message.role === "user";

  // ── User bubble ─────────────────────────────────────────────────────────────
  if (isUser) {
    return (
      <div className="flex justify-end px-4 py-1 animate-fade-in">
        <div className="max-w-[65%] bg-gray-900 text-gray-50 rounded-[18px_18px_6px_18px] px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap break-words">
          {message.content}
        </div>
      </div>
    );
  }

  const isStreaming   = message.status === "streaming";
  const isError       = message.status === "error";
  const thinkingPhase = message.thinkingPhase ?? null;
  const showThinking  = isStreaming && thinkingPhase !== null && message.content === "";

  // ── AI bubble ───────────────────────────────────────────────────────────────
  return (
    <div className="flex justify-start px-4 py-1 animate-fade-in">
      <div className="max-w-[78%] bg-white border border-gray-200 rounded-[18px_18px_18px_6px] px-5 py-4 shadow-sm">

        {isError ? (
          /* Error state */
          <div className="flex items-start gap-2.5 text-sm">
            <div className="w-8 h-8 rounded-full bg-red-50 flex items-center justify-center flex-shrink-0 mt-0.5">
              <AlertCircle className="w-4 h-4 text-red-500" />
            </div>
            <div>
              <p className="font-medium text-gray-800 mb-0.5">Unable to generate response</p>
              <p className="text-gray-500 leading-relaxed">{message.content}</p>
            </div>
          </div>

        ) : showThinking ? (
          /* Thinking / retrieving / generating phase — no tokens yet */
          <ThinkingIndicator
            phase={thinkingPhase!}
            message={message.thinkingMsg}
          />

        ) : isStreaming ? (
          /* Tokens arriving — raw text + blinking cursor */
          <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap break-words">
            {message.content}
            <span className="inline-block w-0.5 h-3.5 bg-gray-400 ml-0.5 align-middle animate-blink rounded-sm" />
          </p>

        ) : (
          /* Complete — structured view with citations + evidence */
          <StructuredResponse content={message.content} meta={message.meta} />
        )}

      </div>
    </div>
  );
}
