"use client";

import { useEffect, useRef } from "react";
import { ChatBubble } from "./ChatBubble";
import { WelcomeScreen } from "./WelcomeScreen";
import type { Message } from "@/types";

interface Props {
  messages:     Message[];
  isStreaming:  boolean;
  onSuggestion: (s: string) => void;
}

export function MessageList({ messages, isStreaming, onSuggestion }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  if (messages.length === 0) {
    return <WelcomeScreen onSuggestion={onSuggestion} />;
  }

  return (
    <div className="flex flex-col py-4 space-y-1">
      {messages.map((msg) => (
        <ChatBubble key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} className="h-4" />
    </div>
  );
}
