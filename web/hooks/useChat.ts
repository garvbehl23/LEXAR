"use client";

import { useCallback, useRef } from "react";
import { useChatContext } from "@/context/ChatContext";
import { streamQuery } from "@/lib/streaming";
import type { MessageMeta, ThinkingPhase } from "@/types";

export function useChat() {
  const {
    state, activeChat,
    createChat, addMessage, appendToken, updateMessage, setPhase, setStreaming, newMessage,
  } = useChatContext();

  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (query: string) => {
      if (!query.trim() || state.isStreaming) return;

      let chatId = state.activeChatId;
      if (!chatId) {
        const chat = createChat();
        chatId = chat.id;
      }

      // User message
      const userMsg = newMessage("user", query);
      addMessage(chatId, userMsg);

      // Placeholder AI message (thinking state)
      const aiMsg = newMessage("assistant", "");
      aiMsg.status       = "streaming";
      aiMsg.thinkingPhase = "thinking";
      aiMsg.thinkingMsg   = "Analyzing legal context...";
      addMessage(chatId, aiMsg);

      setStreaming(true);
      abortRef.current?.abort();
      abortRef.current = new AbortController();

      let meta: MessageMeta = {};

      await streamQuery(
        query,
        state.settings,
        {
          onPhase(phase: ThinkingPhase, message: string) {
            setPhase(chatId!, aiMsg.id, phase, message);
          },
          onToken(token: string) {
            // Clear thinking phase on first token
            setPhase(chatId!, aiMsg.id, null, "");
            appendToken(chatId!, aiMsg.id, token);
          },
          onMeta(m: MessageMeta) {
            meta = m;
          },
          onDone() {
            updateMessage(chatId!, aiMsg.id, {
              status:        "complete",
              thinkingPhase: null,
              thinkingMsg:   "",
              meta,
            });
            setStreaming(false);
          },
          onError(err: string) {
            updateMessage(chatId!, aiMsg.id, {
              content:       err,
              status:        "error",
              thinkingPhase: null,
              thinkingMsg:   "",
              meta,
            });
            setStreaming(false);
          },
        },
        abortRef.current.signal
      );
    },
    [
      state.activeChatId, state.isStreaming, state.settings,
      createChat, addMessage, appendToken, updateMessage,
      setPhase, setStreaming, newMessage,
    ]
  );

  const stopStreaming = useCallback(() => {
    abortRef.current?.abort();
    setStreaming(false);
    if (state.activeChatId && activeChat) {
      const last = activeChat.messages.at(-1);
      if (last?.status === "streaming") {
        updateMessage(state.activeChatId, last.id, {
          status: "complete",
          thinkingPhase: null,
        });
      }
    }
  }, [state.activeChatId, activeChat, setStreaming, updateMessage]);

  return { sendMessage, stopStreaming };
}
