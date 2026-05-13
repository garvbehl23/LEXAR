"use client";

import { useState, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { MessageList } from "@/components/MessageList";
import { InputBar } from "@/components/InputBar";
import { UploadModal } from "@/components/UploadModal";
import { useChatContext } from "@/context/ChatContext";
import { useChat } from "@/hooks/useChat";
import type { UploadResult } from "@/types";

export default function ChatPage() {
  const { state, activeChat, createChat, addMessage, newMessage } = useChatContext();
  const { sendMessage, stopStreaming } = useChat();
  const [showUpload, setShowUpload] = useState(false);

  // Create a chat on first visit if none exist
  useEffect(() => {
    if (!state.activeChatId && Object.keys(state.chats).length === 0) {
      createChat();
    }
  }, [state.activeChatId, state.chats, createChat]);

  function handleUploadSuccess(result: UploadResult) {
    setShowUpload(false);
    // Inject a system notice into the active chat
    if (state.activeChatId) {
      const msg = newMessage(
        "assistant",
        `Document uploaded successfully: **${result.original_filename}**\n${result.num_chunks} chunks indexed. You can now ask questions about this document.`
      );
      addMessage(state.activeChatId, { ...msg, status: "complete" });
    }
  }

  const messages = activeChat?.messages ?? [];

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      <Sidebar />

      {/* Main area — offset by sidebar width */}
      <div className="flex flex-col flex-1 ml-64 h-full min-w-0">
        {/* Scrollable message area */}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-3xl mx-auto w-full h-full">
            <MessageList
              messages={messages}
              isStreaming={state.isStreaming}
              onSuggestion={(s) => sendMessage(s)}
            />
          </div>
        </div>

        {/* Fixed input bar */}
        <InputBar
          onSubmit={sendMessage}
          onUpload={() => setShowUpload(true)}
          isStreaming={state.isStreaming}
          onStop={stopStreaming}
        />
      </div>

      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onSuccess={handleUploadSuccess}
        />
      )}
    </div>
  );
}
