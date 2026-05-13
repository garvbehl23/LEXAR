"use client";

import {
  createContext,
  useContext,
  useReducer,
  useEffect,
  useCallback,
  type ReactNode,
} from "react";
import { v4 as uuidv4 } from "uuid";
import type { Chat, Message, Settings, IndexName, ModelType, ThinkingPhase } from "@/types";

// ── State ─────────────────────────────────────────────────────────────────────

interface ChatState {
  chats:        Record<string, Chat>;
  activeChatId: string | null;
  isStreaming:  boolean;
  settings:     Settings;
}

// ── Actions ───────────────────────────────────────────────────────────────────

type Action =
  | { type: "LOAD_STATE";   payload: Partial<ChatState> }
  | { type: "CREATE_CHAT";  payload: Chat }
  | { type: "SET_ACTIVE";   payload: string }
  | { type: "DELETE_CHAT";  payload: string }
  | { type: "ADD_MESSAGE";  payload: { chatId: string; message: Message } }
  | { type: "UPDATE_MESSAGE"; payload: { chatId: string; messageId: string; updates: Partial<Message> } }
  | { type: "APPEND_TOKEN"; payload: { chatId: string; messageId: string; token: string } }
  | { type: "SET_PHASE";    payload: { chatId: string; messageId: string; phase: ThinkingPhase; msg: string } }
  | { type: "SET_STREAMING"; payload: boolean }
  | { type: "SET_MODEL";    payload: ModelType }
  | { type: "SET_INDEX";    payload: IndexName }
  | { type: "SET_OLLAMA_MODEL"; payload: string };

// ── Reducer ───────────────────────────────────────────────────────────────────

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "LOAD_STATE":
      return { ...state, ...action.payload };

    case "CREATE_CHAT":
      return {
        ...state,
        chats:        { ...state.chats, [action.payload.id]: action.payload },
        activeChatId: action.payload.id,
      };

    case "SET_ACTIVE":
      return { ...state, activeChatId: action.payload };

    case "DELETE_CHAT": {
      const { [action.payload]: _, ...rest } = state.chats;
      const ids = Object.keys(rest);
      return {
        ...state,
        chats:        rest,
        activeChatId:
          state.activeChatId === action.payload ? (ids[0] ?? null) : state.activeChatId,
      };
    }

    case "ADD_MESSAGE": {
      const chat = state.chats[action.payload.chatId];
      if (!chat) return state;
      const isFirstUser =
        action.payload.message.role === "user" &&
        chat.messages.filter((m) => m.role === "user").length === 0;
      return {
        ...state,
        chats: {
          ...state.chats,
          [action.payload.chatId]: {
            ...chat,
            title: isFirstUser ? action.payload.message.content.slice(0, 52) : chat.title,
            messages: [...chat.messages, action.payload.message],
            updatedAt: Date.now(),
          },
        },
      };
    }

    case "UPDATE_MESSAGE": {
      const chat = state.chats[action.payload.chatId];
      if (!chat) return state;
      return {
        ...state,
        chats: {
          ...state.chats,
          [action.payload.chatId]: {
            ...chat,
            messages: chat.messages.map((m) =>
              m.id === action.payload.messageId ? { ...m, ...action.payload.updates } : m
            ),
            updatedAt: Date.now(),
          },
        },
      };
    }

    case "APPEND_TOKEN": {
      const chat = state.chats[action.payload.chatId];
      if (!chat) return state;
      return {
        ...state,
        chats: {
          ...state.chats,
          [action.payload.chatId]: {
            ...chat,
            messages: chat.messages.map((m) =>
              m.id === action.payload.messageId
                ? { ...m, content: m.content + action.payload.token }
                : m
            ),
          },
        },
      };
    }

    case "SET_PHASE": {
      const chat = state.chats[action.payload.chatId];
      if (!chat) return state;
      return {
        ...state,
        chats: {
          ...state.chats,
          [action.payload.chatId]: {
            ...chat,
            messages: chat.messages.map((m) =>
              m.id === action.payload.messageId
                ? { ...m, thinkingPhase: action.payload.phase, thinkingMsg: action.payload.msg }
                : m
            ),
          },
        },
      };
    }

    case "SET_STREAMING":
      return { ...state, isStreaming: action.payload };

    case "SET_MODEL":
      return { ...state, settings: { ...state.settings, model: action.payload } };

    case "SET_INDEX":
      return { ...state, settings: { ...state.settings, indexName: action.payload } };

    case "SET_OLLAMA_MODEL":
      return { ...state, settings: { ...state.settings, ollamaModel: action.payload } };

    default:
      return state;
  }
}

// ── Initial state ─────────────────────────────────────────────────────────────

const INITIAL: ChatState = {
  chats:        {},
  activeChatId: null,
  isStreaming:  false,
  settings:     { model: "gemini", indexName: "ipc", ollamaModel: "" },
};

// ── Context ───────────────────────────────────────────────────────────────────

interface CtxValue {
  state:        ChatState;
  activeChat:   Chat | null;
  createChat:   () => Chat;
  deleteChat:   (id: string) => void;
  setActiveChat:(id: string) => void;
  addMessage:   (chatId: string, msg: Message) => void;
  updateMessage:(chatId: string, msgId: string, updates: Partial<Message>) => void;
  appendToken:  (chatId: string, msgId: string, token: string) => void;
  setPhase:     (chatId: string, msgId: string, phase: ThinkingPhase, msg: string) => void;
  setStreaming: (v: boolean) => void;
  setModel:     (m: ModelType) => void;
  setIndex:     (i: IndexName) => void;
  setOllamaModel: (m: string) => void;
  newMessage:   (role: Message["role"], content?: string) => Message;
}

const ChatCtx = createContext<CtxValue | null>(null);

export function ChatProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("lexar_state");
      if (raw) {
        const saved = JSON.parse(raw) as Partial<ChatState>;
        dispatch({ type: "LOAD_STATE", payload: saved });
      }
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem("lexar_state", JSON.stringify({
        chats:    state.chats,
        settings: state.settings,
      }));
    } catch {}
  }, [state.chats, state.settings]);

  const createChat = useCallback((): Chat => {
    const chat: Chat = {
      id:        uuidv4(),
      title:     "New conversation",
      messages:  [],
      createdAt: Date.now(),
      updatedAt: Date.now(),
    };
    dispatch({ type: "CREATE_CHAT", payload: chat });
    return chat;
  }, []);

  const deleteChat      = useCallback((id: string) => dispatch({ type: "DELETE_CHAT",  payload: id }), []);
  const setActiveChat   = useCallback((id: string) => dispatch({ type: "SET_ACTIVE",   payload: id }), []);
  const setStreaming     = useCallback((v: boolean) => dispatch({ type: "SET_STREAMING", payload: v }), []);
  const setModel        = useCallback((m: ModelType) => dispatch({ type: "SET_MODEL",   payload: m }), []);
  const setIndex        = useCallback((i: IndexName) => dispatch({ type: "SET_INDEX",   payload: i }), []);
  const setOllamaModel  = useCallback((m: string)    => dispatch({ type: "SET_OLLAMA_MODEL", payload: m }), []);

  const addMessage = useCallback(
    (chatId: string, msg: Message) =>
      dispatch({ type: "ADD_MESSAGE", payload: { chatId, message: msg } }),
    []
  );

  const updateMessage = useCallback(
    (chatId: string, msgId: string, updates: Partial<Message>) =>
      dispatch({ type: "UPDATE_MESSAGE", payload: { chatId, messageId: msgId, updates } }),
    []
  );

  const appendToken = useCallback(
    (chatId: string, msgId: string, token: string) =>
      dispatch({ type: "APPEND_TOKEN", payload: { chatId, messageId: msgId, token } }),
    []
  );

  const setPhase = useCallback(
    (chatId: string, msgId: string, phase: ThinkingPhase, msg: string) =>
      dispatch({ type: "SET_PHASE", payload: { chatId, messageId: msgId, phase, msg } }),
    []
  );

  const newMessage = useCallback(
    (role: Message["role"], content = ""): Message => ({
      id:        uuidv4(),
      role,
      content,
      status:    "pending",
      timestamp: Date.now(),
    }),
    []
  );

  const activeChat = state.activeChatId ? (state.chats[state.activeChatId] ?? null) : null;

  return (
    <ChatCtx.Provider
      value={{
        state, activeChat, createChat, deleteChat, setActiveChat,
        addMessage, updateMessage, appendToken, setPhase,
        setStreaming, setModel, setIndex, setOllamaModel, newMessage,
      }}
    >
      {children}
    </ChatCtx.Provider>
  );
}

export function useChatContext() {
  const ctx = useContext(ChatCtx);
  if (!ctx) throw new Error("useChatContext must be used inside <ChatProvider>");
  return ctx;
}
