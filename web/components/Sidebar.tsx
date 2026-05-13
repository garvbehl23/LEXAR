"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import {
  Scale,
  MessageSquare,
  Upload,
  BarChart2,
  Info,
  Plus,
  Trash2,
  Settings,
  ChevronDown,
  Database,
} from "lucide-react";
import { clsx } from "clsx";
import { useChatContext } from "@/context/ChatContext";
import { INDEX_OPTIONS } from "@/lib/constants";
import type { IndexName } from "@/types";

const NAV = [
  { href: "/chat", label: "Chat", icon: MessageSquare },
  { href: "/upload", label: "Upload", icon: Upload },
  { href: "/evaluation", label: "Evaluation", icon: BarChart2 },
  { href: "/about", label: "About", icon: Info },
];

export function Sidebar() {
  const pathname = usePathname();
  const { state, createChat, deleteChat, setActiveChat, setIndex } = useChatContext();
  const [settingsOpen, setSettingsOpen] = useState(false);

  const sortedChats = Object.values(state.chats).sort(
    (a, b) => b.updatedAt - a.updatedAt
  );

  return (
    <aside className="fixed left-0 top-0 h-full w-64 bg-[#111111] flex flex-col border-r border-[#1c1c1c] z-40 select-none">
      {/* Logo */}
      <div className="flex items-center gap-2.5 px-4 py-5 flex-shrink-0">
        <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
          <Scale className="w-4 h-4 text-white" />
        </div>
        <span className="text-white font-semibold tracking-wide text-base">LEXAR</span>
      </div>

      {/* New Chat */}
      <div className="px-3 mb-3 flex-shrink-0">
        <button
          onClick={() => createChat()}
          className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg border border-[#2a2a2a] bg-[#1c1c1c] text-gray-200 text-sm font-medium hover:bg-[#252525] transition-colors"
        >
          <Plus className="w-4 h-4" />
          New chat
        </button>
      </div>

      {/* Chat History */}
      <div className="flex-1 overflow-y-auto px-2 space-y-px min-h-0">
        {sortedChats.length > 0 && (
          <p className="px-2 py-1.5 text-[10px] text-gray-600 uppercase tracking-widest font-medium">
            Recent
          </p>
        )}
        {sortedChats.map((chat) => (
          <div
            key={chat.id}
            className={clsx(
              "group flex items-center gap-2 px-2.5 py-2 rounded-lg cursor-pointer transition-colors",
              state.activeChatId === chat.id
                ? "bg-[#1c1c1c] text-white"
                : "text-gray-400 hover:bg-[#161616] hover:text-gray-200"
            )}
            onClick={() => setActiveChat(chat.id)}
          >
            <MessageSquare className="w-3.5 h-3.5 flex-shrink-0 opacity-50" />
            <span className="text-xs flex-1 truncate">
              {chat.title.length > 34 ? chat.title.slice(0, 34) + "…" : chat.title}
            </span>
            <button
              onClick={(e) => { e.stopPropagation(); deleteChat(chat.id); }}
              className="opacity-0 group-hover:opacity-100 text-gray-600 hover:text-red-400 transition-all"
            >
              <Trash2 className="w-3 h-3" />
            </button>
          </div>
        ))}
      </div>

      {/* Navigation */}
      <div className="px-2 py-2 border-t border-[#1c1c1c] flex-shrink-0">
        {NAV.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={clsx(
              "flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors mb-px",
              pathname === href
                ? "bg-[#1c1c1c] text-white font-medium"
                : "text-gray-500 hover:text-gray-200 hover:bg-[#161616]"
            )}
          >
            <Icon className="w-4 h-4 flex-shrink-0" />
            {label}
          </Link>
        ))}
      </div>

      {/* Settings — knowledge base only (model is now in InputBar) */}
      <div className="px-2 pb-3 border-t border-[#1c1c1c] pt-2 flex-shrink-0">
        <button
          onClick={() => setSettingsOpen((o) => !o)}
          className="w-full flex items-center gap-2 px-3 py-2 text-gray-500 hover:text-gray-300 text-sm transition-colors rounded-lg hover:bg-[#161616]"
        >
          <Settings className="w-4 h-4" />
          <span className="flex-1 text-left">Settings</span>
          <ChevronDown
            className={clsx(
              "w-3.5 h-3.5 transition-transform",
              settingsOpen && "rotate-180"
            )}
          />
        </button>

        {settingsOpen && (
          <div className="mt-2 px-2 pb-1">
            <div className="flex items-center gap-1.5 mb-1.5">
              <Database className="w-3 h-3 text-gray-600" />
              <label className="text-[10px] text-gray-600 uppercase tracking-widest font-medium">
                Knowledge Base
              </label>
            </div>
            <select
              value={state.settings.indexName}
              onChange={(e) => setIndex(e.target.value as IndexName)}
              className="w-full bg-[#1a1a1a] border border-[#2a2a2a] text-gray-300 text-xs rounded-lg px-2.5 py-1.5 outline-none focus:border-gray-600"
            >
              {INDEX_OPTIONS.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>
    </aside>
  );
}
