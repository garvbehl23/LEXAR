"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Scale, ArrowRight, Shield, Zap, BookOpen, FileSearch } from "lucide-react";

const FEATURES = [
  {
    icon: FileSearch,
    title:  "Evidence-Grounded",
    desc:   "Every answer cites exact sections from IPC, CrPC, and IEA",
  },
  {
    icon: Zap,
    title:  "Real-time Streaming",
    desc:   "Answers appear token-by-token, instantly",
  },
  {
    icon: Shield,
    title:  "Privacy Options",
    desc:   "Use Ollama locally — no data leaves your machine",
  },
  {
    icon: BookOpen,
    title:  "Multi-law Coverage",
    desc:   "IPC · CrPC · IEA · Combined index",
  },
];

export default function LandingPage() {
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    // Trigger animations after mount
    const t = setTimeout(() => setMounted(true), 60);
    return () => clearTimeout(t);
  }, []);

  return (
    <div className="min-h-screen bg-white flex flex-col overflow-hidden">
      {/* Background grid */}
      <div className="fixed inset-0 grid-bg opacity-60 pointer-events-none" />

      {/* Radial glow behind hero */}
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] rounded-full bg-gradient-radial from-gray-100/80 via-gray-50/40 to-transparent pointer-events-none" />

      {/* ── NAV ──────────────────────────────────────────────────────────── */}
      <nav className="relative z-10 flex items-center justify-between px-8 py-5">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-xl bg-gray-900 flex items-center justify-center shadow-sm">
            <Scale className="w-4 h-4 text-white" />
          </div>
          <span className="font-semibold text-gray-900 tracking-tight">LEXAR</span>
        </div>
        <button
          onClick={() => router.push("/chat")}
          className="text-sm text-gray-500 hover:text-gray-900 transition-colors"
        >
          Open app →
        </button>
      </nav>

      {/* ── HERO ─────────────────────────────────────────────────────────── */}
      <main className="relative z-10 flex flex-col items-center justify-center flex-1 px-6 pt-8 pb-20 text-center">
        {/* Floating icon */}
        <div
          className={`
            mb-8 transition-all duration-700
            ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"}
            animate-float
          `}
        >
          <div className="w-20 h-20 rounded-3xl bg-gray-900 flex items-center justify-center shadow-2xl shadow-gray-900/25">
            <Scale className="w-10 h-10 text-white" />
          </div>
        </div>

        {/* Title — gradient shimmer */}
        <h1
          className={`
            text-7xl sm:text-8xl font-extrabold tracking-tighter mb-3 text-shimmer
            transition-all duration-700 delay-100
            ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}
          `}
        >
          LEXAR
        </h1>

        {/* Tagline */}
        <p
          className={`
            text-2xl sm:text-3xl font-light text-gray-600 mb-4
            transition-all duration-700 delay-200
            ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}
          `}
        >
          Legal AI for India
        </p>

        {/* Subtitle */}
        <p
          className={`
            text-base text-gray-400 max-w-lg leading-relaxed mb-12
            transition-all duration-700 delay-300
            ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}
          `}
        >
          Evidence-grounded answers from the Indian Penal Code, Code of Criminal
          Procedure, and Indian Evidence Act — with full citations.
        </p>

        {/* CTA button */}
        <button
          onClick={() => router.push("/chat")}
          className={`
            btn-cta group flex items-center gap-2 px-8 py-4 bg-gray-900 text-white
            rounded-2xl font-semibold text-lg shadow-xl shadow-gray-900/20
            hover:shadow-2xl hover:shadow-gray-900/30
            hover:scale-105 active:scale-100
            transition-all duration-200
            delay-400
            ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}
          `}
          style={{
            transition: "all 0.2s ease, opacity 0.7s ease 0.4s, transform 0.7s ease 0.4s",
            opacity: mounted ? 1 : 0,
            transform: mounted ? "translateY(0)" : "translateY(8px)",
          }}
        >
          Start Chat
          <ArrowRight className="w-5 h-5 transition-transform group-hover:translate-x-1" />
        </button>

        {/* Stats row */}
        <div
          className={`
            mt-16 flex flex-wrap gap-10 justify-center
            transition-all duration-700 delay-500
            ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-8"}
          `}
        >
          {[
            { value: "3 Acts",   label: "Laws Indexed" },
            { value: "Cited",    label: "Every Answer" },
            { value: "3 LLMs",   label: "Model Options" },
            { value: "Real-time", label: "Streaming" },
          ].map(({ value, label }) => (
            <div key={label} className="text-center">
              <div className="text-2xl font-bold text-gray-900">{value}</div>
              <div className="text-sm text-gray-400 mt-0.5">{label}</div>
            </div>
          ))}
        </div>
      </main>

      {/* ── FEATURES GRID ────────────────────────────────────────────────── */}
      <section
        className={`
          relative z-10 max-w-4xl mx-auto px-6 pb-20
          grid grid-cols-2 sm:grid-cols-4 gap-4
          transition-all duration-700 delay-700
          ${mounted ? "opacity-100" : "opacity-0"}
        `}
      >
        {FEATURES.map(({ icon: Icon, title, desc }) => (
          <div
            key={title}
            className="bg-white border border-gray-200 rounded-2xl p-5 shadow-sm hover:shadow-md transition-shadow"
          >
            <div className="w-9 h-9 rounded-xl bg-gray-100 flex items-center justify-center mb-3">
              <Icon className="w-4.5 h-4.5 text-gray-700" strokeWidth={1.5} />
            </div>
            <p className="font-semibold text-sm text-gray-900 mb-1">{title}</p>
            <p className="text-xs text-gray-400 leading-relaxed">{desc}</p>
          </div>
        ))}
      </section>

      {/* Footer */}
      <footer className="relative z-10 text-center pb-6 text-xs text-gray-300">
        LEXAR — Indian Legal AI · Not legal advice
      </footer>
    </div>
  );
}
