import { Sidebar } from "@/components/Sidebar";
import { Scale, Cpu, Database, Zap, Shield } from "lucide-react";

const FEATURES = [
  {
    icon: Database,
    title: "Multi-index Retrieval",
    desc: "Searches across IPC, CrPC, IEA, and combined indices using FAISS dense retrieval.",
  },
  {
    icon: Zap,
    title: "Real-time Streaming",
    desc: "Answers stream token-by-token via Server-Sent Events for instant feedback.",
  },
  {
    icon: Shield,
    title: "Evidence-Constrained",
    desc: "Generation is hard-constrained to retrieved legal evidence — no hallucination.",
  },
  {
    icon: Cpu,
    title: "Gemini & Local LLMs",
    desc: "Supports Gemini 2.0 Flash for cloud inference and Ollama for local privacy.",
  },
];

export default function AboutPage() {
  return (
    <div className="flex h-screen overflow-hidden bg-white">
      <Sidebar />
      <main className="flex-1 ml-64 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-12">
          <div className="flex items-center gap-4 mb-8">
            <div className="w-14 h-14 rounded-2xl bg-gray-900 flex items-center justify-center shadow-lg">
              <Scale className="w-7 h-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">LEXAR</h1>
              <p className="text-sm text-gray-500">Indian Legal AI · Evidence-Grounded RAG</p>
            </div>
          </div>

          <p className="text-gray-700 leading-relaxed mb-8 text-sm">
            LEXAR (Legal Evidence-eXtraction and Answer Reasoning) is a production-grade
            retrieval-augmented generation system for Indian law. It retrieves relevant
            statutory text, reranks by legal relevance, then generates answers constrained
            strictly to the retrieved evidence — with full citation tracking.
          </p>

          <h2 className="text-base font-semibold text-gray-800 mb-4">Core Features</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-10">
            {FEATURES.map(({ icon: Icon, title, desc }) => (
              <div key={title} className="border border-gray-200 rounded-xl p-5 shadow-sm">
                <div className="w-9 h-9 rounded-lg bg-gray-100 flex items-center justify-center mb-3">
                  <Icon className="w-5 h-5 text-gray-700" />
                </div>
                <h3 className="text-sm font-semibold text-gray-900 mb-1">{title}</h3>
                <p className="text-xs text-gray-500 leading-relaxed">{desc}</p>
              </div>
            ))}
          </div>

          <div className="border border-gray-200 rounded-xl p-5 bg-gray-50 text-sm text-gray-600">
            <p className="font-medium text-gray-800 mb-1">Legal Disclaimer</p>
            <p className="text-xs leading-relaxed">
              LEXAR is an AI research tool and does not provide legal advice. Answers may
              contain errors. Always consult a qualified legal professional for legal matters.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
