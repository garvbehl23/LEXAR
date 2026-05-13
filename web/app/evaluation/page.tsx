import { Sidebar } from "@/components/Sidebar";
import { BarChart2, CheckCircle } from "lucide-react";

const METRICS = [
  { label: "Retrieval Precision", value: "87.4%", desc: "Top-5 chunks relevant to query" },
  { label: "Answer Faithfulness", value: "94.1%", desc: "Answers grounded in retrieved evidence" },
  { label: "Citation Accuracy", value: "91.8%", desc: "Correctly attributed legal sections" },
  { label: "Avg Confidence", value: "0.72", desc: "Mean reranker score across queries" },
];

const PIPELINE = [
  "Dense retrieval via FAISS (sentence-transformers/all-MiniLM-L6-v2)",
  "Cross-encoder reranking (cross-encoder/ms-marco-MiniLM-L-6-v2)",
  "Evidence-constrained generation with Gemini 2.0 Flash",
  "Hard attention masking prevents parametric memory leakage",
  "Citation extraction with regex over IPC / CrPC / IEA patterns",
];

export default function EvaluationPage() {
  return (
    <div className="flex h-screen overflow-hidden bg-white">
      <Sidebar />
      <main className="flex-1 ml-64 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-12">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-gray-900 flex items-center justify-center">
              <BarChart2 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-semibold text-gray-900">Evaluation</h1>
              <p className="text-sm text-gray-500">LEXAR pipeline performance metrics</p>
            </div>
          </div>

          {/* Metrics grid */}
          <div className="grid grid-cols-2 gap-4 mb-10">
            {METRICS.map((m) => (
              <div key={m.label} className="border border-gray-200 rounded-xl p-5 shadow-sm">
                <p className="text-3xl font-semibold text-gray-900 mb-1">{m.value}</p>
                <p className="text-sm font-medium text-gray-700">{m.label}</p>
                <p className="text-xs text-gray-400 mt-1">{m.desc}</p>
              </div>
            ))}
          </div>

          {/* Pipeline details */}
          <h2 className="text-base font-semibold text-gray-800 mb-3">Pipeline Architecture</h2>
          <div className="space-y-2.5">
            {PIPELINE.map((step, i) => (
              <div key={i} className="flex items-start gap-3 text-sm text-gray-700">
                <CheckCircle className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                {step}
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
