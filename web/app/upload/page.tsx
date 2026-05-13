"use client";

import { useState } from "react";
import { Sidebar } from "@/components/Sidebar";
import { UploadModal } from "@/components/UploadModal";
import { FileText, Upload, Plus } from "lucide-react";
import type { UploadResult } from "@/types";

export default function UploadPage() {
  const [showModal, setShowModal] = useState(false);
  const [uploads, setUploads] = useState<UploadResult[]>([]);

  function handleSuccess(r: UploadResult) {
    setUploads((prev) => [r, ...prev]);
    setShowModal(false);
  }

  return (
    <div className="flex h-screen overflow-hidden bg-white">
      <Sidebar />
      <main className="flex-1 ml-64 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-6 py-12">
          <h1 className="text-2xl font-semibold text-gray-900 mb-1">Document Upload</h1>
          <p className="text-gray-500 text-sm mb-8">
            Upload PDF documents to query against alongside the built-in legal indices.
          </p>

          <button
            onClick={() => setShowModal(true)}
            className="flex items-center gap-2 px-5 py-2.5 bg-gray-900 text-white rounded-xl text-sm font-medium hover:bg-gray-800 transition-colors mb-10"
          >
            <Plus className="w-4 h-4" />
            Upload PDF
          </button>

          {uploads.length > 0 ? (
            <div className="space-y-3">
              <h2 className="text-xs font-medium text-gray-500 uppercase tracking-widest">
                Uploaded this session
              </h2>
              {uploads.map((u) => (
                <div
                  key={u.document_id}
                  className="flex items-start gap-3 bg-white border border-gray-200 rounded-xl px-4 py-3.5 shadow-sm"
                >
                  <FileText className="w-5 h-5 text-gray-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-gray-800">{u.original_filename}</p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {u.num_chunks} chunks · {u.size_mb.toFixed(2)} MB
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center text-center py-16 text-gray-400">
              <Upload className="w-10 h-10 mb-3 opacity-30" />
              <p className="text-sm">No documents uploaded yet</p>
            </div>
          )}
        </div>
      </main>

      {showModal && (
        <UploadModal onClose={() => setShowModal(false)} onSuccess={handleSuccess} />
      )}
    </div>
  );
}
