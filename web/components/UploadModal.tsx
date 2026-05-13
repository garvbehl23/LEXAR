"use client";

import { useState, useRef, type DragEvent } from "react";
import { X, Upload, FileText, CheckCircle, AlertCircle } from "lucide-react";
import { uploadFile } from "@/lib/api";
import type { UploadResult } from "@/types";

type UploadState = "idle" | "uploading" | "success" | "error";

interface Props {
  onClose: () => void;
  onSuccess: (result: UploadResult) => void;
}

export function UploadModal({ onClose, onSuccess }: Props) {
  const [uploadState, setUploadState] = useState<UploadState>("idle");
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleFile(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are accepted.");
      setUploadState("error");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setError("File must be under 10 MB.");
      setUploadState("error");
      return;
    }

    setUploadState("uploading");
    setError("");
    try {
      const res = await uploadFile(file);
      setResult(res);
      setUploadState("success");
      setTimeout(() => onSuccess(res), 1200);
    } catch (err) {
      setError((err as Error).message);
      setUploadState("error");
    }
  }

  function onDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-md mx-4 overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100">
          <h2 className="text-base font-semibold text-gray-900">Upload Document</h2>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg hover:bg-gray-100 text-gray-500 hover:text-gray-700 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-6">
          {uploadState === "success" && result ? (
            <div className="flex flex-col items-center gap-3 py-4 text-center animate-fade-in">
              <CheckCircle className="w-12 h-12 text-green-500" />
              <p className="font-medium text-gray-900">{result.original_filename}</p>
              <p className="text-sm text-gray-500">
                {result.num_chunks} chunks indexed · {result.size_mb.toFixed(2)} MB
              </p>
            </div>
          ) : uploadState === "uploading" ? (
            <div className="flex flex-col items-center gap-3 py-8 text-center">
              <div className="w-10 h-10 border-4 border-gray-200 border-t-gray-900 rounded-full animate-spin" />
              <p className="text-sm text-gray-600">Processing document…</p>
            </div>
          ) : (
            <>
              <div
                onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                onDragLeave={() => setDragging(false)}
                onDrop={onDrop}
                onClick={() => inputRef.current?.click()}
                className={`
                  flex flex-col items-center gap-3 py-10 rounded-xl border-2 border-dashed cursor-pointer transition-all
                  ${dragging
                    ? "border-gray-900 bg-gray-50"
                    : "border-gray-200 hover:border-gray-400 hover:bg-gray-50"}
                `}
              >
                <Upload className="w-8 h-8 text-gray-400" />
                <div className="text-center">
                  <p className="text-sm font-medium text-gray-700">
                    Drop a PDF here, or{" "}
                    <span className="text-gray-900 underline">browse</span>
                  </p>
                  <p className="text-xs text-gray-400 mt-1">PDF · Max 10 MB</p>
                </div>
              </div>
              <input
                ref={inputRef}
                type="file"
                accept=".pdf"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleFile(f);
                }}
              />

              {uploadState === "error" && (
                <div className="mt-3 flex items-center gap-2 text-sm text-red-600 bg-red-50 rounded-lg px-3 py-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  {error}
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        {uploadState === "idle" || uploadState === "error" ? (
          <div className="px-6 pb-4 flex justify-end gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
            >
              Cancel
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
