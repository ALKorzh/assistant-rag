import { CheckCircle2, FileUp, Loader2, Trash2, XCircle } from "lucide-react";
import { ChangeEvent, useRef, useState } from "react";

import { uploadDocument } from "../lib/api";

interface Props {
  online: boolean;
  onClearChat: () => void;
}

type UploadStatus =
  | { kind: "idle" }
  | { kind: "uploading"; name: string }
  | { kind: "success"; message: string }
  | { kind: "error"; message: string };

export function Sidebar({ online, onClearChat }: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [status, setStatus] = useState<UploadStatus>({ kind: "idle" });

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    event.target.value = "";

    setStatus({ kind: "uploading", name: file.name });
    try {
      const message = await uploadDocument(file);
      setStatus({ kind: "success", message });
    } catch (error) {
      const message = error instanceof Error ? error.message : "Ошибка загрузки";
      setStatus({ kind: "error", message });
    }
  };

  return (
    <aside className="flex h-full w-72 flex-none flex-col border-r border-slate-800 bg-slate-950/80 p-5">
      <div className="mb-6">
        <div className="flex items-center gap-2 text-lg font-semibold text-slate-100">
          <span className="inline-flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600/20 text-brand-500 ring-1 ring-brand-500/40">
            🤖
          </span>
          Assistant RAG
        </div>
        <p className="mt-1 text-xs text-slate-400">
          Личный помощник с маршрутизацией, RAG и внешними инструментами.
        </p>
      </div>

      <div className="mb-6 flex items-center gap-2 rounded-lg bg-slate-900/60 px-3 py-2 text-xs ring-1 ring-slate-800">
        <span
          className={`h-2 w-2 rounded-full ${
            online ? "bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]" : "bg-rose-500"
          }`}
        />
        <span className="text-slate-300">{online ? "Сервер доступен" : "Сервер недоступен"}</span>
      </div>

      <section className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          База знаний
        </h3>
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="flex w-full items-center gap-2 rounded-xl border border-dashed border-slate-700 bg-slate-900/40 px-3 py-3 text-sm text-slate-200 transition hover:border-brand-500 hover:bg-brand-500/5"
        >
          <FileUp size={16} className="text-brand-500" />
          Загрузить в базу знаний
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.webp"
          className="hidden"
          onChange={handleFileChange}
        />

        {status.kind === "uploading" && (
          <div className="flex items-center gap-2 text-xs text-slate-300">
            <Loader2 size={14} className="animate-spin text-brand-500" />
            Индексирую {status.name}…
          </div>
        )}
        {status.kind === "success" && (
          <div className="flex items-start gap-2 rounded-lg bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200 ring-1 ring-emerald-500/30">
            <CheckCircle2 size={14} className="mt-0.5 flex-none" />
            <span>{status.message}</span>
          </div>
        )}
        {status.kind === "error" && (
          <div className="flex items-start gap-2 rounded-lg bg-rose-500/10 px-3 py-2 text-xs text-rose-200 ring-1 ring-rose-500/30">
            <XCircle size={14} className="mt-0.5 flex-none" />
            <span>{status.message}</span>
          </div>
        )}
      </section>

      <div className="mt-auto space-y-3 pt-6">
        <button
          type="button"
          onClick={onClearChat}
          className="flex w-full items-center gap-2 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2 text-xs text-slate-300 transition hover:border-rose-500/40 hover:text-rose-300"
        >
          <Trash2 size={14} />
          Очистить диалог
        </button>
        <p className="text-[11px] leading-relaxed text-slate-500">
          PDF, текст (TXT/MD), изображения с текстом (PNG/JPG/WebP) идут в Qdrant. Спрашивайте «что в моём файле?» —
          ответ пойдёт из ваших документов. Также: погода, Wikipedia, YouTube, веб, калькулятор. Ответы — Gemini 2.5
          Flash.
        </p>
      </div>
    </aside>
  );
}
