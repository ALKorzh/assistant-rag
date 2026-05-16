import { Sparkles } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { ChatBubble } from "./components/ChatBubble";
import { ChatInput } from "./components/ChatInput";
import { Sidebar } from "./components/Sidebar";
import { TypingIndicator } from "./components/TypingIndicator";
import { checkHealth, sendChatMessage } from "./lib/api";
import type { ChatMessage } from "./types";

const SUGGESTIONS = [
  "Расскажи кратко, как работает агент RAG",
  "Какая сейчас погода в Минске?",
  "Найди видео по LangGraph на YouTube",
  "Что такое langchain в одном абзаце?",
];

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [online, setOnline] = useState(false);
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      const ok = await checkHealth();
      if (!cancelled) setOnline(ok);
    };
    ping();
    const interval = window.setInterval(ping, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    scrollerRef.current?.scrollTo({
      top: scrollerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, pending]);

  const handleSubmit = async (text: string) => {
    const userMessage: ChatMessage = {
      id: makeId(),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setPending(true);
    setError(null);

    try {
      const answer = await sendChatMessage(text);
      const assistantMessage: ChatMessage = {
        id: makeId(),
        role: "assistant",
        content: answer,
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Не удалось получить ответ";
      setError(message);
    } finally {
      setPending(false);
    }
  };

  const isEmpty = useMemo(() => messages.length === 0, [messages]);

  return (
    <div className="flex h-screen w-full bg-gradient-to-br from-slate-950 via-slate-950 to-slate-900 font-sans">
      <Sidebar online={online} onClearChat={() => setMessages([])} />

      <main className="flex h-full flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-slate-800/80 bg-slate-950/60 px-6 py-4 backdrop-blur">
          <div>
            <h1 className="text-lg font-semibold text-slate-50">Чат с ассистентом</h1>
            <p className="text-xs text-slate-400">
              Gemini 3.1 Flash-Lite · LangGraph · Qdrant · DuckDuckGo · Wikipedia · YouTube
            </p>
          </div>
          <div className="hidden items-center gap-2 rounded-full bg-brand-500/10 px-3 py-1 text-xs font-medium text-brand-500 ring-1 ring-brand-500/30 md:inline-flex">
            <Sparkles size={14} />
            Agentic RAG
          </div>
        </header>

        <div
          ref={scrollerRef}
          className="flex-1 overflow-y-auto scrollbar-thin"
        >
          <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 px-4 py-6">
            {isEmpty && !pending && (
              <div className="rounded-2xl border border-slate-800 bg-slate-900/40 p-6 shadow-inner">
                <h2 className="text-base font-semibold text-slate-100">Начните диалог</h2>
                <p className="mt-1 text-sm text-slate-400">
                  Задайте вопрос про ваши документы, попросите найти видео, узнайте погоду
                  или просто поговорите.
                </p>
                <div className="mt-4 grid gap-2 sm:grid-cols-2">
                  {SUGGESTIONS.map((suggestion) => (
                    <button
                      key={suggestion}
                      type="button"
                      onClick={() => handleSubmit(suggestion)}
                      className="rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-2 text-left text-sm text-slate-300 transition hover:border-brand-500/40 hover:bg-brand-500/5 hover:text-slate-100"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((message) => (
              <ChatBubble key={message.id} message={message} />
            ))}

            {pending && <TypingIndicator />}

            {error && (
              <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                {error}
              </div>
            )}
          </div>
        </div>

        <footer className="border-t border-slate-800/80 bg-slate-950/60 px-4 py-4 backdrop-blur">
          <div className="mx-auto w-full max-w-3xl">
            <ChatInput onSubmit={handleSubmit} disabled={pending} />
            <p className="mt-2 text-[11px] text-slate-500">
              Ассистент может ошибаться — проверяйте важные факты.
            </p>
          </div>
        </footer>
      </main>
    </div>
  );
}
