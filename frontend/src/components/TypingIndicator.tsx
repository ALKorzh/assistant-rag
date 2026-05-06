import { Bot } from "lucide-react";

export function TypingIndicator() {
  return (
    <div className="flex items-center gap-3 animate-fade-in">
      <div className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-brand-600/20 text-brand-500 ring-1 ring-brand-500/30">
        <Bot size={18} />
      </div>
      <div className="flex items-center gap-1.5 rounded-2xl bg-slate-900/80 px-4 py-3 ring-1 ring-slate-800">
        <span className="h-2 w-2 rounded-full bg-slate-400 animate-pulse-soft" />
        <span
          className="h-2 w-2 rounded-full bg-slate-400 animate-pulse-soft"
          style={{ animationDelay: "150ms" }}
        />
        <span
          className="h-2 w-2 rounded-full bg-slate-400 animate-pulse-soft"
          style={{ animationDelay: "300ms" }}
        />
      </div>
    </div>
  );
}
