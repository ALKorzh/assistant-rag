import { Bot, User } from "lucide-react";
import ReactMarkdown from "react-markdown";

import type { ChatMessage } from "../types";

interface Props {
  message: ChatMessage;
}

export function ChatBubble({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div
      className={`flex w-full gap-3 animate-fade-in ${
        isUser ? "justify-end" : "justify-start"
      }`}
    >
      {!isUser && (
        <div className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-brand-600/20 text-brand-500 ring-1 ring-brand-500/30">
          <Bot size={18} />
        </div>
      )}

      <div
        className={`max-w-[78%] rounded-2xl px-4 py-3 text-sm leading-relaxed shadow-sm ring-1 ${
          isUser
            ? "bg-brand-600 text-white ring-brand-700/40"
            : "bg-slate-900/80 text-slate-100 ring-slate-800"
        }`}
      >
        <div className="markdown break-words">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
      </div>

      {isUser && (
        <div className="flex h-9 w-9 flex-none items-center justify-center rounded-full bg-slate-800 text-slate-300 ring-1 ring-slate-700">
          <User size={18} />
        </div>
      )}
    </div>
  );
}
