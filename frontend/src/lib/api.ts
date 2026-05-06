const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? "/api").replace(/\/$/, "");

interface ChatResponse {
  answer: string;
}

interface UploadResponse {
  status: string;
}

async function jsonFetch<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(text || `Request failed with status ${response.status}`);
  }
  return (await response.json()) as T;
}

export async function sendChatMessage(text: string, signal?: AbortSignal): Promise<string> {
  const data = await jsonFetch<ChatResponse>(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });
  return data.answer;
}

export async function uploadDocument(file: File, signal?: AbortSignal): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);

  const data = await jsonFetch<UploadResponse>(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
    signal,
  });
  return data.status;
}

export async function checkHealth(signal?: AbortSignal): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`, { signal });
    return response.ok;
  } catch {
    return false;
  }
}
