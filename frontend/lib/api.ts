const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const api = {
  // Tokenizer
  trainBPE: (text: string, vocab_size: number) =>
    apiFetch("/api/tokenizer/bpe/train", {
      method: "POST",
      body: JSON.stringify({ text, vocab_size }),
    }),
  encodeBPE: (text: string) =>
    apiFetch("/api/tokenizer/bpe/encode", { method: "POST", body: JSON.stringify({ text }) }),
  encodeTiktoken: (text: string) =>
    apiFetch("/api/tokenizer/tiktoken/encode", { method: "POST", body: JSON.stringify({ text }) }),
  compareTokenizers: (text: string, bpe_vocab_size: number) =>
    apiFetch("/api/tokenizer/compare", {
      method: "POST",
      body: JSON.stringify({ text, bpe_vocab_size }),
    }),

  // Architecture
  buildModel: (config: object) =>
    apiFetch("/api/architecture/build", { method: "POST", body: JSON.stringify(config) }),
  getPresets: () => apiFetch("/api/architecture/presets"),
  validateConfig: (config: object) =>
    apiFetch("/api/architecture/validate", { method: "POST", body: JSON.stringify(config) }),

  // Attention
  simpleAttention: (tokens: string[], d_model: number, n_heads: number) =>
    apiFetch("/api/attention/simple", {
      method: "POST",
      body: JSON.stringify({ tokens, d_model, n_heads }),
    }),
  causalAttention: (tokens: string[], d_model: number, n_heads: number) =>
    apiFetch("/api/attention/causal", {
      method: "POST",
      body: JSON.stringify({ tokens, d_model, n_heads }),
    }),
  multiheadAttention: (tokens: string[], d_model: number, n_heads: number) =>
    apiFetch("/api/attention/multihead", {
      method: "POST",
      body: JSON.stringify({ tokens, d_model, n_heads }),
    }),

  // Training
  startTraining: (config: object) =>
    apiFetch("/api/training/start", { method: "POST", body: JSON.stringify(config) }),
  cancelTraining: (run_id: string) =>
    apiFetch(`/api/training/cancel/${run_id}`, { method: "POST" }),
  getTrainingStatus: (run_id: string) => apiFetch(`/api/training/status/${run_id}`),
  getTrainingPresets: () => apiFetch("/api/training/presets"),

  // Chat
  formatPreview: (instruction: string, input_text: string) =>
    apiFetch("/api/chat/format-preview", {
      method: "POST",
      body: JSON.stringify({ instruction, input_text }),
    }),
  getInstructionSamples: (limit = 20, offset = 0) =>
    apiFetch(`/api/chat/instruction-data/sample?limit=${limit}&offset=${offset}`),
};

export const WS_BASE = API_BASE.replace(/^http/, "ws");
