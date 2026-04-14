"use client";
import { useState } from "react";
import { api } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";

const DEFAULT_TOKENS = ["The", " quick", " brown", " fox", " jumps"];
const ATTENTION_TYPES = ["simple", "causal", "multihead"] as const;
type AttentionType = typeof ATTENTION_TYPES[number];

function AttentionHeatmap({ tokens, weights, numHeads = 1, activeHead = 0 }: {
  tokens: string[];
  weights: number[][] | number[][][];
  numHeads?: number;
  activeHead?: number;
}) {
  const matrix: number[][] = numHeads > 1
    ? (weights as number[][][])[activeHead]
    : (weights as number[][]);

  if (!matrix || !matrix.length) return null;

  const max = Math.max(...matrix.flat());
  const min = Math.min(...matrix.flat().filter(v => v > -1000));

  return (
    <div className="overflow-auto">
      <table className="text-xs font-mono border-collapse">
        <thead>
          <tr>
            <th className="w-20 text-right pr-2 text-gray-500">Q\K →</th>
            {tokens.map((t, i) => (
              <th key={i} className="px-1 text-gray-400 font-normal max-w-[60px] truncate" title={t}>
                {t.trim().slice(0, 8)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="text-right pr-2 text-gray-400 truncate max-w-[80px]">{tokens[i]?.trim().slice(0, 8)}</td>
              {row.map((val, j) => {
                const isMasked = val < -100;
                const intensity = isMasked ? 0 : (val - min) / (max - min + 1e-8);
                const r = Math.round(intensity * 59 + (1 - intensity) * 17);
                const g = Math.round(intensity * 130 + (1 - intensity) * 24);
                const b = Math.round(intensity * 246 + (1 - intensity) * 39);
                const bg = isMasked ? "rgb(40,20,20)" : `rgb(${r},${g},${b})`;
                return (
                  <td
                    key={j}
                    className="w-12 h-10 text-center rounded border border-gray-800/50 cursor-default"
                    style={{ backgroundColor: bg, color: intensity > 0.5 ? "#fff" : "#aaa" }}
                    title={isMasked ? "masked" : val.toFixed(3)}
                  >
                    {isMasked ? "×" : val.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function AttentionPage() {
  const [tokenInput, setTokenInput] = useState(DEFAULT_TOKENS.join(", "));
  const [dModel, setDModel] = useState(64);
  const [nHeads, setNHeads] = useState(4);
  const [activeHead, setActiveHead] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<Record<AttentionType, any>>({
    simple: null, causal: null, multihead: null,
  });
  const [activeType, setActiveType] = useState<AttentionType>("simple");

  const tokens = tokenInput.split(",").map((t) => t.trim()).filter(Boolean);

  async function runAttention(type: AttentionType) {
    if (tokens.length < 2) { setError("至少需要 2 个 token"); return; }
    setLoading(true); setError("");
    try {
      let result;
      if (type === "simple") result = await api.simpleAttention(tokens, dModel, nHeads);
      else if (type === "causal") result = await api.causalAttention(tokens, dModel, nHeads);
      else result = await api.multiheadAttention(tokens, dModel, nHeads);
      setResults((prev) => ({ ...prev, [type]: result }));
      setActiveType(type);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  async function runAll() {
    if (tokens.length < 2) { setError("至少需要 2 个 token"); return; }
    setLoading(true); setError("");
    try {
      const [s, c, m] = await Promise.all([
        api.simpleAttention(tokens, dModel, nHeads),
        api.causalAttention(tokens, dModel, nHeads),
        api.multiheadAttention(tokens, dModel, nHeads),
      ]);
      setResults({ simple: s, causal: c, multihead: m });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const currentResult = results[activeType];

  return (
    <div className="p-6 max-w-6xl space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-white mb-1">👁️ 注意力可视化</h1>
        <p className="text-gray-400 text-sm">
          第 3 章 · 交互式注意力权重热力图，直观理解自注意力、因果掩码和多头注意力。
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Config */}
        <Card className="bg-gray-900 border-gray-700">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm text-gray-300">配置</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-xs text-gray-400 block mb-1">Token 列表（逗号分隔）</label>
              <input
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
                className="w-full bg-gray-800 text-gray-200 text-sm rounded px-3 py-2 border border-gray-700 focus:border-blue-500 focus:outline-none"
                placeholder="The, quick, brown, fox"
              />
              <div className="text-xs text-gray-600 mt-1">{tokens.length} tokens</div>
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-gray-400">模型维度 (d_model)</span>
                <span className="text-blue-400 font-mono">{dModel}</span>
              </div>
              <Slider value={[dModel]} onValueChange={(v) => { const n = Array.isArray(v) ? v[0] : v; setDModel(n); setActiveHead(0); }}
                min={16} max={128} step={16} className="[&_[role=slider]]:bg-blue-500" />
            </div>

            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-gray-400">注意力头数 (n_heads)</span>
                <span className="text-blue-400 font-mono">{nHeads}</span>
              </div>
              <Slider value={[nHeads]} onValueChange={(v) => { const n = Array.isArray(v) ? v[0] : v; setNHeads(n); setActiveHead(0); }}
                min={1} max={8} step={1} className="[&_[role=slider]]:bg-blue-500" />
            </div>

            {dModel % nHeads !== 0 && (
              <div className="text-xs text-red-400 bg-red-500/10 rounded p-2">
                ⚠️ d_model 必须能被 n_heads 整除
              </div>
            )}

            {error && <p className="text-red-400 text-xs">{error}</p>}

            <button
              onClick={runAll}
              disabled={loading || dModel % nHeads !== 0}
              className="w-full py-2 rounded bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm font-medium"
            >
              {loading ? "计算中..." : "运行全部注意力类型"}
            </button>
          </CardContent>
        </Card>

        {/* Heatmap */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex gap-2 flex-wrap">
            {(["simple", "causal", "multihead"] as AttentionType[]).map((type) => {
              const labels = { simple: "自注意力 (v1)", causal: "因果注意力", multihead: "多头注意力" };
              return (
                <button
                  key={type}
                  onClick={() => { setActiveType(type); if (!results[type]) runAttention(type); }}
                  className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
                    activeType === type
                      ? "bg-blue-600 text-white"
                      : "bg-gray-800 text-gray-300 hover:bg-gray-700"
                  }`}
                >
                  {labels[type]}
                  {results[type] && <span className="ml-1.5 text-green-400">✓</span>}
                </button>
              );
            })}
          </div>

          {/* Explanation */}
          <Card className="bg-gray-900 border-gray-700">
            <CardContent className="pt-4 pb-3">
              <div className="text-xs text-gray-400 space-y-1">
                {activeType === "simple" && (
                  <p>🔹 <strong>自注意力 v1</strong>：每个 token 都可以看到所有其他 token，权重通过 Q·K^T / √d_k 计算。</p>
                )}
                {activeType === "causal" && (
                  <p>🔹 <strong>因果注意力</strong>：上三角部分被掩码（×），每个位置只能看到它之前的 token，用于自回归生成。</p>
                )}
                {activeType === "multihead" && (
                  <p>🔹 <strong>多头注意力</strong>：将 d_model 分成 {nHeads} 个头（每头 {dModel / nHeads} 维），每个头学习不同的注意力模式。</p>
                )}
              </div>
            </CardContent>
          </Card>

          {currentResult ? (
            <Card className="bg-gray-900 border-gray-700">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm text-gray-300 flex items-center justify-between">
                  <span>注意力权重热力图</span>
                  {activeType === "multihead" && currentResult.num_heads > 1 && (
                    <div className="flex gap-1">
                      {Array.from({ length: currentResult.num_heads }).map((_, h) => (
                        <button
                          key={h}
                          onClick={() => setActiveHead(h)}
                          className={`w-7 h-7 rounded text-xs font-mono ${h === activeHead ? "bg-blue-600" : "bg-gray-700 hover:bg-gray-600"}`}
                        >
                          {h}
                        </button>
                      ))}
                    </div>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <AttentionHeatmap
                  tokens={currentResult.tokens}
                  weights={currentResult.weights}
                  numHeads={activeType === "multihead" ? currentResult.num_heads : 1}
                  activeHead={activeHead}
                />
                <div className="flex items-center gap-3 mt-3 text-xs text-gray-500">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: "rgb(17,24,39)" }} />
                    <span>低注意力</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 rounded" style={{ backgroundColor: "rgb(59,130,246)" }} />
                    <span>高注意力</span>
                  </div>
                  {currentResult.is_masked && (
                    <div className="flex items-center gap-1">
                      <span className="text-red-400">× 因果掩码</span>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="bg-gray-900 border-gray-700 min-h-48 flex items-center justify-center">
              <p className="text-gray-600 text-sm">点击上方按钮计算注意力权重</p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
